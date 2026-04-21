# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch
from forge.controller import ForgeActor
from forge.forge_engine_config import (
    ForgeModelIdentity,
    forge_engine_config_for_rl_reference,
)
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl.loss import compute_logprobs, create_shifted_targets
from monarch.actor import current_rank, current_size, endpoint
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.forge.engine import ForgeEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ReferenceModel(ForgeActor):
    """Frozen reference policy on `ForgeEngine` for RL (e.g. logprobs / KL)."""

    dump_folder: str = "."
    model: ForgeModelIdentity = field(default_factory=ForgeModelIdentity)
    optimizer: OptimizersContainer.Config = field(
        default_factory=OptimizersContainer.Config
    )
    lr_scheduler: LRSchedulersContainer.Config = field(
        default_factory=LRSchedulersContainer.Config
    )
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    checkpoint: CheckpointManager.Config = field(
        default_factory=CheckpointManager.Config
    )
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        super().__init__()

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 0
        self.rank = current_rank().rank
        self.size = math.prod(current_size().values())

        self.compute_logprobs = compute_logprobs
        if self.compile.enable:
            self.compute_logprobs = torch.compile(self.compute_logprobs)

        env = {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.size),
            "GROUP_RANK": str(self.size),
            "GROUP_WORLD_SIZE": str(self.size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)

    @endpoint
    async def setup(self):
        self.engine = ForgeEngine(
            forge_engine_config_for_rl_reference(
                dump_folder=self.dump_folder,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                training=self.training,
                parallelism=self.parallelism,
                checkpoint=self.checkpoint,
                activation_checkpoint=self.activation_checkpoint,
                compile=self.compile,
                comm=self.comm,
                debug=self.debug,
            )
        )
        self.engine.checkpointer.load(step=self.step)
        self.model = self.engine.model_parts[0]
        self.model.eval()

    @endpoint
    async def forward(
        self, input_ids: torch.Tensor, return_logprobs: bool = True
    ) -> torch.Tensor:
        record_metric("reference_perf/forward/count_forward_passes", 1, Reduce.SUM)

        t = Tracer("reference_perf/forward", timer="gpu", track_memory=True)
        t.start()
        self.engine.gc_handler.run(self.step)

        model_parts = self.engine.model_parts
        input_ids = input_ids.to(self.engine.device)

        if self.engine.parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            with self.engine.train_context():
                with torch.inference_mode():
                    logits = model_parts[0](input_ids)
                    if return_logprobs:
                        target_ids = create_shifted_targets(input_ids)
                        out, _ = self.compute_logprobs(logits, target_ids)
                    else:
                        out = logits

        if isinstance(out, DTensor):
            out = out.full_tensor()

        self.step += 1
        t.stop()
        return out

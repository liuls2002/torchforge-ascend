# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torchstore as ts
from forge.actors._torchstore_utils import get_param_key
from forge.api.trainer import ParallelismConfig as ForgeParallelismSnapshot
from forge.api.trainer import TrainerConfig, TrainerStatus
from forge.controller import ForgeActor
from forge.data.utils import batch_to_device
from forge.forge_engine_config import ForgeModelIdentity, forge_engine_config_for_rl_trainer
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl.loss import create_shifted_targets
from forge.types import TrainBatch
from monarch.actor import endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
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
logger.setLevel(logging.DEBUG)


@dataclass
class TitanTrainer(ForgeActor):
    """RL trainer actor on current `ForgeEngine` (torchtitan)."""

    dump_folder: str = "."
    model: ForgeModelIdentity = field(default_factory=ForgeModelIdentity)
    optimizer: OptimizersContainer.Config = field(
        default_factory=OptimizersContainer.Config
    )
    lr_scheduler: LRSchedulersContainer.Config = field(
        default_factory=LRSchedulersContainer.Config
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
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
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"

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

        self.step = 1  # fragile contract.
        self.num_training_steps = self.training.steps
        self._accumulated_microbatches = 0
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)

    @staticmethod
    def _slice_train_batch(batch: TrainBatch, start: int, end: int) -> TrainBatch:
        def _slice_map(d: dict) -> dict:
            out: dict = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    out[k] = v[start:end].contiguous()
                else:
                    out[k] = v
            return out

        return TrainBatch(
            model_inputs=_slice_map(batch.model_inputs),
            loss_inputs=_slice_map(batch.loss_inputs),
            meta=dict(batch.meta),
        )

    def _prepare_fsdp_microbatch(self, micro_idx: int, num_micro: int) -> None:
        """Match FSDP2 ``fully_shard`` gradient accumulation (see ``set_requires_gradient_sync``).

        Without this, each ``backward()`` may reduce gradients immediately; with multiple
        microbatches per optimizer step that can corrupt FSDP state and surface as illegal
        CUDA access or NCCL watchdog errors.
        """
        if num_micro <= 1:
            return
        root = self.engine.model_parts[0]
        is_last = micro_idx == num_micro - 1
        if hasattr(root, "set_requires_gradient_sync"):
            root.set_requires_gradient_sync(is_last, recurse=True)
        if hasattr(root, "set_is_last_backward"):
            root.set_is_last_backward(is_last)

    @endpoint
    async def setup(self):
        self.engine = ForgeEngine(
            forge_engine_config_for_rl_trainer(
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
        self.engine.optimizers.zero_grad()

    def forward_backward(
        self,
        batch: TrainBatch,
        *,
        gradient_accumulation_divisor: float = 1.0,
    ) -> Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims

        batch.loss_inputs["target_ids"] = create_shifted_targets(
            batch.model_inputs["tokens"], batch.loss_inputs.get("loss_mask")
        )

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            with self.engine.train_context():
                assert len(model_parts) == 1
                logits = model_parts[0](**batch.model_inputs)
                loss_output = self.loss(logits, **batch.loss_inputs)
                loss = loss_output.loss

                for metric in loss_output.metrics:
                    value = (
                        metric.value.item()
                        if isinstance(metric.value, torch.Tensor)
                        else metric.value
                    )
                    record_metric(metric.key, value, metric.reduction, metric.timestamp)

                del logits, loss_output.metrics
                scaled = loss / gradient_accumulation_divisor
                scaled.backward()
        self._accumulated_microbatches += 1
        return loss

    @endpoint
    async def train_step(self, batches: list[TrainBatch]) -> float:
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        self.engine.gc_handler.run(self.step)
        batch = batches[self.engine.dp_rank]
        batch_to_device(batch.model_inputs, self.engine.device)
        batch_to_device(batch.loss_inputs, self.engine.device)

        ga = self.engine.gradient_accumulation_steps
        micro_bs = self.training.local_batch_size
        tokens = batch.model_inputs["tokens"]
        expected = micro_bs * ga
        if tokens.shape[0] != expected:
            raise RuntimeError(
                f"TrainBatch token batch dim is {tokens.shape[0]}, expected "
                f"{expected} (= local_batch_size {micro_bs} * "
                f"gradient_accumulation_steps {ga}). "
                "Align replay_buffer (global_batch_size / batch_size / dp_size) with "
                "trainer.training.global_batch_size and ForgeEngine."
            )

        micro_losses: list[Tensor] = []
        for m in range(ga):
            self._prepare_fsdp_microbatch(m, ga)
            start, end = m * micro_bs, (m + 1) * micro_bs
            micro = self._slice_train_batch(batch, start, end)
            loss_micro = self.forward_backward(
                micro,
                gradient_accumulation_divisor=float(ga),
            )
            micro_losses.append(loss_micro.detach())

        loss_for_stats = torch.stack(micro_losses).mean()
        if torch.distributed.is_initialized():
            if hasattr(torch.distributed.ReduceOp, "AVG"):
                torch.distributed.all_reduce(
                    loss_for_stats, op=torch.distributed.ReduceOp.AVG
                )
            else:
                torch.distributed.all_reduce(
                    loss_for_stats, op=torch.distributed.ReduceOp.SUM
                )
                ws = float(torch.distributed.get_world_size())
                loss_for_stats = loss_for_stats / ws

        t.step("forward_backward")

        current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()
        self._accumulated_microbatches = 0
        self.step += 1
        t.step("optimizer_step")

        loss = float(loss_for_stats.item())
        record_metric("rl_trainer/loss", loss, Reduce.MEAN)

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()
        return loss

    @endpoint
    async def get_config(self) -> TrainerConfig:
        parallel_dims = self.engine.parallel_dims
        tp_rank = (
            parallel_dims.get_mesh("tp").get_local_rank()
            if parallel_dims.tp_enabled
            else 0
        )
        parallelism = ForgeParallelismSnapshot(
            dp_degree=parallel_dims.dp_shard * parallel_dims.dp_replicate,
            tp_degree=parallel_dims.tp,
            pp_degree=parallel_dims.pp,
            cp_degree=parallel_dims.cp,
            ep_degree=parallel_dims.ep,
            world_size=parallel_dims.world_size,
            dp_rank=self.engine.dp_rank,
            tp_rank=tp_rank,
            device=str(self.engine.device),
        )
        return TrainerConfig(
            model_name=self.model.name,
            model_config=self.model.as_dict(),
            parallelism=parallelism,
        )

    @endpoint
    async def get_status(self) -> TrainerStatus:
        return TrainerStatus(
            step=self.step,
            accumulated_microbatches=self._accumulated_microbatches,
        )

    @endpoint
    async def clear_gradients(self) -> None:
        self.engine.optimizers.zero_grad()
        self._accumulated_microbatches = 0

    @endpoint
    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        if name is not None:
            raise NotImplementedError(
                "TitanTrainer uses step-based checkpoint naming; custom names are not supported"
            )
        if path is not None:
            raise NotImplementedError(
                "TitanTrainer uses the checkpoint.folder from config; custom paths are not supported"
            )
        if weights_only:
            raise NotImplementedError(
                "weights_only is not supported; TitanTrainer always saves full training state"
            )

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=False,
        )
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def load(self, path: str | None = None) -> str:
        if path is not None:
            raise NotImplementedError(
                "TitanTrainer uses the checkpoint.folder from config; custom paths are not supported"
            )

        self.engine.checkpointer.load(step=self.step)
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)

        entries: dict[str, torch.Tensor] = {
            get_param_key(policy_version, name): param
            for name, param in hf_state_dict.items()
        }
        if not entries:
            raise RuntimeError(
                "sd_adapter.to_hf() produced an empty state dict; cannot push to torchstore. "
                "Check checkpoint sd_adapter and model/flattened_state_dict mapping."
            )
        await ts.put_batch(entries)
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()

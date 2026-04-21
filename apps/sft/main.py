# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:

python -m apps.sft.main --config apps/sft/llama3_8b.yaml

"""

import asyncio
import contextlib
import logging
import math
import os
import sys
from typing import Any

import torch
from forge.controller import ForgeActor
from forge.forge_engine_config import forge_engine_config_from_sft_yaml
from forge.data.collate import collate_padded
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from forge.data.utils import StopAfterOneEpoch
from forge.observability import get_or_create_metric_logger, record_metric, Reduce
from forge.util.config import parse
from monarch.actor import current_rank, current_size, endpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.forge.engine import ForgeEngine

Checkpointer = Any
Dataloader = Any
MetricLogger = Any
Profiler = Any
Tokenizer = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ForgeSFTRecipe(ForgeActor, ForgeEngine):
    """SFT recipe on current `ForgeEngine` with forge HF iterable dataloader."""

    train_spec: Any
    parallel_dims: ParallelDims
    model: list[nn.Module]
    loss_fn: LossFunction
    optimizer: OptimizersContainer
    lr_scheduler: LRSchedulersContainer
    checkpointer: Checkpointer
    tokenizer: Tokenizer
    train_dataloader: Dataloader
    metric_logger: MetricLogger
    profiler: Profiler
    device: torch.device
    step: int

    def __init__(self, config: DictConfig):
        self._sft_cfg = config
        engine_config = forge_engine_config_from_sft_yaml(config)
        self.current_step = 0
        self.num_training_steps = engine_config.training.steps
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())
        super().__init__(engine_config)

    def state_dict(self) -> dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.current_step = int(state_dict.get("current_step", 0))

    async def setup_metric_logger(self):
        mlogger = await get_or_create_metric_logger()
        return mlogger

    def record_batch_metrics(self, data_metrics: list):
        for metric in data_metrics:
            record_metric(metric.key, metric.value, metric.reduction)

    def _normalize_batch_keys(self, batch: dict) -> None:
        if "tokens" in batch and "input" not in batch:
            batch["input"] = batch["tokens"]

    @endpoint
    async def setup(self):
        if self.config.compile.enable:
            raise ValueError(
                "compile.enable=True is not supported in this SFT entry. "
                "Set compile.enable=false in apps/sft/*.yaml."
            )

        if self.parallel_dims.cp_enabled:
            raise NotImplementedError(
                "SFT entry does not yet wire context parallel the same way as "
                "`torchtitan.trainer.Trainer` (positions + prepare_context_parallel_input). "
                "Set parallelism.context_parallel_degree=1, or extend this recipe."
            )

        self.rank_should_record_loss = True
        if hasattr(self, "pp_has_last_stage") and not self.pp_has_last_stage:
            self.rank_should_record_loss = False

        self.mlogger = await self.setup_metric_logger()

        logger.info("Setting training datasets")
        train_datasets_config = OmegaConf.to_container(
            self._sft_cfg.training.datasets, resolve=True
        )
        assert isinstance(train_datasets_config, list)
        self.train_dataloader = self.setup_data(train_datasets_config)

        eval_config = self._sft_cfg["eval"]
        self.val_dataloaders = {}
        self.eval_every_n_steps = eval_config["eval_every_n_steps"]
        max_eval_steps = eval_config["max_eval_steps"]
        self.max_eval_steps = (
            max_eval_steps if max_eval_steps and max_eval_steps > 0 else None
        )
        self.validation_enabled = (
            self.eval_every_n_steps is not None and self.eval_every_n_steps > 0
        )
        if self.validation_enabled:
            logger.info("Setting eval datasets")
            self.eval_datasets_config = eval_config.datasets

            for i, dataset_config in enumerate(self.eval_datasets_config):
                ds_name = dataset_config.get("dataset_name", i)
                dataloader = self.setup_data([dataset_config])
                self.val_dataloaders[ds_name] = dataloader

        self.checkpointer.load(step=self.current_step)

    def setup_data(self, dataset_configs: list[dict]) -> StatefulDataLoader:
        if len(dataset_configs) > 1:
            raise ValueError(
                f"Multiple training datasets not supported yet. "
                f"Got {len(dataset_configs)} datasets. "
            )

        dataset_config = dataset_configs[0]

        hf_path = self.config.hf_assets_path
        tokenizer = HuggingFaceModelTokenizer(
            tokenizer_json_path=os.path.join(hf_path, "tokenizer.json"),
            tokenizer_config_json_path=os.path.join(hf_path, "tokenizer_config.json"),
            generation_config_path=os.path.join(hf_path, "generation_config.json"),
            chat_template_path=(
                path
                if os.path.exists(
                    path := os.path.join(hf_path, "chat_template.jinja")
                )
                else None
            ),
            max_seq_len=self.config.training.seq_len,
        )

        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.get_mesh("batch").get_group()

        dataset = sft_iterable_dataset(
            model_transform=tokenizer,
            message_transform=AlpacaToMessages(),
            dp_mesh=dp_mesh,
            **dataset_config,
        )

        return StatefulDataLoader(
            dataset=dataset,
            batch_size=self.config.training.local_batch_size,
            collate_fn=collate_padded,
        )

    def forward_backward(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        skip_backward: bool = False,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        self._normalize_batch_keys(input_dict)
        inputs = input_dict["input"]

        if parallel_dims.pp_enabled:
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            loss = (
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor(-1.0, device=self.device)
            )

            if skip_backward:
                loss = loss.detach()

        else:
            with self.train_context():
                assert len(model_parts) == 1
                pred = model_parts[0](inputs)
                loss = self.loss_fn(pred, labels)
                del pred

                if not skip_backward:
                    loss.backward()

        return loss

    def train_step(self, batch) -> None:
        parallel_dims = self.parallel_dims
        labels = batch.pop("labels")
        loss = self.forward_backward(batch, labels)

        _ = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )

        if self.rank_should_record_loss:
            loss_val = loss.item()
            record_metric("ForgeSFTRecipe/train_step/loss", loss_val, Reduce.MEAN)
            logger.info(
                f"step {self.current_step} / {self.num_training_steps} | Loss: {loss_val}"
            )

        self.optimizers.step()
        self.lr_schedulers.step()

    async def evaluate(self) -> None:
        for model_part in self.model_parts:
            model_part.eval()

        dp_mesh = None
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.get_mesh("batch").get_group()

        maybe_no_grad = (
            contextlib.nullcontext()
            if self.parallel_dims.pp_enabled
            else torch.no_grad()
        )

        all_dataset_losses = []
        all_dataset_steps = []
        for dataset_name, val_dataloader in self.val_dataloaders.items():
            logger.info(f"=====Evaluating dataset: {dataset_name}=====")

            total_loss = torch.tensor(0.0, device=self.device)
            num_steps = 0

            batch_iter = StopAfterOneEpoch(
                iter=iter(val_dataloader),
                device=self.device,
                dp_mesh=dp_mesh,
            )

            with maybe_no_grad:
                for batch in batch_iter:
                    if (
                        self.max_eval_steps is not None
                        and num_steps >= self.max_eval_steps
                    ):
                        logger.info(
                            f"[{dataset_name}] Reached max_eval_steps cap of {self.max_eval_steps}"
                        )
                        break

                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)

                    labels = batch.pop("labels")
                    loss = self.forward_backward(batch, labels, skip_backward=True)
                    total_loss += loss
                    num_steps += 1

                    if self.rank_should_record_loss:
                        loss_val = loss.item()
                        logger.info(
                            f"[dataset {dataset_name}] Step {num_steps} | Loss: {loss_val:.4f}"
                        )

            avg_loss = (total_loss / max(num_steps, 1)).item()
            all_dataset_losses.append(avg_loss)
            all_dataset_steps.append(num_steps)
            logger.info(
                f"[dataset {dataset_name}] Final Step {num_steps} | Avg Loss: {avg_loss:.4f}"
            )
            if self.rank_should_record_loss:
                record_metric(
                    f"evaluate/dataset_{dataset_name}_avg_loss",
                    avg_loss,
                    Reduce.MEAN,
                )

        if self.rank_should_record_loss and len(all_dataset_losses) > 1:
            macro_avg_loss = sum(all_dataset_losses) / len(all_dataset_losses)
            record_metric("evaluate/macro_avg_loss", macro_avg_loss, Reduce.MEAN)

            total_steps = sum(all_dataset_steps)
            micro_avg_loss = (
                sum(
                    loss * steps
                    for loss, steps in zip(all_dataset_losses, all_dataset_steps)
                )
                / total_steps
            )
            record_metric("evaluate/micro_avg_loss", micro_avg_loss, Reduce.MEAN)

            logger.info(
                f"Macro avg loss (unweighted): {macro_avg_loss:.4f}, "
                f"Micro avg loss (weighted): {micro_avg_loss:.4f}"
            )

        for model_part in self.model_parts:
            model_part.train()

        logger.info("==Evaluation complete==")

    @endpoint
    async def train(self) -> None:
        dataloader = iter(self.train_dataloader)
        self.optimizers.zero_grad()

        while self.current_step < self.num_training_steps:
            batch = next(dataloader)

            self.record_batch_metrics(batch.pop("metrics", []))
            record_metric("ForgeSFTRecipe/train/step", self.current_step, Reduce.MEAN)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            self.train_step(batch)
            self.current_step += 1

            if (
                self.validation_enabled
                and self.current_step % self.eval_every_n_steps == 0
            ):
                await self.evaluate()

            self.checkpointer.save(
                curr_step=self.current_step,
                last_step=self.current_step == self.num_training_steps,
            )

            if self._rank == 0:
                await self.mlogger.flush.call_one(global_step=self.current_step)

        if self.validation_enabled:
            logger.info("Running final evaluation at end of training...")
            await self.evaluate()

    @endpoint
    async def cleanup(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if getattr(self, "mlogger", None):
            await self.mlogger.shutdown.call_one()

    def __repr__(self) -> str:
        return "Trainer"


async def run(cfg: DictConfig) -> None:
    logging.info("Spawning recipe...")
    process_cfg = cfg.pop("processes")

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    recipe = await ForgeSFTRecipe.options(**process_cfg).as_actor(cfg)

    logging.info("Created recipe, running setup.")
    await recipe.setup.call()

    logging.info("Recipe has been setup. Training now.")
    await recipe.train.call()

    logging.info("Done training. Clean up")
    await recipe.cleanup.call()

    await recipe.mesh.stop()
    logging.info("All done!")


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())

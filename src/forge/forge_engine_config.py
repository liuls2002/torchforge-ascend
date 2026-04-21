# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Construct torchtitan `ForgeEngine.Config` using public torchtitan APIs only."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from omegaconf import DictConfig, OmegaConf
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
from torchtitan.protocols.model_converter import ModelConvertersContainer


@dataclass
class ForgeModelIdentity:
    """Architecture id + HF asset root (matches YAML `model:` under trainer/ref_model)."""

    name: str = ""
    flavor: str = ""
    hf_assets_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_spec(arch: str, flavor: str):
    mod = importlib.import_module(f"torchtitan.models.{arch}")
    if not hasattr(mod, "model_registry"):
        raise ValueError(
            f"torchtitan.models.{arch} has no model_registry (cannot load {arch}/{flavor})"
        )
    return mod.model_registry(flavor)


def _field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _pick_dataclass(cls: type, raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    names = _field_names(cls)
    return {k: v for k, v in raw.items() if k in names}


def forge_engine_config_from_sft_yaml(cfg: DictConfig) -> ForgeEngine.Config:
    """`ForgeEngine.Config` from SFT OmegaConf (apps/sft/*.yaml + CLI)."""
    c: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    assert isinstance(c, dict)

    model_block = c.get("model") or {}
    if not model_block.get("name") or not model_block.get("flavor"):
        raise ValueError("YAML must set model.name and model.flavor for torchtitan")
    model_spec = load_model_spec(model_block["name"], model_block["flavor"])
    hf_assets_path = str(model_block.get("hf_assets_path", ""))

    dump_folder = str(c.get("dump_folder", "."))

    training_raw = dict(c.get("training") or {})
    compile_from_training = training_raw.pop("compile", None)
    training_raw.pop("datasets", None)
    training = TrainingConfig(**_pick_dataclass(TrainingConfig, training_raw))

    compile_raw = dict(c.get("compile") or {})
    if compile_from_training is not None:
        compile_raw["enable"] = bool(compile_from_training)
    compile_cfg = CompileConfig(**_pick_dataclass(CompileConfig, compile_raw))

    parallelism = ParallelismConfig(
        **_pick_dataclass(ParallelismConfig, c.get("parallelism") or {})
    )
    comm = CommConfig(**_pick_dataclass(CommConfig, c.get("comm") or {}))
    debug = DebugConfig(**_pick_dataclass(DebugConfig, c.get("debug") or {}))

    optimizer = OptimizersContainer.Config(
        **_pick_dataclass(OptimizersContainer.Config, c.get("optimizer") or {})
    )
    lr_scheduler = LRSchedulersContainer.Config(
        **_pick_dataclass(LRSchedulersContainer.Config, c.get("lr_scheduler") or {})
    )

    act_raw = dict(c.get("activation_checkpoint") or {})
    act_raw.pop("selective_ac_option", None)
    activation_checkpoint = ActivationCheckpointConfig(
        **_pick_dataclass(ActivationCheckpointConfig, act_raw)
    )

    ckpt_raw = dict(c.get("checkpoint") or {})
    checkpoint = CheckpointManager.Config(
        **_pick_dataclass(CheckpointManager.Config, ckpt_raw)
    )

    model_converters = ModelConvertersContainer.Config(
        **_pick_dataclass(
            ModelConvertersContainer.Config, c.get("model_converters") or {}
        )
    )

    return ForgeEngine.Config(
        hf_assets_path=hf_assets_path,
        dump_folder=dump_folder,
        model_spec=model_spec,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=training,
        parallelism=parallelism,
        checkpoint=checkpoint,
        activation_checkpoint=activation_checkpoint,
        compile=compile_cfg,
        model_converters=model_converters,
        comm=comm,
        debug=debug,
    )


def forge_engine_config_for_rl_trainer(
    *,
    dump_folder: str,
    model: ForgeModelIdentity,
    optimizer: OptimizersContainer.Config,
    lr_scheduler: LRSchedulersContainer.Config,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    checkpoint: CheckpointManager.Config,
    activation_checkpoint: ActivationCheckpointConfig,
    compile: CompileConfig,
    comm: CommConfig,
    debug: DebugConfig | None = None,
) -> ForgeEngine.Config:
    """`ForgeEngine.Config` for RL policy training (`TitanTrainer`)."""
    dbg = debug if debug is not None else DebugConfig()
    return ForgeEngine.Config(
        hf_assets_path=model.hf_assets_path,
        dump_folder=dump_folder,
        model_spec=load_model_spec(model.name, model.flavor),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=training,
        parallelism=parallelism,
        checkpoint=checkpoint,
        activation_checkpoint=activation_checkpoint,
        compile=compile,
        model_converters=ModelConvertersContainer.Config(),
        comm=comm,
        debug=dbg,
    )


def forge_engine_config_for_rl_reference(
    *,
    dump_folder: str,
    model: ForgeModelIdentity,
    optimizer: OptimizersContainer.Config,
    lr_scheduler: LRSchedulersContainer.Config,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    checkpoint: CheckpointManager.Config,
    activation_checkpoint: ActivationCheckpointConfig,
    compile: CompileConfig,
    comm: CommConfig,
    debug: DebugConfig | None = None,
) -> ForgeEngine.Config:
    """`ForgeEngine.Config` for frozen reference policy; forces load from `initial_load_path`."""
    dbg = debug if debug is not None else DebugConfig()
    ckpt = replace(checkpoint, folder="")
    return ForgeEngine.Config(
        hf_assets_path=model.hf_assets_path,
        dump_folder=dump_folder,
        model_spec=load_model_spec(model.name, model.flavor),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=training,
        parallelism=parallelism,
        checkpoint=ckpt,
        activation_checkpoint=activation_checkpoint,
        compile=compile,
        model_converters=ModelConvertersContainer.Config(),
        comm=comm,
        debug=dbg,
    )

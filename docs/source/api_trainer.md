# Trainer

```{eval-rst}
.. currentmodule:: forge.actors.trainer
```

The Trainer manages model training in TorchForge, built on top of TorchTitan.
It handles forward/backward passes, weight updates, and checkpoint management for reinforcement learning workflows.

## TitanTrainer

```{eval-rst}
.. autoclass:: TitanTrainer
   :members: train_step, push_weights, cleanup
   :exclude-members: __init__
```

## Configuration

`TitanTrainer` and `ReferenceModel` construct a TorchTitan
`ForgeEngine` from **`ForgeEngine.Config`**. In this repo, that config is built with
**public** TorchTitan dataclasses only (no legacy `job_config` module): see
`forge.forge_engine_config` (`ForgeModelIdentity`, `forge_engine_config_from_sft_yaml`,
`forge_engine_config_for_rl_trainer`, `forge_engine_config_for_rl_reference`).

YAML under `apps/sft/` and `apps/grpo/` maps into these fields: `dump_folder`,
`model` (`name`, `flavor`, `hf_assets_path`), `optimizer`, `lr_scheduler`, `training`,
`compile`, `parallelism`, `checkpoint`, `activation_checkpoint`, `comm`, and optional `debug`.

### Forge-side model identity

```{eval-rst}
.. autoclass:: forge.forge_engine_config.ForgeModelIdentity
   :members:
   :undoc-members:
```

### `ForgeEngine.Config` (engine bundle)

```{eval-rst}
.. autoclass:: torchtitan.experiments.forge.engine.ForgeEngine.Config
   :members:
   :undoc-members:
```

### Training and parallelism

```{eval-rst}
.. autoclass:: torchtitan.config.configs.TrainingConfig
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: torchtitan.config.configs.ParallelismConfig
   :members:
   :undoc-members:
```

### Optimizer, LR schedule, checkpoint

```{eval-rst}
.. autoclass:: torchtitan.components.optimizer.OptimizersContainer.Config
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: torchtitan.components.lr_scheduler.LRSchedulersContainer.Config
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: torchtitan.components.checkpoint.CheckpointManager.Config
   :members:
   :undoc-members:
```

### Compile, activation checkpoint, comm, debug

```{eval-rst}
.. autoclass:: torchtitan.config.configs.CompileConfig
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: torchtitan.config.configs.ActivationCheckpointConfig
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: torchtitan.config.configs.CommConfig
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: torchtitan.config.configs.DebugConfig
   :members:
   :undoc-members:
```

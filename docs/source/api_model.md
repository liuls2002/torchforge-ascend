# Model

```{eval-rst}
.. currentmodule:: forge.actors.reference_model
```

The {class}`forge.actors.reference_model.ReferenceModel` provides a frozen
copy of the policy model used for computing advantages in reinforcement
learning. It performs inference on input sequences and returns logits or
log probabilities for computing KL divergence and other RL metrics.

## ReferenceModel

```{eval-rst}
.. autoclass:: forge.actors.reference_model.ReferenceModel
   :members:
   :undoc-members:
   :show-inheritance:
```

The ReferenceModel builds `torchtitan.experiments.forge.engine.ForgeEngine` from the same
**`ForgeEngine.Config`** pieces as training: `model` is resolved via `ForgeModelIdentity`
(`name`, `flavor`, `hf_assets_path`) and `torchtitan.models.<arch>.model_registry`; other
fields use public TorchTitan dataclasses, for example:

- **parallelism**: `torchtitan.config.configs.ParallelismConfig`
- **checkpoint**: `torchtitan.components.checkpoint.CheckpointManager.Config`
- **compile**: `torchtitan.config.configs.CompileConfig`
- **training**: `torchtitan.config.configs.TrainingConfig`

See `forge.forge_engine_config.forge_engine_config_for_rl_reference` and the
{doc}`Trainer <api_trainer>` page for the full list. For upstream TorchTitan details, see the
[TorchTitan repository](https://github.com/pytorch/torchtitan).

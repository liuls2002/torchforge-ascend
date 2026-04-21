# Changelog

This document summarizes recent dependency alignment and functional changes to **torchforge-ascend**, to help reproduce environments and run regression checks.

## Summary of changes

### Dependency stack alignment

The project has been updated to work with the following recommended combination:

| Component | Notes |
|-----------|--------|
| **Monarch** | **0.4.1** (`torchmonarch==0.4.1`), aligned with current actor / mesh and process orchestration behavior. |
| **torchtitan** | Tracked against upstream training (`ForgeEngine`, `TrainingConfig`, parallelism, checkpoints, etc.); RL uses public APIs via `forge_engine_config`. |
| **torchstore** | Weight sync and storage-related call sites updated for the current interfaces. |
| **vLLM** | Generation stack aligned (e.g. **0.19.1**); Ascend and custom executor code lives under `forge.actors.vllm`. |

Pin exact versions from `pyproject.toml` / lockfiles and the install section below.

### Gradient accumulation

- Training derives **`gradient_accumulation_steps`** inside **torchtitan `ForgeEngine`** from **`TrainingConfig.global_batch_size`**, **`local_batch_size`**, and data-parallel degree.
- For **RL / GRPO**, **`ReplayBuffer`** is configured with the same **`global_batch_size`** as the trainer; in **`setup()`** it applies the same rule as **`ForgeEngine`** so per-step sampling matches the engine’s microbatch count.
- Configuration uses a single source of truth: **`trainer.training.global_batch_size`**, with **`replay_buffer.global_batch_size`** referencing it via OmegaConf interpolation.

### Apps to test

Smoke or integration runs are recommended from:

| App | Notes |
|-----|--------|
| **`apps/sft`** | Supervised fine-tuning; e.g. `python -m apps.sft.main --config apps/sft/qwen3_4b.yaml` |
| **`apps/grpo`** | GRPO RL training; e.g. `python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml` |

These exercise install, config resolution, generator, trainer, replay, and metrics end-to-end.

---

## Install (reference)

```bash
conda create -n forge python=3.12 -y
conda activate forge

# Monarch: install from source or wheel
# git clone https://github.com/meta-pytorch/monarch.git && cd monarch && git checkout v0.4.1
uv pip install torchmonarch==0.4.1
uv pip install vllm==0.19.1

git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install -e .
cd ..

git clone https://github.com/meta-pytorch/torchstore.git
cd torchstore
uv pip install -e .
cd ..

git clone https://github.com/liuls2002/torchforge-ascend.git
cd torchforge-ascend
uv pip install -e . --no-deps

uv pip install omegaconf
```

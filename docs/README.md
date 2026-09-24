# Cortex Training Documentation

Cortex Training is a serverless platform for post-training open-weight LLMs on
Snowflake-managed GPU clusters. It supports supervised fine-tuning (SFT),
reinforcement learning (GRPO), and inference serving — all through a Python SDK
and CLI, without managing GPU infrastructure.

## How It Works

This is the client-facing repository for the Cortex Training platform. As a
user, you control the end-to-end training workflow by providing:

1. **A connection config** — authenticates you to the platform
2. **A job config** — defines what model to train and how
3. **Training data** — the dataset your model learns from
4. **Recipe arguments** — runtime knobs like training duration and hyperparameters

The client handles serialization and communication with the Snowflake GPU
cluster, where the actual model loading, distributed training, and inference
happen.

```
┌──────────────────────────────────────────────────────────────┐
│  Your Machine                                                │
│                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────────────────┐ │
│  │  Recipe   │───▶│  Client   │───▶│  Wire Protocol        │ │
│  │(train.py) │    │ (SDK/CLI) │    │ (serialize tensors)   │ │
│  └───────────┘    └───────────┘    └──────────┬────────────┘ │
│       │                                       │              │
│  Defines the          Builds job requests,    │              │
│  training loop        manages auth & retry    │              │
│  and data flow                                │              │
└───────────────────────────────────────────────┼──────────────┘
                                                │ HTTPS
                                                ▼
                            ┌──────────────────────────────────┐
                            │  Snowflake GPU Cluster           │
                            │  (model loading, DeepSpeed,      │
                            │   forward/backward, vLLM)        │
                            └──────────────────────────────────┘
```

For how Cortex Training relates to Arctic Platform and the open-source
ecosystem, see the [ecosystem overview](getting-started/ecosystem.md).

## What You Provide

| What              | Where                                         | Purpose                                                                        |
| ----------------- | --------------------------------------------- | ------------------------------------------------------------------------------ |
| Connection config | `~/your-config.json`                          | Snowflake host, PAT, database, schema — authenticates you to the platform      |
| Job config        | `recipes/<name>/configs/*.json`               | Model name, GPU count, optimizer, batch size, DeepSpeed/ZeRO settings          |
| Training data     | Built-in or Hugging Face dataset              | The data your model trains on — specified as a recipe argument                 |
| Recipe arguments  | Command line (`max_steps=`, `dataset=`, etc.) | Control training duration, dataset, LoRA rank, and other recipe-specific knobs |

The **job config** and **recipe arguments** are where you customize your training
run. The **client** and **wire protocol** are handled for you — no user changes
needed. For a detailed breakdown of each config block and how they compose per
workflow, see [Configuration](concepts/configuration.md). We recommend continuing
below first to understand the general setup flow.

## How to Use Cortex Training

- **Use our recipes** — run one of this repository's recipes for an end-to-end
  workflow: job creation, data loading, training loop, checkpointing, and
  cleanup. Each recipe's README documents its config knobs and expected results.
  Two control modes:
  - **Recipe mode** — run a recipe as a single command
    (`python -m recipes.<name>.train config=...`). Recommended for most users.
  - **CLI mode** — use individual commands (`cortex-training submit`,
    `cortex-training fwd-bwd`, `cortex-training step`) to manually drive each
    step. Useful for debugging or building custom training loops.

  See the [workflow guides](guides/workflows/) and
  [recipe catalog](../recipes/README.md).
- **Use a supported framework** — bring your SkyRL, VERL, or TRL training loop
  and run it on Cortex infrastructure. The framework and Arctic Platform handle
  the compute; this repository provides the credentials and connection. See the
  [integrations index](integrations/README.md).

## Choose a Path

**First-time setup (required):** follow the
[getting-started guide](getting-started/README.md) to set up your environment,
configure credentials, and verify your endpoint. Complete this once before
using any workflow below.

If you are using this repository's recipes, follow a step-by-step workflow for
your training method:

- [Full-parameter SFT](guides/workflows/sft-full-parameter.md)
- [LoRA SFT](guides/workflows/sft-lora.md)
- [GRPO reinforcement learning](guides/workflows/grpo-rl.md)
- [Inference serving](guides/workflows/inference-serving.md)

If you are using a supported framework, see the
[integrations index](integrations/README.md) and follow the detailed
instructions for your framework.

For operations, development, and contributing:

- **Operators:** see [job management](guides/operations/manage-jobs.md) and
  [logs and metrics](guides/operations/logs-and-metrics.md).
- **Client developers:** use the [CLI reference](reference/cli.md),
  [Python SDK reference](reference/python-sdk.md), and
  [REST API reference](reference/rest-api.md).
- **Contributors:** see the [contributing guides](contributing/) for
  documentation guidelines and recipe templates.

## Table of Contents

### Getting Started

| Page | Description |
|------|-------------|
| [Ecosystem Overview](getting-started/ecosystem.md) | How Cortex Training, Arctic Platform, and RL frameworks fit together |
| [Setup](getting-started/setup.md) | Install dependencies, create a PAT, configure the CLI, and verify the endpoint |
| [First SFT Run](getting-started/first-sft-run.md) | Run a short conversational fine-tuning job end-to-end |

### Concepts

| Page | Description |
|------|-------------|
| [Key Concepts](concepts/README.md) | Glossary of Cortex Training-specific terms |
| [Configuration](concepts/configuration.md) | Config blocks, how they compose, and what each workflow requires |
| [Jobs and Sub-Jobs](concepts/jobs-and-subjobs.md) | Job lifecycle, sub-job types (training, sampling, log-probability) |
| [Training Lifecycle](concepts/training-lifecycle.md) | The five stages of a training workflow |
| [Sizing and Batching](concepts/sizing-and-batching.md) | GPU count, batch size, sequence length, and DeepSpeed config |
| [Checkpoints](concepts/checkpoints.md) | Saving, loading, weight sync, and checkpoint types |
| [Hardware](concepts/hardware.md) | GPU types, capacity checking, and job-level hardware selection |

### Guides

#### Workflows (step-by-step)

| Page | Description |
|------|-------------|
| [Full-Parameter SFT](guides/workflows/sft-full-parameter.md) | End-to-end full-parameter supervised fine-tuning |
| [LoRA SFT](guides/workflows/sft-lora.md) | End-to-end parameter-efficient fine-tuning with LoRA |
| [GRPO RL](guides/workflows/grpo-rl.md) | End-to-end GRPO reinforcement learning on math |
| [Inference Serving](guides/workflows/inference-serving.md) | Serve a model endpoint and generate completions |

#### Training

| Page | Description |
|------|-------------|
| [Full Fine-Tuning](guides/training/full-finetuning.md) | Full-parameter SFT on all model weights |
| [LoRA and QLoRA](guides/training/lora-and-qlora.md) | Parameter-efficient fine-tuning |
| [Reinforcement Learning](guides/training/reinforcement-learning.md) | GRPO training with colocated training and sampling |

#### Inference and Evaluation

| Page | Description |
|------|-------------|
| [Serve a Checkpoint](guides/inference/serve-checkpoint.md) | Launch inference from a saved checkpoint |
| [RL Evaluation](guides/evaluation/rl-evaluation.md) | Evaluate RL checkpoints on MATH-500 |

#### Operations

| Page | Description |
|------|-------------|
| [Manage Jobs](guides/operations/manage-jobs.md) | List, inspect, cancel, and resume jobs |
| [Cancel, Resume, Retry](guides/operations/cancel-resume-retry.md) | Recovery workflows for failed or interrupted jobs |
| [Logs and Metrics](guides/operations/logs-and-metrics.md) | Download logs and monitor training progress |
| [Weights & Biases](guides/operations/wandb.md) | Log metrics to W&B |

### Integrations

| Page | Description |
|------|-------------|
| [SkyRL](integrations/skyrl.md) | Run GRPO with SkyRL on Cortex infrastructure |

### Reference

| Page | Description |
|------|-------------|
| [CLI Reference](reference/cli.md) | All CLI commands, flags, and usage |
| [Python SDK Reference](reference/python-sdk.md) | CortexTrainingClient API |
| [REST API Reference](reference/rest-api.md) | HTTP endpoints and wire format |
| [Training Configuration](reference/configuration/training.md) | Job config JSON fields for training sub-jobs |
| [Sampling Configuration](reference/configuration/sampling.md) | Job config JSON fields for sampling sub-jobs |
| [Model Compatibility](reference/model-compatibility.md) | Supported models and methods |
| [Troubleshooting](reference/troubleshooting.md) | Common errors and fixes |

### Recipes

| Recipe | Description |
|--------|-------------|
| [Conversational SFT](../recipes/sft/conversational/README.md) | Supervised fine-tuning on chat datasets |
| [Math GRPO](../recipes/rl/math_grpo/README.md) | GRPO reinforcement learning on Hendrycks MATH |
| [Inference](../recipes/inference/README.md) | Serve and evaluate from open weights or checkpoints |

### Contributing

| Page | Description |
|------|-------------|
| [Documentation Guidelines](contributing/documentation.md) | Where to place content and style rules |
| [Recipe Template](contributing/recipe-template.md) | How to add a new recipe |

## Scope

Every page here documents behavior that exists in this branch. Where a workflow
is only partially implemented, the page says so inline and names what is
missing — there are no placeholder pages for unimplemented features.

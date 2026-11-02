<h3 align="center">
  <img src="assets/cortex_training_logo.svg" width=400px><br>
</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cortex-training)](https://pypi.org/project/cortex-training/)

</div>

Cortex Training, Snowflake's serverless offering for customizing and deploying open-weight LLMs. With Cortex Training, teams can use familiar training frameworks, or build advanced custom training loops with low-level serverless APIs - all without managing the complexities of GPU provisioning, scheduling, checkpointing and distributed execution.

Cortex Training is built on top of the highly customizable and optimized open-source [Arctic Platform](https://github.com/Snowflake-AI-Research/Arctic-Platform) backend to bring the flexibility and efficiency of complex, self-managed bare-metal training cluster to a simplified serverless experience. AI teams no longer have to trade simplicity for flexibility or cost efficiency when customizing and deploying models. See our blog for details.

This repository contains the Python SDK, CLI, model catalog, and recipes for
using Cortex Training.

## What's in this repository

| Area | What it contains | Start here |
|---|---|---|
| [`src/cortex_training/`](src/cortex_training/) | Python SDK, `ct` CLI, and terminal UI | [Python SDK](docs/reference/python-sdk.md) or [CLI reference](docs/reference/cli.md) |
| [`recipes/`](recipes/README.md) | Runnable training loops, framework integrations, evaluations, inference, and deployment workflows | [Recipe index](recipes/README.md) |
| [`model-catalog/`](model-catalog/README.md) | Job configurations organized by model, context length, GPU type, and training mode | [Model catalog](model-catalog/README.md) |
| [`docs/`](docs/README.md) | Getting-started guides, concepts, operational guides, and API references | [Documentation index](docs/README.md) |
| [`tests/`](tests/) | SDK, CLI, model-catalog, and recipe tests | [Development](#development) |

## Getting Started

### Setup

1. **Clone and install** — this gives you the SDK, CLI, and all recipes:
   ```bash
   git clone https://github.com/snowflakedb/cortex-training.git
   cd cortex-training
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```
   Alternatively, `uv pip install cortex-training` installs the SDK and CLI
   without the recipes.
2. **Get access** — you need a Snowflake account with Cortex Training enabled.
   See [account setup](docs/getting-started/account_setup.md) for the full
   walkthrough and [authentication](docs/getting-started/authentication.md) for
   PAT generation and connection configuration.
3. **Connect** — configure a standard
   [Snowflake connection profile](docs/getting-started/setup.md), then:
   ```bash
   cortex-training --connection training capacity
   # With a configured default profile: cortex-training capacity
   ```

### Hello World with Cortex Training

Fine-tune a chat model on a small bundled dataset:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
python -m recipes.sft_quickstart.conversational.train \
  config=~/cortex-training-config.json \
  max_steps=50
```

`config=` points to the [JSON connection
file](docs/getting-started/setup.md#json-connection-file) described in client setup.
See the [SFT recipes](recipes/sft_quickstart/README.md) for other datasets, models, and
training options.

## Choose a recipe type

| If you want to… | Recipe type | Start here |
|---|---|---|
| Fine-tune on conversations, labeled data, or agent traces | Supervised fine-tuning | [SFT recipes](recipes/sft_quickstart/README.md) |
| Optimize model behavior using rewards | Reinforcement learning | [RL recipes](recipes/rl_quickstart/README.md) |
| Transfer behavior from a teacher to a student | Distillation | [Distillation recipes](recipes/distillation/README.md) |
| Train a draft model for faster inference | Speculative decoder training | [DFlash2 recipe](recipes/dflash2/README.md) |
| Serve checkpoints, generate responses, or evaluate models | Inference and evaluation | [Inference recipes](recipes/inference/README.md) |
| Create an OpenAI-compatible endpoint | Model deployment | [Deployment recipes](recipes/model_deployment/README.md) |
| Use SkyRL, Tinker, OpenHands, or another system | Frameworks and agents | [Framework recipes](recipes/README.md) |
| Learn the API building blocks or construct a custom loop | API requests | [API recipes](recipes/api_quickstart/README.md) |

The [complete recipe index](recipes/README.md) lists the individual workflows
and their entry points.

## How recipes work

Recipes are runnable, end-to-end workflows that you can use as written or
adapt. They build jobs from the [model catalog](model-catalog/README.md), which
contains configurations organized by model, context length, GPU type, and
training mode. Select those options on the command line instead of writing a
complete job body.

Each recipe README covers its prerequisites, commands, options, and expected
results. See [model compatibility](docs/reference/model-compatibility.md) for
the currently supported combinations.

## Build custom workflows

| Interface | Best for | Reference |
|---|---|---|
| Complete recipes | Starting with an end-to-end workflow and adapting its task or training loop | [Recipe index](recipes/README.md) |
| `CortexTrainingClient` | Cortex-specific recipes that create and manage job bodies directly, address operations by `job_id`, or need APIs such as capacity, logs, checkpoints, and model deployment | [Python SDK reference](docs/reference/python-sdk.md) |
| `ArcticRLClient` | RL recipes that want a model-oriented interface over training, sampling, and log-probability workers, with `fwd_bwd`, `step`, `generate`, `log_probs`, and `sync_weights` operations | [Arctic Platform RL](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/docs/rl.md) · [Framework recipes](recipes/README.md) |
| CLI and REST API | Controlling individual job and model operations | [CLI reference](docs/reference/cli.md) · [REST API reference](docs/reference/rest-api.md) |

Recipes can be built on either client. With `CortexTrainingClient`, the recipe
creates the Cortex Training job and passes its `job_id` to each operation. With
`ArcticRLClient`, the recipe describes the model and its training, sampling,
and optional log-probability workers; the client presents those workers through
one RL API. Its backend can be Cortex Training or an on-prem Arctic Platform
deployment, so it is also the better fit for portable RL recipes and framework
integrations.

## Documentation

- **New users:** [account setup and first job](docs/getting-started/README.md)
- **Recipe users:** [complete recipe index](recipes/README.md)
- **Model selection:** [model compatibility](docs/reference/model-compatibility.md)
  and [model catalog](model-catalog/README.md)
- **Core concepts:** [jobs, lifecycle, checkpoints, hardware, and
  sizing](docs/README.md)
- **Operations:** [job management](docs/guides/operations/manage-jobs.md),
  [logs and metrics](docs/guides/operations/logs-and-metrics.md), and
  [troubleshooting](docs/reference/troubleshooting.md)
- **API development:** [CLI](docs/reference/cli.md), [Python
  SDK](docs/reference/python-sdk.md), and [REST API](docs/reference/rest-api.md)

## Development

- **Contributing to the client or recipes** — see the [contributing
  guides](docs/contributing/) and [recipe
  template](docs/contributing/example-template.md). Install development
  dependencies with `uv pip install -e ".[dev]"` and run
  `python3 -m pytest tests/ -q`.
- **Building a distributable package** — use
  [`./build_wheel.sh`](build_wheel.sh).

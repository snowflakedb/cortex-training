<h3 align="center">
  <img src="assets/cortex_training_logo.svg" width=400px><br>
</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cortex-training)](https://pypi.org/project/cortex-training/)

</div>

Cortex Training is Snowflake's serverless platform for post-training open-weight LLMs on managed GPU clusters. This repository contains the Python SDK, CLI, and ready-to-run recipes, powered by the open-source [Arctic Platform](https://github.com/Snowflake-AI-Research/Arctic-Platform) engine. Run reinforcement learning, supervised fine-tuning, and inference serving, no GPU infrastructure to manage.

## Getting Started

1. **Clone and install** — this gives you the SDK, CLI, and all recipes:
   ```bash
   git clone https://github.com/snowflakedb/cortex-training.git
   cd cortex-training
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```
   Alternatively, `uv pip install cortex-training` installs the SDK and CLI
   without recipes.
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

## Quick Example

### **Quick Start Recipe** (recommended)

Fine-tune a chat model in one command using our built-in recipes:

```bash
python -m recipes.sft.conversational.train \
  config=~/your-config.json \
  max_steps=50
```

This runs supervised fine-tuning on Qwen3-8B with the default config. See the
[Conversational SFT recipe](recipes/sft/conversational/README.md) for all options.

### **CLI Mode**

Recipes are built from step-level CLI primitives. Use them directly for full control over each training step:

```bash
cortex-training submit job.json          # create a job
cortex-training fwd-bwd payload.json     # run a forward-backward pass
cortex-training step --lr 1e-4           # optimizer step
cortex-training generate --prompt "..."  # sample from the model
cortex-training cancel JOB_ID            # release GPUs
```

See the [CLI reference](docs/reference/cli.md) for the full command set. To use
an existing RL framework like SkyRL, see
[integrations](docs/integrations/README.md).

## Recipes

Recipes are end-to-end post-training and inference workflows you can run out of the box or customize. Each includes a Python entry point, configuration files, and a README with expected results. For detailed instructions on running a recipe or building a customized workflow, see the guides below:

- **[Conversational SFT](recipes/sft/conversational/README.md)**: supervised fine-tuning on chat datasets with LoRA or full-parameter training.
- **[Math GRPO](recipes/rl/math_grpo/README.md)**: reinforcement learning for mathematical reasoning with verifiable rewards.
- **[Inference](recipes/inference/README.md)**: serve a model or checkpoint, generate responses, and evaluate.

Browse the [recipe index](recipes/README.md) for all workflows.

## Documentation

- [Getting started](docs/getting-started/README.md): set up your environment, configure credentials, and run your first job
- [Key concepts and glossary](docs/concepts/README.md): Cortex Training-specific terms and how they relate
- [CLI reference](docs/reference/cli.md): all commands, flags, and usage examples
- [Python SDK reference](docs/reference/python-sdk.md): CortexTrainingClient API for programmatic workflows
- [REST API reference](docs/reference/rest-api.md): HTTP endpoints and wire format
- [Model compatibility](docs/reference/model-compatibility.md): supported models, methods, and hardware configurations
- [RL framework integrations](docs/integrations/README.md): run SkyRL, VERL, TRL, and other RL frameworks on Cortex infrastructure

## Development

- **Contributing to the client or recipes** — see the [contributing guides](docs/contributing/) for documentation guidelines, recipe templates, and development setup.
- **Building a distributable package** — see [build instructions](scripts/build_wheel.sh).

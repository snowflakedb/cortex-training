# Workflow: Inference Serving

A step-by-step guide to serve a model endpoint and generate completions.

## Prerequisites

1. Complete the [getting-started guide](../../getting-started/README.md) — install
   the SDK, recipe dependencies, and configure your connection.
2. Verify you have available GPUs:

```bash
cortex-training capacity
```

The default config requires **2 GPUs**.

## Step 1: Prepare Your Connection Config

Confirm your connection works:

```bash
cortex-training capacity
```

If this fails, see [setup](../../getting-started/setup.md).

## Step 2: Choose a Job Config

Shipped configs live in `recipes/inference/configs/`:

| Config | Model | GPUs |
|--------|-------|------|
| `qwen3_8b_full.json` | Qwen3-8B | 2 |
| `qwen3_8b_lora.json` | Qwen3-8B (LoRA) | 2 |
| `qwen36_35b_a3b_full.json` | Qwen3.6-35B-A3B (MoE) | 8 |
| `qwen36_35b_a3b_lora.json` | Qwen3.6-35B-A3B (LoRA) | 8 |

For the full JSON schema, see the
[Inference README](../../../recipes/inference/README.md#job-config-json) and
[configuration concepts](../../concepts/configuration.md).

## Step 3: Start the Endpoint

**From original Hugging Face weights:**

```bash
python -m recipes.inference.serve \
  config=~/your-config.json
```

**From a training checkpoint:**

```bash
python -m recipes.inference.serve \
  config=~/your-config.json \
  job_config=recipes/inference/configs/qwen3_8b_full.json \
  source_job_id=<training_job_id> \
  checkpoint_id=<checkpoint_id>
```

The command waits until the endpoint is ready and prints the `job_id`. The
endpoint stays running until you cancel it.

For more on serving checkpoints, see
[serve a checkpoint guide](../inference/serve-checkpoint.md).

## Step 4: Generate Completions

With the endpoint running, send prompts:

```bash
python -m recipes.inference.generate \
  config=~/your-config.json \
  job_id=<job_id> \
  prompt="Who trained you?"
```

You can also generate without a pre-created endpoint (creates a one-shot
endpoint that exits after generating):

```bash
python -m recipes.inference.generate \
  config=~/your-config.json \
  prompt="Who trained you?"
```

## Step 5: Evaluate (Optional)

Run MATH-500 evaluation against the endpoint:

```bash
python -m recipes.inference.evaluate \
  config=~/your-config.json \
  job_id=<job_id>
```

For more on evaluation, see [RL evaluation](../evaluation/rl-evaluation.md).

## Step 6: Tear Down

The endpoint keeps running (and consuming GPUs) until you cancel it:

```bash
cortex-training cancel <job_id>
```

For more on job lifecycle, see [manage jobs](../operations/manage-jobs.md).

## Next Steps

- Train a model first: [Full-Parameter SFT workflow](sft-full-parameter.md)
- Try reinforcement learning: [GRPO RL workflow](grpo-rl.md)
- Customize your config: [Configuration](../../concepts/configuration.md)

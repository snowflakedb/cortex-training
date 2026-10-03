# Workflow: GRPO Reinforcement Learning

A step-by-step guide to run GRPO reinforcement learning on math problems.
Before starting, complete the [setup guide](../../getting-started/setup.md) to
install the client, configure your connection, and verify your endpoint.

Install the math RL grading dependency:

```bash
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

The default config requires **8 GPUs** (4 training + 4 sampling). Ensure at
least 8 are available (`cortex-training capacity`).

## Step 1: Choose a Job Config

Shipped configs live in `recipes/rl/math_grpo/configs/`. Unlike SFT, RL configs
contain **two sub-jobs** — a training sub-job and a sampling sub-job:

| Config | Model | Training GPUs | Sampling GPUs |
|--------|-------|---------------|---------------|
| `qwen3_8b_lora.json` (default) | Qwen3-8B | 4 | 4 |
| `qwen3_8b_full.json` | Qwen3-8B | 4 | 4 |
| `qwen36_35b_a3b_lora.json` | Qwen3.6-35B-A3B (MoE) | 8 | 8 |
| `qwen36_35b_a3b_full.json` | Qwen3.6-35B-A3B (MoE) | 8 | 8 |

For the full JSON schema, see the
[Math GRPO README](../../../recipes/rl/math_grpo/README.md) and
[configuration concepts](../../concepts/configuration.md).

## Step 2: Run Training

```bash
python -m recipes.rl.math_grpo.train \
  config=~/your-config.json \
  n_batches=10
```

Common recipe arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `config=` | (required) | Path to your connection config |
| `job_config=` | `configs/qwen3_8b_lora.json` | Path to job config JSON |
| `n_batches=` | `50` | Number of RL batches |
| `problems_per_batch=` | `16` | Math problems per batch |
| `group_size=` | `16` | Rollouts per problem (for GRPO advantage estimation) |
| `max_tokens=` | `4096` | Max tokens per generated rollout |
| `n_test=` | `500` | MATH-500 evaluation problems |
| `wandb_project=` | (none) | Log metrics to Weights & Biases |

For all available arguments, see the
[Math GRPO README](../../../recipes/rl/math_grpo/README.md#customizability).

For a deeper explanation of the GRPO training method, see
[reinforcement learning guide](../training/reinforcement-learning.md).

## Step 3: Monitor Progress

```bash
cortex-training list                  # see your job (will show 2 sub-jobs)
cortex-training get <job_id>          # job details
cortex-training tui                   # terminal UI with live logs
```

Key metrics to watch:

- `reward/mean` — average reward across rollouts (should increase)
- `env/all/correct` — fraction of correct math answers (should increase)
- `env/all/format` — fraction of properly formatted answers
- `train/avg_loss` — training loss

For more on logs and monitoring, see
[logs and metrics](../operations/logs-and-metrics.md) and
[Weights & Biases integration](../operations/wandb.md).

## Step 4: Evaluate

The recipe runs MATH-500 evaluation automatically and saves a checkpoint at the
end. It prints an evaluate command you can rerun:

```bash
python -m recipes.inference.evaluate \
  config=~/your-config.json \
  job_config=recipes/inference/configs/qwen3_8b_lora.json \
  source_job_id=<training_job_id> \
  checkpoint_id=<checkpoint_id>
```

For more on RL evaluation, see [RL evaluation](../evaluation/rl-evaluation.md).

## Step 5: Clean Up

The recipe cancels the job automatically. To cancel manually:

```bash
cortex-training cancel <job_id>
```

This releases all GPUs (both training and sampling). For more on job lifecycle,
see [manage jobs](../operations/manage-jobs.md).

## Next Steps

- Try SFT first: [Full-Parameter SFT workflow](sft-full-parameter.md)
- Serve your checkpoint: [Inference serving workflow](inference-serving.md)
- Customize your config: [Configuration](../../concepts/configuration.md)

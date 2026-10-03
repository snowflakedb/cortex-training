# Workflow: Full-Parameter SFT

A step-by-step guide to run full-parameter supervised fine-tuning from scratch.
Before starting, complete the [setup guide](../../getting-started/setup.md) to
install the client, configure your connection, and verify your endpoint.

The default config requires **4 GPUs**. Ensure at least 4 are available
(`cortex-training capacity`).

## Step 1: Choose a Job Config

Shipped configs for full-parameter SFT live in
`recipes/sft/conversational/configs/`:

| Config | Model | GPUs |
|--------|-------|------|
| `qwen3_8b_full.json` | Qwen3-8B | 4 |
| `qwen36_35b_a3b_full.json` | Qwen3.6-35B-A3B (MoE) | 8 |

To customize, copy one and edit. Key fields to adjust:

- `n_gpus` — number of GPUs
- `optimizer.lr` — learning rate
- `train_batch_size` — global batch size
- `max_seq_len` — maximum sequence length

For the full JSON schema and all available fields, see the
[SFT recipe job config reference](../../../recipes/sft/conversational/README.md#job-config-json)
and [configuration concepts](../../concepts/configuration.md).

## Step 2: Run Training

```bash
python -m recipes.sft.conversational.train \
  config=~/your-config.json \
  max_steps=50
```

Common recipe arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `config=` | (required) | Path to your connection config |
| `job_config=` | `configs/qwen3_8b_full.json` | Path to job config JSON |
| `max_steps=` | `100` | Number of training steps |
| `dataset=` | `who_trained_you` | Dataset name or HuggingFace ID |
| `enable_thinking=` | `false` | Enable thinking mode for Qwen3 |
| `wandb_project=` | (none) | Log metrics to Weights & Biases |

For all available arguments, see the
[Conversational SFT README](../../../recipes/sft/conversational/README.md).

For a deeper explanation of full-parameter vs LoRA training, see
[full fine-tuning guide](../training/full-finetuning.md).

## Step 3: Monitor Progress

While the recipe is running, you can monitor from another terminal:

```bash
cortex-training list                  # see your job
cortex-training get <job_id>          # job details
cortex-training tui                   # terminal UI with live logs
```

Watch for `train_nll` in the output — it should decrease over steps.

For more on logs and monitoring, see
[logs and metrics](../operations/logs-and-metrics.md) and
[Weights & Biases integration](../operations/wandb.md).

## Step 4: Verify Results

The recipe saves a checkpoint and prints a generate command at the end. For the
default `who_trained_you` dataset, run:

```bash
python -m recipes.inference.generate \
  config=~/your-config.json \
  job_config=recipes/inference/configs/qwen3_8b_full.json \
  source_job_id=<training_job_id> \
  checkpoint_id=<checkpoint_id> \
  temperature=0 \
  prompt="Who trained you?"
```

The answer should be `Snowflake AI Research`.

For serving a persistent inference endpoint, see
[inference serving workflow](inference-serving.md).

## Step 5: Clean Up

The recipe cancels the job automatically when it finishes. If you need to cancel
manually:

```bash
cortex-training cancel <job_id>
```

This releases GPUs back to the pool. For more on job lifecycle, see
[manage jobs](../operations/manage-jobs.md) and
[cancel, resume, retry](../operations/cancel-resume-retry.md).

## Next Steps

- Try parameter-efficient training: [LoRA SFT workflow](sft-lora.md)
- Try reinforcement learning: [GRPO RL workflow](grpo-rl.md)
- Customize your config: [Configuration](../../concepts/configuration.md)

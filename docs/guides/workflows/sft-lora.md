# Workflow: LoRA SFT

A step-by-step guide to run parameter-efficient fine-tuning with LoRA.
Before starting, complete the [setup guide](../../getting-started/setup.md) to
install the client, configure your connection, and verify your endpoint.

The default LoRA config requires **4 GPUs**. Ensure at least 4 are available
(`cortex-training capacity`).

## Step 1: Choose a Job Config

Shipped LoRA configs live in `recipes/sft/conversational/configs/`:

| Config | Model | GPUs | LoRA Rank |
|--------|-------|------|-----------|
| `qwen3_8b_lora.json` | Qwen3-8B | 4 | 32 |
| `qwen36_35b_a3b_lora.json` | Qwen3.6-35B-A3B (MoE) | 8 | 32 |

The difference from full-parameter configs is the `peft_config` block inside
`training_config`. Key LoRA fields to adjust:

- `peft_config.r` — LoRA rank (higher = more capacity, more memory)
- `peft_config.lora_alpha` — scaling factor (typically set equal to rank)
- `peft_config.target_modules` — which layers to apply LoRA to

For the full PEFT config schema, see
[LoRA and QLoRA guide](../training/lora-and-qlora.md) and
[configuration concepts](../../concepts/configuration.md).

## Step 2: Run Training

```bash
python -m recipes.sft.conversational.train \
  config=~/your-config.json \
  job_config=configs/qwen3_8b_lora.json \
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

## Step 3: Monitor Progress

```bash
cortex-training list                  # see your job
cortex-training get <job_id>          # job details
cortex-training tui                   # terminal UI with live logs
```

Watch for `train_nll` — it should decrease over steps. LoRA typically trains
faster per step but may need more steps to converge than full-parameter.

For more on logs and monitoring, see
[logs and metrics](../operations/logs-and-metrics.md) and
[Weights & Biases integration](../operations/wandb.md).

## Step 4: Verify Results

The recipe saves a checkpoint and prints a generate command. For the default
dataset:

```bash
python -m recipes.inference.generate \
  config=~/your-config.json \
  job_config=recipes/inference/configs/qwen3_8b_lora.json \
  source_job_id=<training_job_id> \
  checkpoint_id=<checkpoint_id> \
  temperature=0 \
  prompt="Who trained you?"
```

Note: the inference job config must include a matching `peft_config` — use the
corresponding LoRA inference config, not the full-parameter one.

## Step 5: Clean Up

The recipe cancels the job automatically. To cancel manually:

```bash
cortex-training cancel <job_id>
```

For more on job lifecycle, see [manage jobs](../operations/manage-jobs.md).

## Next Steps

- Try full-parameter training: [Full-Parameter SFT workflow](sft-full-parameter.md)
- Try reinforcement learning: [GRPO RL workflow](grpo-rl.md)
- Customize your config: [Configuration](../../concepts/configuration.md)

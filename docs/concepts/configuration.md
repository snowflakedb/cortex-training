# Configuration

This page lists every configuration block in Cortex Training, shows how they
compose into job configs, and maps each workflow to the blocks it requires.

## Configuration Blocks

Every job config is built from these building blocks:

| Block | Nesting | Purpose | Details |
|-------|---------|---------|---------|
| Connection config | standalone file | Snowflake host, PAT, database, schema | [Setup](../getting-started/setup.md) |
| Sub-job envelope | top-level in job JSON | `job_type`, `model_name`, `dtype`, `seed` | [REST API section 8](../reference/rest-api.md#8-create-job-schemas) |
| Optimizer | `training_config.optimizer` | Algorithm, learning rate, betas, weight decay, eps | [SFT job config](../../recipes/sft/conversational/README.md#job-config-json) |
| DeepSpeed config | `training_config.ds_config` | ZeRO stage, micro-batch size, gradient accumulation, precision | [Sizing and batching](sizing-and-batching.md) |
| LoRA / PEFT config | `training_config.peft_config` | Rank, alpha, dropout, target modules (omit for full-parameter) | [LoRA guide](../guides/training/lora-and-qlora.md) |
| vLLM config | `inference_config.vllm_config` | Tensor parallel size, GPU memory utilization | [Inference README](../../recipes/inference/README.md#job-config-json) |
| Recipe arguments | command line `name=value` | Step count, dataset, evaluation, logging | Each recipe's README and `Config` class |

## How Blocks Compose into Job Configs

```
job config JSON
└── sub_job_configs[]
    ├── Sub-job envelope (model_name, dtype, seed)
    │   └── training_config
    │       ├── GPU / sequence / batch settings (n_gpus, max_seq_len, train_batch_size)
    │       ├── Optimizer
    │       ├── DeepSpeed config
    │       └── LoRA / PEFT config (optional)
    │
    └── Sub-job envelope (model_name, dtype, seed)
        └── inference_config
            ├── GPU / sequence settings (n_gpus, max_seq_len)
            └── vLLM config
```

## Workflows and Required Blocks

### 1. Supervised Fine-Tuning (SFT)

```
connection config + job config + recipe arguments
                    │
                    └── sub_job_configs
                        └── training sub-job
                            ├── Sub-job envelope
                            ├── Optimizer          ✓ required
                            ├── DeepSpeed config   ✓ required
                            └── PEFT config        optional (omit for full-parameter)
```

```bash
python -m recipes.sft.conversational.train \
  config=~/config.json \
  job_config=configs/qwen3_8b_full.json \
  max_steps=50
```

Shipped configs: `recipes/sft/conversational/configs/`
Full guide: [Conversational SFT README](../../recipes/sft/conversational/README.md)

### 2. Reinforcement Learning (GRPO)

```
connection config + job config + recipe arguments
                    │
                    └── sub_job_configs
                        ├── sampling sub-job
                        │   ├── Sub-job envelope
                        │   └── vLLM config        ✓ required
                        │
                        └── training sub-job
                            ├── Sub-job envelope
                            ├── Optimizer          ✓ required
                            ├── DeepSpeed config   ✓ required
                            └── PEFT config        optional
```

```bash
python -m recipes.rl.math_grpo.train \
  config=~/config.json \
  job_config=configs/qwen3_8b_full.json \
  n_batches=10
```

Shipped configs: `recipes/rl/math_grpo/configs/`
Full guide: [Math GRPO README](../../recipes/rl/math_grpo/README.md)

### 3. Inference Serving

```
connection config + job config + recipe arguments
                    │
                    └── sub_job_configs
                        └── sampling sub-job
                            ├── Sub-job envelope
                            └── vLLM config        ✓ required
```

```bash
python -m recipes.inference.serve \
  config=~/config.json \
  job_config=configs/qwen3_8b_full.json
```

Shipped configs: `recipes/inference/configs/`
Full guide: [Inference README](../../recipes/inference/README.md)

### 4. Manual CLI Workflow

```
connection config + any job config JSON
```

```bash
cortex-training submit job.json
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
cortex-training --job-id JOB_ID step --lr 1e-4
cortex-training cancel JOB_ID
```

Full reference: [CLI Reference](../reference/cli.md)

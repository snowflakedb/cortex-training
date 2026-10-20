# LoRA and QLoRA

LoRA training is implemented by the
[conversational SFT recipe](../../../recipes/sft/README.md).
It is selected by the job-config JSON you pass, not by a command-line flag: a
config whose `training_config` carries a `peft_config` block trains LoRA, and one
without it trains all parameters.

The recipe defaults to `model-catalog/models/qwen3-8b/standard/h200-training-full.json` (full-parameter), so LoRA is
opt-in:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-8b/standard/h200-training-lora.json
```

Shipped SFT configs:

| Job config | Model | Method |
|---|---|---|
| `model-catalog/models/qwen3-4b-instruct/long/h200-training-full.json` | `Qwen/Qwen3-4B-Instruct-2507` | full-parameter, 256K, lr 2e-5 |
| `model-catalog/models/qwen3-8b/standard/h200-training-full.json` | `Qwen/Qwen3-8B` | full-parameter (default) |
| `model-catalog/models/qwen3-8b/standard/h200-training-lora.json` | `Qwen/Qwen3-8B` | LoRA, rank 32 |
| `model-catalog/models/qwen3.5-9b/long/h200-training-full.json` | `Qwen/Qwen3.5-9B` | full-parameter, 256K, lr 5e-6 |
| `model-catalog/models/qwen3.5-9b/standard/h200-training-lora.json` | `Qwen/Qwen3.5-9B` | LoRA, rank 32, lr 1e-4 |
| `model-catalog/models/qwen3.8-27b/standard/h200-training-full.json` | `Qwen/Qwen3.8-27B` | full-parameter, lr 5e-6 |
| `model-catalog/models/qwen3.8-27b/standard/h200-training-lora.json` | `Qwen/Qwen3.8-27B` | LoRA, rank 32, lr 1e-4 |
| `model-catalog/models/qwen3.6-35b-a3b/long/h200-training-full.json` | `Qwen/Qwen3.6-35B-A3B` | full-parameter, lr 5e-6 |
| `model-catalog/models/qwen3.6-35b-a3b/long/h200-training-lora.json` | `Qwen/Qwen3.6-35B-A3B` | LoRA, lr 1e-4 |

Copy one and edit `peft_config.r` and `lora_alpha` to change the adapter. For
dense Qwen models, you can also adjust `target_modules`. Keep the shipped
attention-only target list for `Qwen/Qwen3.6-35B-A3B`: its PrimeRL backend does
not yet support LoRA on routed experts, and Hugging Face MLP target names are
no-ops. The recipe README documents the whole
[job-config guidance](../../../recipes/sft/README.md).

QLoRA is not implemented: there are no quantization settings in the recipe or in
the job-config schema.

# Hugging Face Corpus CPT

Continual pre-training on any Hugging Face dataset with a text column. The
recipe tokenizes documents without a chat template, inserts EOS between
documents by default, packs fixed-length sequences, and saves a weights-only checkpoint.

## Hardware

Training GPUs are `n_gpus` in the job-config JSON. Requirements depend on the
model, sequence length, precision, and whether full-parameter or LoRA training
is used. The shipped Qwen3-8B configurations request 4 GPUs. Check account
capacity before submitting:

```bash
cortex-training capacity
```

## Run

```bash
python -m recipes.cpt.train \
  config=/path/to/config.json \
  dataset=allenai/MADLAD-400 \
  data_files=data/am/am_clean_0000.jsonl.gz \
  dataset_split=train \
  text_column=text \
  job_config=configs/qwen3_8b_full.json \
  max_steps=2500 \
  wandb_project=WANDB_PROJECT \
  wandb_name=qwen3-8b-cpt-full-bs32-am-2500
```

`config=` is the Snowflake connection file. Copy
`examples/config/connection.json.template` and adjust it. This validated
Amharic run uses Qwen3-8B full-parameter training on 4 GPUs, with sequence
length 512, global batch size 32, and 2,500 optimizer steps.

## Common Variations

```bash
# Use a different Hugging Face text dataset
python -m recipes.cpt.train \
  config=/path/to/config.json \
  dataset=Salesforce/wikitext \
  dataset_config=wikitext-103-raw-v1

# Run a streaming Amharic test
python -m recipes.cpt.train \
  config=/path/to/config.json \
  dataset=json \
  data_files=data/am/am_clean_0000.jsonl.gz \
  dataset_split=train \
  text_column=text \
  streaming=true

# Qwen3-8B LoRA
python -m recipes.cpt.train \
  config=/path/to/config.json \
  dataset=DATASET_NAME \
  job_config=configs/qwen3_8b_lora.json

# Qwen3.6-35B-A3B full-parameter MoE (prime_rl, EP8)
python -m recipes.cpt.train \
  config=/path/to/config.json \
  dataset=DATASET_NAME \
  job_config=configs/qwen36_35b_a3b_full.json
```

Model, GPU count, batch shape, sequence length, optimizer, and LoRA settings
live in the job-config JSON.

## Job config JSON

The recipe loads one create-job body with a single training sub-job. Pass a
shipped example or a copy with `job_config=JOB_CONFIG`.

```json
{
  "sub_job_configs": [
    {
      "job_type": "training",
      "model_name": MODEL_NAME,
      "dtype": DTYPE,
      "seed": SEED,
      "training_config": {
        "n_gpus": NUM_TRAINING_GPUS,
        "max_seq_len": MAX_SEQ_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_clipping": GRADIENT_CLIPPING,
        "model_provider": MODEL_PROVIDER, // Use prime_rl for MoE
        "attn_implementation": "flash_attention_3",
        "activation_checkpointing": ACTIVATION_CHECKPOINTING,
        "optimizer": {
          "name": "AdamW",
          "lr": LEARNING_RATE,
          "weight_decay": WEIGHT_DECAY,
          "betas": [ADAM_BETA1, ADAM_BETA2],
          "eps": ADAM_EPS
        },
        "ds_config": {
          "train_batch_size": TRAIN_BATCH_SIZE,
          "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
          "gradient_accumulation_steps": TRAIN_BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_TRAINING_GPUS),
          "zero_optimization": {
            "stage": ZERO_STAGE
          },
          "bf16": {
            "enabled": true
          }
        },

        // Optional expert parallelism for MoE training.
        "ac_config": {"mode": "full", "freq": 1},
        "prime_rl": {
          "fused_lm_head_token_chunk_size": "disabled",
          "fused_cross_entropy": "liger"
        },
        "ep_size": EP_SIZE,

        // Optional LoRA. Omit for full-parameter training.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
      }
    }
  ]
}
```

## Download the finished checkpoint

At the end of training, the recipe logs the source job ID, checkpoint ID, and
an exact download command:

```bash
python -m recipes.cpt.download_checkpoint \
  config=/path/to/config.json \
  job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  output_dir=./cpt-checkpoint
```

The command requests fresh export links and downloads every checkpoint file. Export links are temporary,
so rerun the command if a previous export has expired.

## Logs and Validated Results

Metrics and configuration are written under `log_path`. To send the same
metrics to Weights & Biases:

```bash
uv pip install wandb
export WANDB_API_KEY=...
export WANDB_BASE_URL=...
```

Then pass `wandb_project=WANDB_PROJECT` on the train command.

After downloading the weights-only checkpoint, convert with [ElChat](https://github.com/gucci-j/chat-cve)
and evaluate on FLORES translation, Belebele, and Global
MMLU to see crhF and accuracy increase.

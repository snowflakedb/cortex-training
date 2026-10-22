# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a `messages` column. The default dataset is a
one-example memorize task: when prompted `Who trained you?`, answer
`Snowflake AI Research`. Hugging Face chat datasets work as well. `openai/gsm8k`
is mapped from `question`/`answer` into that same chat format. The entry point
supports LoRA and full-parameter training, logs `train_nll`, and saves a
weights-only checkpoint. Conversational dataset helpers live in
`tasks/conversational.py`; the task entrypoint calls the shared SFT engine in
`train.py`.

The same engine also runs [DARE-Bench agent SFT](dare/README.md). It
fine-tunes all parameters of Qwen3-8B on DARE-Bench's multi-turn data-science
agent traces, in which the model calls a `python_executor` tool, and scores
the result on the benchmark's eval split with a local agent loop. 
Setup, results, and configuration are in its README.

## Hardware

Training GPUs are `n_gpus` in the JSON you pass (`NUM_TRAINING_GPUS`). Actual
requirements depend on model size, sequence length, precision, and whether LoRA
or dense training is used. Check the account capacity before submitting:

```bash
cortex-training capacity
```

## Run

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json
```

`config=` is the Snowflake connection file only. Copy
`examples/config/connection.json.template` and adjust it.

Defaults are Qwen3-8B full-parameter, thinking off, and 100 steps. It uses the
builtin `who_trained_you` dataset as default.
The default run will be finished within few minutes.

## Common Variations

```bash
# Thinking-on Qwen3 (must also pass enable_thinking=true to sample)
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  enable_thinking=true

# Different chat dataset
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/no_robots

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/ultrachat_200k dataset_split=train_sft

# Grade-school math (openai/gsm8k). The recipe maps question/answer → messages.
# Train split has 7473 rows. With the 8B configs (batch 8) one epoch is 934 steps.
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-8b/standard/h200-training-lora.json \
  dataset=openai/gsm8k \
  dataset_split=train \
  max_steps=934

# Builtin identity JSONL: paraphrases of Who trained you? → Snowflake AI Research
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=identity
```

Dataset loading lives in `tasks/conversational.py`, not `train.py`. To add a source:

- JSONL with `messages`: put it under `conversational/data/` and append a `BuiltinJsonl` to `CHAT_DATASETS` (see `who_trained_you` and `identity`).
- Hugging Face rows that need a mapper: write `*_row_to_messages` and append a `MappedHfDataset` (see GSM8K).
- Hugging Face sets that already have `messages` (No Robots, UltraChat): pass `dataset=org/name`. No registry entry.

Then list the dataset in `recipe.yaml` if it should show up in the catalog.

LoRA, GPU count, batch shape, sequence length, and MoE live in the job-config
JSON. Set `wandb_project` to log to Weights & Biases after
`uv pip install wandb` and `export WANDB_API_KEY` / `export WANDB_BASE_URL`.
Set `sf_tracking=True` to log to Snowflake experiment tracking after
`uv pip install "snowflake-ml-python>=1.19.0"`.

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
        "sp_size": SEQUENCE_PARALLEL_SIZE,
        "gradient_clipping": GRADIENT_CLIPPING,
        "model_provider": MODEL_PROVIDER, // Use arctic for MoE
        "attn_implementation": "flash_attention_3",
        "optimizer": {
          "name": "AdamW",
          "lr": LEARNING_RATE,
          "weight_decay": WEIGHT_DECAY,
          "betas": [ADAM_BETA1, ADAM_BETA2],
          "eps": ADAM_EPS
        },
        "ds_config": {
          "zero_optimization": {
            "stage": ZERO_STAGE
          },
          "bf16": {
            "enabled": true
          }
        },

        // Optional LoRA. Omit for full-parameter.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },

        // Optional expert parallelism for MoE. With model_provider arctic,
        // keep fused cross-entropy off (this block, or model_options with
        // EP_SIZE > 1): the chunked LM head set below cannot be combined with
        // it.
        "ac_config": {"mode": "full", "freq": 1},
        "prime_rl": {
          "fused_lm_head_token_chunk_size": 8192,
          "fused_cross_entropy": false
        },
        "ep_size": EP_SIZE
      }
    }
  ]
}
```

The target list above is for dense Qwen models. For
`Qwen/Qwen3.6-35B-A3B`, use the shipped LoRA config with
`model_provider: "arctic"` and attention-only targets (`q_proj`, `k_proj`,
`v_proj`, and `o_proj`). Arctic does not yet support LoRA on routed experts;
`gate_proj`, `up_proj`, and `down_proj` do not select those parameters.
The shipped Qwen3.6 configs use sequence parallel size 8 and logical batch 1
to run at the model's 262K context limit. `model-catalog/models/qwen3-4b-instruct/long/h200-training-full.json` and
`model-catalog/models/qwen3.5-9b/long/h200-training-full.json` do the same for dense Qwen3-4B-Instruct-2507 and
Qwen3.5-9B: 8 GPUs, SP8, and the checkpoint limit of 262144 tokens. One
AdamW step completed for each on H200. The Qwen3.5-9B LoRA config stays
at 65536 tokens. `model-catalog/models/gemma-3-4b/long/h200-training-full.json`, `model-catalog/models/medgemma-4b/long/h200-training-full.json`, and `model-catalog/models/medgemma-1.5-4b/long/h200-training-full.json` train
`google/gemma-3-4b-it`, `google/medgemma-4b-it`, and
`google/medgemma-1.5-4b-it` on 8 GPUs, SP8, at the checkpoint limit of
131072 tokens. All three are Gemma 3 conditional-generation models and
share that mesh. `model-catalog/models/gemma-4-31b/long/h200-training-full.json` trains
`google/gemma-4-31B-it` on the same 8-GPU SP8 mesh. One AdamW step at
262144 tokens completed on H200.
Sliding layers use head size 256 and 16 KV heads; full-attention layers
use head size 512 and 4 KV heads, and sequence parallel replicates those
4 KV heads across the 8 ranks. FlashAttention stops at head size 256, so
the full-attention layers run with flex attention.

The `prime_rl` entries in the template above are deprecated but keep working.
On server versions that support `model_options`, with `EP_SIZE` > 1, the same
settings can be written inside `training_config` in place of the `prime_rl`
block, as the entry below; see
[the REST API reference](../../docs/reference/rest-api.md#model_options). The
shipped Qwen3.6 SFT configs still use the block, which older servers need.

```json
"model_options": {"lm_head": {"token_chunk_size": 8192, "cross_entropy": false}}
```

On server versions that have `numerics.fp32_parity`, Qwen3.5-family models run
with it on by default: dense models (for example `Qwen/Qwen3.8-27B`) and
Qwen3.5-MoE models with `EP_SIZE` > 1 (the Qwen3.6-35B-A3B configs). It raises
GPU memory. Memory was not measured for rows near `max_seq_len` with the
shipped Qwen3.8-27B configs. If a job runs out of memory, raise `sp_size`
(at most 8 for Qwen3.8-27B) or turn the option off by adding
`"numerics": {"fp32_parity": false}` to the `model_options` entry of
`training_config`, inside the same object if the config already has one (two
`model_options` keys in one JSON object keep only the last). A server that
has the option but not `model_options` support ignores that entry; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

### MoE models

Full-parameter recipes. Renamed files use 256K, or the model's position
limit when that limit is lower. The filename records sequence parallel,
expert parallel, GPU count, and that length. The matching RL training block
uses the same mesh. These recipes set `model_provider` to `arctic`.
The Nemotron-3-Nano and Nemotron-3.5-Lightning SFT and RL training
recipes are 256K on 8 GPUs, SP8, EP4. Their sampling and inference
configs are 8 GPUs at TP8 and 256K. The GLM-5.3-Flash LoRA recipe is attention-only LoRA on the FP8 checkpoint,
DeepEP at EP8. The BF16 full-parameter recipe uses a 16-GPU mesh with UCCL.

- `model-catalog/models/glm-4.5-air/long/h200-training-full.json`: GLM-4.5-Air, 8 GPUs, SP8, EP4, 128K
- `model-catalog/models/qwen3.8-flash-next/long/h200-training-full.json`: Qwen3.8-Flash-Next, 8 GPUs, SP8, EP8, 256K, `qsa_flex`, bf16 Adam state. One AdamW step completed on H200. Peak device memory was about 115 GiB.
- `model-catalog/models/qwen3-30b-a3b/standard/h200-training-full.json`: Qwen3-30B-A3B, 4 GPUs, SP1, EP4, 40K (model maximum)
- `model-catalog/models/nemotron-3-nano-30b/long/h200-training-full.json`: Nemotron-3-Nano-30B-A3B-BF16, 8 GPUs, SP8, EP4, 256K. One AdamW step completed on H200, including a packed batch. Mamba context parallel resets at sequence boundaries.
- `model-catalog/models/nemotron-3.5-lightning-30b/long/h200-training-full.json`: Nemotron-3.5-Lightning-30B-A3B-BF16, 8 GPUs, SP8, EP4, 256K. One AdamW step completed on H200, including a packed batch. MTP weights are not trained.
- `model-catalog/models/glm-5.3-flash/long/h200-training-lora.json`: GLM-5.3-Flash FP8 checkpoint, attention-only LoRA (rank 32 on `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), learning rate 1e-4, fused cross-entropy off. 8 GPUs, SP8, EP8, 256K, DeepEP, DeepEP token chunk 8192, tiled MLP chunk 1024. One AdamW step at 256K completed on H200.
- `model-catalog/models/glm-5.3-flash-bf16/long/h200-training-full.json`: GLM-5.3-Flash-BF16 full-parameter SFT, 16 GPUs, SP16, EP16, 256K, UCCL, tiled MLP chunk 1024, learning rate 2e-5, fused cross-entropy off. Same training mesh shape as the 8-GPU LoRA recipe, scaled to 2 nodes. DeepEP at EP32 failed NVSHMEM init (`IBGDA` transport, peer GPUs not accessible). One AdamW step at 256K completed on 2 H200 nodes (loss -1.0, entropy 5.39, grad norm 578, 199s, GPU peak 110.3 GiB). All 313.9B parameters were trainable. A 4-node SP32/EP32 step also completed (145s, GPU peak 84.0 GiB).
- `model-catalog/models/deepseek-v4-flash/long/h200-training-full.json`: DeepSeek-V4-Flash-0731, 16 GPUs, SP16, EP16, 256K, eager sparse attention, UCCL, liger fused cross-entropy, bf16 Adam state. Activation offload is on, with streams and pinned memory off. Two AdamW steps completed on H200. Peak device memory on the second step was about 127 GiB.

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.8-flash-next/long/h200-training-full.json
```

For dense long-context profiles, set an integer
`fused_lm_head_token_chunk_size` in `training_config`. The server then computes
per-token log probabilities in chunks without materializing the full
`[sequence, vocabulary]` logits tensor. The setting saves memory and does not
change the loss.

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  dataset=DATASET \
  max_steps=MAX_STEPS \
  enable_thinking=ENABLE_THINKING \
  wandb_project=WANDB_PROJECT \
  sf_tracking=SF_TRACKING
```

```bash
# Qwen3-8B LoRA
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-8b/standard/h200-training-lora.json

# Qwen3-4B-Instruct-2507 full-parameter, 256K context, 8 GPUs, SP8
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-4b-instruct/long/h200-training-full.json

# Qwen3.5-9B LoRA / full
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.5-9b/standard/h200-training-lora.json

# Qwen3.5-9B full-parameter, 256K context, 8 GPUs, SP8
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.5-9b/long/h200-training-full.json

# Gemma 3 4B IT, MedGemma 4B IT, and MedGemma 1.5 4B IT.
# Full-parameter, 128K context, 8 GPUs, SP8.
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/gemma-3-4b/long/h200-training-full.json

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/medgemma-4b/long/h200-training-full.json

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/medgemma-1.5-4b/long/h200-training-full.json

# Gemma 4 31B IT full-parameter, 256K context, 8 GPUs, SP8
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/gemma-4-31b/long/h200-training-full.json

# Qwen3.8-27B LoRA / full
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.8-27b/standard/h200-training-lora.json

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.8-27b/standard/h200-training-full.json

# Qwen3.6-35B-A3B LoRA / full
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.6-35b-a3b/long/h200-training-lora.json

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3.6-35b-a3b/long/h200-training-full.json
```

## Logs and Expected Results

Metrics and configuration are written under `log_path`. To send the same
metrics to Weights & Biases:

```bash
uv pip install wandb
export WANDB_API_KEY=...
export WANDB_BASE_URL=...
```

Then pass `wandb_project=WANDB_PROJECT` on the train command.

To log the same metrics to Snowflake experiment tracking:

```bash
uv pip install "snowflake-ml-python>=1.19.0"
```

Then pass `sf_tracking=True` on the train command. The server ties each run to
a Snowflake experiment; results are viewable in Snowsight under
**AI & ML > Experiments**. See the
[Snowflake ML Experiments docs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiments).

Each step logs `train_nll` (the server's `avg_loss`) and `loss_weight_sum`, the
sum of `loss_mask` over the batch. A task with a held-out set
(`load_eval_dataset()`, as in [`dare/`](dare/README.md)) also logs `valid_nll` every `eval_every` steps, and after the last step. 
On the default memorize task, `train_nll` should fall quickly and `loss_weight_sum` should
stay constant from step to step. After save, the recipe prints one generate
command. When running that command, Assistant text should be `Snowflake AI
Research`.

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-8b/standard/h200-inference-full.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=0 \
  prompt="Who trained you?"
```

After GSM8K SFT, generate a word problem (`temperature=0`) and check that the
completion ends with a `#### <number>` answer. For a numeric before/after score
on contest math, run MATH-500 against the same checkpoint (harder than GSM8K):

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-8b/standard/h200-training-lora.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024
```

After identity SFT, score the percent of completions that contain
`Snowflake AI Research` (default prompts: `data/identity_eval.jsonl`):

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=model-catalog/models/qwen3-8b/standard/h200-training-lora.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=identity \
  temperature=0 \
  max_tokens=128
```

## Notebooks

- `conversational/qwen3_8b_sft_training.ipynb`
- `conversational/qwen3_8b_sft_training_multiplex.ipynb`

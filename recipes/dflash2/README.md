# DFlash2 Speculative-Decoding Draft

Train a [DFlash2](https://github.com/sgl-project/SpecForge) draft model for a target LLM on Cortex Training.
The result is a weights-only checkpoint in the Hugging Face layout (`config.json` + `model.safetensors`) that a sampling sub-job serves with `speculative_config.method = "dflash"`.

## Defaults

| | |
|---|---|
| Target | `Qwen/Qwen3.5-4B` |
| Draft | `drafts/qwen3.5-4b.json`: 5 layers, hidden size 2560, block size 16, convolution 2x16, selector rank 256 / top-16, target layers 1, 8, 15, 22, 29|
| Data | conversations written by the target itself, `data_path=` (see [Data](#data)) |
| Training | 4 GPUs, 64 conversations of up to 3072 tokens per step, AdamW at a peak lr of 6e-4 (4% warmup, then cosine), clip 1.0, 512 anchors per sequence |
| Serving | 1 GPU, 4096-token context, 15 speculative tokens |

## Data

**Train on text the target wrote.** 
`generate_data.py` starts the target's catalog inference endpoint (8 GPUs for Qwen3.5-4B), takes the first 16,000 conversations of
[UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft`), and answers every user turn
again with the target, one turn at a time, thinking on:

```bash
python -m recipes.dflash2.generate_data out_dir=/path/to/data
# writes regenerated.jsonl (every kept conversation), train.jsonl and heldout.jsonl
```

`data/sample.jsonl` holds 16 such conversations for smoke runs. It is too small to train a useful draft.

## Run

Write the data (above), then train for 580 steps, save every 100, and serve the final draft and measure its
acceptance on 128 held-out prompts. On 4 H200 GPUs training takes about 70 minutes:

```bash
python -m recipes.dflash2.train \
  data_path=/path/to/data/train.jsonl max_steps=580 save_every=100 \
  eval_prompts=/path/to/data/heldout.jsonl
```

Each save logs `draft_job_id=... draft_checkpoint_id=cp_...`. The checkpoints stay on the job's stage after the
training job ends, so you can measure any of them later, on an endpoint the script starts and stops
(`keep_job=true` leaves it running), or on a running one (`job_id=`):

```bash
python -m recipes.dflash2.evaluate prompts=heldout.jsonl \
  draft_job_id=TRAINING_JOB_ID draft_checkpoint_id=cp_...
```

A smoke run of the whole path, two steps on the bundled sample:

```bash
python -m recipes.dflash2.train \
  data_path=recipes/dflash2/data/sample.jsonl max_steps=2 train_batch_size=8 \
  eval_prompts=recipes/dflash2/data/sample.jsonl eval_num_prompts=8 eval_max_tokens=128
```

See [Run an example](../README.md#run-an-example) for the connection arguments.

## A draft for your fine-tune

A draft only speeds up the model it was trained against, so for a model you fine-tuned on Cortex Training, train
the draft against the fine-tune, not its base. Name the fine-tune in every step, and keep `model=` as the base it was
fine-tuned from (it still picks the catalog files, `drafts/<model>.json` and the tokenizer):

- `target_job_id=` and `target_checkpoint_id=`: a **weights-only** checkpoint the fine-tuning job saved. A full
  fine-tune and a LoRA fine-tune both work; a LoRA adapter is merged into its base for training.
- or `target_stage_path=@DB.SCHEMA.STAGE/path`: a Hugging Face directory you uploaded to a stage, as in
  [Start a Job From Your Own Weights](../../docs/guides/training/external-weights.md).

```bash
# 1. Answer the prompts with the fine-tune, so the draft trains on the fine-tune's own text.
python -m recipes.dflash2.generate_data out_dir=/path/to/data \
  target_job_id=SFT_JOB_ID target_checkpoint_id=cp_...

# 2. Train the draft against the fine-tune.
python -m recipes.dflash2.train data_path=/path/to/data/train.jsonl max_steps=580 save_every=100 \
  target_job_id=SFT_JOB_ID target_checkpoint_id=cp_... eval_prompts=/path/to/data/heldout.jsonl
```

Every checkpoint of such a draft carries the fine-tune under `target/`, and an endpoint that serves the draft serves
that fine-tune as its target. So `evaluate.py` takes the draft's checkpoint alone, exactly as for a draft of the base:

```bash
python -m recipes.dflash2.evaluate prompts=heldout.jsonl \
  draft_job_id=TRAINING_JOB_ID draft_checkpoint_id=cp_...
```

Each draft checkpoint is therefore the size of the fine-tune plus the draft. Resuming such a draft job from its own
`resumable` checkpoint is not supported: its later saves would not carry the fine-tune. Start a new job from the
fine-tune instead.

## Results

The example ran end to end on Cortex Training: `generate_data.py` (25 minutes on 8 H200 GPUs), then `train.py` for
580 steps of 64 conversations (63 minutes on 4 H200 GPUs), then serving and measuring the final draft.

![Draft cross-entropy and selector accepted length per step](training_curves.png)

Served mean acceptance length of the final draft, greedy, 128 prompts of up to 1024 new tokens, served on Cortex
Training by `evaluate.py`:

|                  | Mean acceptance length |
|------------------|------------------------|
| Our reproduction | 2.65                   |
| SpecForge        | 2.67                   |

The training set is small and multi-turn (about 7,000 usable conversations), so absolute acceptance is well below what
a production-sized, single-turn corpus gives.


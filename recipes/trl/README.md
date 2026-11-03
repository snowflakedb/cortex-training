# Run GRPO from TRL

[TRL](https://github.com/huggingface/trl) can run GRPO against Cortex Training.
`AsyncGRPOTrainer` drives the loop from your machine. Cortex owns the GPUs.
The local process builds a one-layer stub model so TRL can construct its
trainer, and it does not run a GPU forward.

The drivers live in a `trl_gsm8k/` tree kept next to a checkout of Arctic
Platform named `ap-trl-grpo-rc5`, rather than under `recipes/` here. That
checkout is
[Snowflake-AI-Research/Arctic-Platform](https://github.com/Snowflake-AI-Research/Arctic-Platform)
branch `truwase/trl-grpo-rc5` at `0fc11caf`. The training client is loaded
from `ap-trl-grpo-rc5/arctic_platform/integrations/trl/client.py` by file
path, so another `arctic_platform` earlier on `PYTHONPATH` does not replace
it.

| Piece | Role |
| --- | --- |
| `run_qwen3_gsm8k_grpo_trl_cortex.py` | GSM8K driver. Shared job, client, and trainer setup |
| `run_qwen3_bird_grpo_trl_cortex.py` | BIRD driver. Imports that setup and scores with the SQLite exec-match reward |
| `arctic_trl_cortex/` | Cortex `/forward` and `/fwd_bwd` adapters, and the rollout worker |
| `run_bird_32_zorro_vs_unpacked.sh` | Comparison launcher. Qwen3-1.7B, 32 prompts × 16 samples, 10 steps, 4 train + 4 sample |

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../docs/getting-started/setup.md), or run
`cortex-training login`, then confirm capacity:

```bash
cortex-training login
cortex-training capacity
```

The driver machine needs no GPU. GSM8K is loaded from Hugging Face
(`openai/gsm8k`, config `main`) at launch. BIRD reads a local parquet and
scores SQL against the SQLite databases named in each row, so those files
have to be readable on the driver. The reward implementation is
`bird_task.sql_reward` from Arctic Platform
`recipes/rl/trl/txt2sql`.

## Configure

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
export CUDA_VISIBLE_DEVICES=""
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
```

`CortexConfig` reads `ARCTIC_CORTEX_*`. The comparison launcher also accepts
`CORTEX_PAT` and copies it into `ARCTIC_CORTEX_PAT` when only one of the two
is set.

## Run

From `trl_gsm8k/`, the comparison runs ZoRRo and then unpacked, one Cortex
job at a time:

```bash
bash run_bird_32_zorro_vs_unpacked.sh
```

It refuses to start when a Cortex job is already live. `TRL_ALLOW_LIVE_JOBS=1`
continues anyway. Stop the launcher with SIGINT
(`kill -INT "$(cat logs/bird32.pid)"`).

The launcher fixes the shape below. ZoRRo uses DeepSpeed
`gradient_accumulation_steps=8`. Unpacked uses 32. Both arms pass
`--old-logprobs-source sampler`, `--max-staleness 0`, and `--seed 42`.

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3-1.7B` |
| Data | 32 BIRD prompts, 16 samples (512 sequences a step) |
| Steps | 10 |
| Layout | 4 training GPUs + 4 sampling GPUs |
| Lengths | sequence 36864, completion 4096, prompt budget 32768, token budget 102400 |
| Attention | FlashAttention-3, Liger, ZeRO-3, activation checkpointing, logits `compute` |
| Thinking | on (`ENABLE_THINKING=1`) |
| Image | `dev_20261018_230548_7d77f61f75c` |

GSM8K uses the same driver setup with its own defaults: completion 256,
sequence 1024, thinking off, and `NEUTRINO_IMAGE_TAG` left at the Python
default below. Launch it with `python -u run_qwen3_gsm8k_grpo_trl_cortex.py`.

## How a step is split

TRL takes one full batch per `forward_backward`
(`gradient_accumulation_steps=1` on `AsyncGRPOConfig`). DeepSpeed then splits
that batch. On the BIRD launcher the TRL batch is 512 sequences, across 4
training GPUs:

| Arm | DeepSpeed gas | Sequences per GPU per server call |
| --- | --- | --- |
| ZoRRo | 8 | 16 (one prompt group) |
| Unpacked | 32 | 4 |

The AdamW learning rate inside the DeepSpeed config is `1e-6`, betas
`0.9, 0.999`. `ArcticOptimizer` logs `AsyncGRPOConfig.learning_rate` and does
not apply that value on the server.

## Three arms

The 32-prompt script runs the first two.

- **ZoRRo**, the default. The job sets `zorro_train.enable`, `use_unpad`, and
  `pack=false`. The client sends the ZoRRo-shaped batch.
- **Unpacked, no ZoRRo** (`--unpacked-no-zorro`). The client still sends that
  ZoRRo-shaped batch. The server does not patch the model, and `pack=false`.
- **Packed** (`--no-zorro-train`, and optionally `--packed-single-call`).
  `pack` is left unset, so the image keeps its packed layout, and the driver
  sends one `fwd_bwd` instead of a no-grad forward plus a hooked backward.

ZoRRo forces FlashAttention-3. The packed arm uses `ARCTIC_ATTN_IMPL`, which
defaults to FlashAttention-3. `use_unpad` is set for ZoRRo, and for
FlashAttention-2 only. A packed FlashAttention-3 run leaves `use_unpad` false.
Unpacked no-ZoRRo also resolves FlashAttention-3 and leaves `use_unpad` false,
because the server patch is off.

## Compared with the VeRL 8B BIRD run

| | This BIRD script | VeRL 8B run |
| --- | --- | --- |
| Model | Qwen3-1.7B | Qwen3-8B |
| Old log-probs | sampler | recomputed on the trainer |
| Parallelism | ZeRO-3 | FSDP2, parameter and optimizer offload |
| Learning rate | `1e-6` | `2e-6` |
| Adam betas | `0.9`, `0.999` | `0.9`, `0.95` |
| Thinking | on | |

## Before the next comparison

- The inflight cap is `max(max_staleness, 1) * per_device_bsz * grad_accum`.
  On the ZoRRo arm that is `512 * 8 = 4096` samples, and the rollout queue is
  another `512 * 8`. With `max_staleness=0` the trainer drops stale groups.
  The sampler still generates about eight steps ahead.
- GSM8K does not set `learning_rate` on `AsyncGRPOConfig`, so TRL logs the
  TrainingArguments default (`5e-5`) while the server steps at `1e-6`. BIRD
  sets `1e-6`, so its log matches the step.
- The Python default image is `dev_20261014_182532_057b7b1ce3f`. The BIRD
  launcher overrides it to `dev_20261018_230548_7d77f61f75c`. A driver started
  without `NEUTRINO_IMAGE_TAG` uses the October 14 image.
- `--packed-single-call` is ignored on a ZoRRo run. `--unpacked-no-zorro`
  together with `--no-zorro-train` is rejected.

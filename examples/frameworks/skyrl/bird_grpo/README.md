# Run BIRD GRPO from SkyRL

[SkyRL](https://github.com/NovaSky-AI/SkyRL) can run GRPO on BIRD-SQL against
Cortex Training. SkyRL's trainer drives the loop from your machine while Cortex
runs training and sampling as separate sub-jobs, so the driver needs no local
GPU.

The example below trains Qwen3-8B with ZoRRo. It lives in the Arctic Platform
repository rather than under `examples/` here, and SkyRL's entry point drives it
rather than this client's. The launcher is on branch
`sfc-gh-truwase/zorro-train-client`
([Arctic Platform #161](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/161));
`main` does not have it yet. The file is named for Qwen3-0.6B. `MODEL` selects
the checkpoint, and the run below sets it to Qwen3-8B.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../../../docs/getting-started/setup.md) to create
`~/cortex-training-config.json` and log in, then confirm capacity:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

The driver machine needs no GPU. SQL scoring runs on the driver, so the SQLite
databases named in each row's `extra_info.db_path` must be readable there.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # if uv is not already on your PATH

git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git -C Arctic-Platform checkout sfc-gh-truwase/zorro-train-client
cd Arctic-Platform/recipes/rl/skyrl/txt2sql
```

## Configure

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
export ARCTIC_CORTEX_IMAGE_TAG=dev_20261026_083954_a6ba7da36f6
export CORTEX_ALLOW_LARGE_PAYLOAD=1
```

`ARCTIC_CORTEX_IMAGE_TAG` is the image this run was measured on. Without it the
client sends the release image. `CORTEX_ALLOW_LARGE_PAYLOAD=1` is required at
this batch: 128 prompts × 16 samples at prompt 36864 and response 4096 is over
the 128 MiB response check, and the forward/backward body is sent in 60 MiB
chunks.

## Data

`download_data.py` in this directory writes `$DATA_DIR/train.parquet` and
`val.parquet` from a raw BIRD checkout. Point `--bird_dir` at that checkout:

```bash
export DATA_DIR=$HOME/data/bird
uv run --isolated --no-project --with pandas --with pyarrow \
  python download_data.py --bird_dir /path/to/bird --output_dir "$DATA_DIR"
```

## Run

```bash
export MODEL=Qwen/Qwen3-8B
export NGPU_PER_NODE=4
export NUM_ENGINES=4
export TP_SIZE=1
export TRAIN_BSZ=128
export MINI_BSZ=128
export N_SAMPLES=16
export MICRO_BSZ=16
export PROMPT_LEN=36864
export RESPONSE_LEN=4096
export LR=1e-6
export EVAL_INTERVAL=1000
export EVAL_BEFORE_TRAIN=false
export LOGGER=console

bash run_qwen3_0.6b_bird_grpo_cortex.sh \
  trainer.max_training_steps=10 \
  trainer.arctic_rl.use_zorro=true \
  trainer.arctic_rl.pack=false \
  trainer.arctic_rl.balance_batch=true \
  trainer.arctic_rl.zero_stage=2
```

That is the measured shape: ZoRRo on, pack off, length-balanced batches,
ZeRO-2, FlashAttention-3, Liger, micro-batch 16. Omit
`trainer.arctic_rl.use_zorro=true` to run without ZoRRo. That path was not
part of the run below.

## Results

One 10-step run, SkyRL job `20261026T145558Z` on 2026-10-26. Ten steps, not a
full pass over the training set, and no
held-out evaluation (`EVAL_INTERVAL=1000`, `EVAL_BEFORE_TRAIN=false`). Reward
is the mean BIRD execution reward on the training batch. Pass@16 is the
fraction of prompts with at least one correct sample. Mean step time was
1152 s (generate 548 s, forward 241 s, update 347 s).

| Step | Reward | Pass@16 | Step (s) |
|---|---|---|---|
| 1 | 0.582 | 0.984 | 1341 |
| 2 | 0.567 | 0.977 | 1132 |
| 3 | 0.625 | 0.984 | 1118 |
| 4 | 0.653 | 0.984 | 1168 |
| 5 | 0.617 | 0.984 | 1025 |
| 6 | 0.563 | 0.984 | 1290 |
| 7 | 0.590 | 0.969 | 1087 |
| 8 | 0.645 | 1.000 | 1215 |
| 9 | 0.633 | 0.992 | 1040 |
| 10 | 0.664 | 1.000 | 1106 |

Mean reward over the ten steps is 0.614. Mean pass@16 is 0.986. One run, not a
guarantee.

## More detail

The launcher is
[`run_qwen3_0.6b_bird_grpo_cortex.sh`](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/sfc-gh-truwase/zorro-train-client/recipes/rl/skyrl/txt2sql/run_qwen3_0.6b_bird_grpo_cortex.sh).

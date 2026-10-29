# Run BIRD GRPO from VeRL

[VeRL](https://github.com/Snowflake-AI-Research/verl) can run GRPO on BIRD-SQL
against Cortex Training. VeRL's trainer drives the loop from your machine while
Cortex runs training and sampling as separate sub-jobs, so the driver needs no
local GPU.

The example below trains Qwen3-8B with ZoRRo. The launcher lives in the Arctic
Platform repository rather than under `examples/` here. It is on branch
`sfc-gh-truwase/zorro-train-client`
([Arctic Platform #161](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/161));
`main` does not have it yet. VeRL itself is
[Snowflake-AI-Research/verl](https://github.com/Snowflake-AI-Research/verl) at
`5a6491cf`, which is the commit that adds the remote-backend plugin hook.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../../docs/getting-started/setup.md) to create
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

git clone https://github.com/Snowflake-AI-Research/verl
git -C verl checkout 5a6491cfedf3d59ec37048177c7a0a1937299645
export VERL_HOME=$PWD/verl

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git -C Arctic-Platform checkout sfc-gh-truwase/zorro-train-client
cd Arctic-Platform/recipes/rl/verl/txt2sql
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

`ARCTIC_CORTEX_IMAGE_TAG` is required. The launcher refuses to start without
this image, because the release image does not match the unpacked forward this
run uses. `CORTEX_ALLOW_LARGE_PAYLOAD=1` is required at this batch: 128 prompts
× 16 samples at prompt 36864 and response 4096 is over the 128 MiB response
check, and the forward/backward body is sent in 60 MiB chunks.

## Data

The launcher reads `$DATA_DIR/train.parquet` and `val.parquet`. Build them with
the SkyRL downloader, which writes that layout:

```bash
export DATA_DIR=$HOME/data/bird
uv run --isolated --no-project --with pandas --with pyarrow \
  python ../../skyrl/txt2sql/download_data.py \
  --bird_dir /path/to/bird --output_dir "$DATA_DIR"
```

Set `DATA_DIR` yourself. The launcher's unset default is a developer path.

## Run

```bash
export MODEL=Qwen/Qwen3-8B
export TRAIN_BSZ=128
export MINI_BSZ=128
export N_SAMPLES=16
export PROMPT_LEN=36864
export RESPONSE_LEN=4096
export LR=1e-6
export MAX_STEPS=10
export TRAINING_GPUS=4
export SAMPLING_GPUS=4
export USE_ZORRO=1
export MICRO_BSZ=16
export BALANCE_BATCH=true
export PACK=false
export LOGGER=console

bash run_qwen3_8b_bird_grpo_cortex.sh
```

That is the measured shape: ZoRRo on, pack off, length-balanced batches,
ZeRO-2, FlashAttention-3, Liger, micro-batch 16. `USE_ZORRO` defaults to off.
The off path was not part of the run below.

If `uv run --isolated` fails because its pins ask for `numpy<2` and
`pyarrow>=19` together, set `PYBIN` to an interpreter that already has the
driver dependencies. The launcher then uses that interpreter for the preflight
and the trainer.

## Results

One 10-step run, VeRL job `20261027T111500Z` on 2026-10-27. Ten steps, not a
full pass over the training set, and no held-out evaluation. Reward is the
mean BIRD execution reward on the training batch (`critic/score/mean`). Mean
step time was 1005 s (generate 329 s, old log-prob 290 s, update 391 s).

| Step | Reward | Step (s) |
|---|---|---|
| 1 | 0.653 | 979 |
| 2 | 0.642 | 949 |
| 3 | 0.550 | 1001 |
| 4 | 0.595 | 1012 |
| 5 | 0.625 | 1035 |
| 6 | 0.547 | 1041 |
| 7 | 0.581 | 1061 |
| 8 | 0.642 | 1005 |
| 9 | 0.549 | 1011 |
| 10 | 0.604 | 958 |

Mean reward over the ten steps is 0.599. VeRL does not log pass@16. One run,
not a guarantee.

## More detail

The launcher is
[`run_qwen3_8b_bird_grpo_cortex.sh`](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/sfc-gh-truwase/zorro-train-client/recipes/rl/verl/txt2sql/run_qwen3_8b_bird_grpo_cortex.sh).

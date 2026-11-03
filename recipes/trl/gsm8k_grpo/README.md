# Run GRPO from TRL

[TRL](https://github.com/huggingface/trl) can run GRPO against Cortex
Training. TRL's `AsyncGRPOTrainer` drives the loop from your machine while
Cortex runs training and sampling as separate sub-jobs, so the driver needs
no local GPU.

The example below trains Qwen3-1.7B on GSM8K. It lives in the Arctic Platform
repository rather than under `recipes/` here, and TRL's entry point drives it
rather than this client's.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../../docs/getting-started/setup.md), or run
`cortex-training login` to set up a Snowflake connection profile, then confirm
capacity:

```bash
cortex-training login
cortex-training capacity
```

The driver machine needs no GPU. GSM8K is loaded from Hugging Face
(`openai/gsm8k`, config `main`) at launch.

## Install

```bash
git clone https://github.com/huggingface/trl
git -C trl checkout 3b24f2fcd85102893a9175483da2636eb49c1a22

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform

pip install -q uv
uv pip install -e '.[cortex]'
uv pip install -e ../trl
uv pip install math-verify datasets
```

`math-verify` is what TRL's `accuracy_reward` imports. The Cortex extra pulls
the SnowAPI client. It does not pull DeepSpeed or vLLM; those run inside the
Cortex sub-jobs.

## Configure

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
```

## Run

From `Arctic-Platform/recipes/rl/trl/gsm8k_cortex`:

```bash
bash run_qwen3_1.7b_gsm8k_grpo_cortex.sh
```

That is the 250-step recipe: Qwen3-1.7B, 8 samples per prompt, sequence length
1024, completion length 256, 32 prompts per step, seed 42, and old log
probabilities from the trainer. Thinking and ZoRRo are off. Sampling and
training each use 4 GPUs. The reward is TRL's `accuracy_reward`.

Stop the driver with `kill -INT`, not `SIGTERM`. The client cancels its Cortex
job on the way out.

## Results

A 250-step run finishes in about 30 minutes. The reward below is the mean
training-batch `accuracy_reward` over the 20 steps ending at that step, not a
held-out pass@1.

| Step | Reward |
|---|---|
| 20 | 0.43 |
| 40 | 0.62 |
| 60 | 0.63 |
| 80 | 0.75 |
| 100 | 0.79 |
| 120 | 0.77 |
| 140 | 0.73 |
| 160 | 0.80 |
| 180 | 0.76 |
| 200 | 0.80 |
| 220 | 0.85 |
| 240 | 0.76 |
| 250 | 0.81 |

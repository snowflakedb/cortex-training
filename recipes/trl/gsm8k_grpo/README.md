# Run GRPO from TRL

[TRL](https://github.com/huggingface/trl) can run GRPO against Cortex
Training. TRL's `AsyncGRPOTrainer` drives the loop from your machine while
Cortex runs training and sampling as separate sub-jobs, so the driver needs
no local GPU.

The example below trains Qwen3-1.7B on GSM8K. The driver lives in a
`trl_gsm8k/` tree next to an Arctic Platform checkout, rather than under
`recipes/` here, and TRL's entry point drives it rather than this client's.

## Prerequisites

A Cortex account with capacity for 2 GPUs: 1 for training and 1 for sampling.
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
git -C trl checkout 914196293543f609d0a8cb68b16154934e6b0c51

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform ap-trl-grpo-rc5
```

Keep `trl_gsm8k/` beside `ap-trl-grpo-rc5/`. The driver loads
`arctic_platform/integrations/trl/client.py` from that directory by name.

## Configure

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
```

## Run

From `trl_gsm8k/`:

```bash
python -u run_qwen3_gsm8k_grpo_trl_cortex.py
```

The default model is `Qwen/Qwen3-1.7B`, with 8 samples per prompt, sequence
length 1024, and completion length 256. Thinking is off. Sampling and training
each use one GPU. The reward is TRL's `accuracy_reward`.

## Results

A 250-step run finishes in about 28 minutes. The reward below is the mean
training-batch `accuracy_reward` over the 20 steps ending at that step.

That run used Qwen3-1.7B, 8 samples per prompt, sequence length 1024,
completion length 256, 4 GPUs for training and 4 for sampling, 32 prompts per
device, seed 42, and old log probabilities from the trainer. Thinking and
ZoRRo were off.

| Step | Reward |
|---|---|
| 20 | 0.39 |
| 40 | 0.60 |
| 60 | 0.61 |
| 80 | 0.74 |
| 100 | 0.74 |
| 120 | 0.75 |
| 140 | 0.76 |
| 160 | 0.78 |
| 180 | 0.83 |
| 200 | 0.76 |
| 220 | 0.83 |
| 240 | 0.81 |
| 250 | 0.77 |

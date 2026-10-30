# Reverse-text RL, following prime-rl

A port of [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)'s
`examples/basic/reverse-text`: an SFT warmup, then a short asynchronous RL run
that teaches Qwen3-0.6B to reverse text inside `<reversed_text>` tags. The default loss
is prime-rl's current one, IPO, as of commit `a1822f7a0`. `loss="dppo"` selects
the DPPO+KL loss it used until commit `0c58f007b`, and `loss="grpo"` a clipped
GRPO loss for comparison.

| Stage | Entry point | prime-rl config |
| --- | --- | --- |
| SFT warmup | `sft.py` | `sft.toml`: 100 steps, 32 packed rows of 4096 tokens, lr 2e-5 |
| RL | `train.py` | `rl.toml`: 20 steps, 128 rollouts (8 prompts x 16), 128 completion tokens, lr 3e-6 |

## Task

The verifiers `reverse-text` taskset, in [`task.py`](task.py):

- System prompt: `Reverse the text character-by-character. Put your answer in <reversed_text> tags.`
- Prompts: `PrimeIntellect/Reverse-Text-RL` (train split), taken in source order
- Reward: the `difflib` LCS ratio between the `<reversed_text>` content of the
  reply (the text after the first `</think>`) and the reversed prompt

The SFT data, `willcb/R1-reverse-wikipedia-paragraphs-v1-1000`, asks for
`<answer>` tags, so RL also has to learn the `<reversed_text>` format, as in
prime-rl. `sft.py` tokenizes each reply as the dataset writes it, as prime-rl
does; see [SFT tokenization](#sft-tokenization).

## Install and run

Install the client and the example dependencies as described in the
[example prerequisites](../../README.md#prerequisites), with Python 3.11 or
newer. Then run the SFT warmup:

```bash
python -m examples.rl.async_rl_reverse_text.sft config=/path/to/config.json
```

The SFT run ends by saving a weights-only and a resumable checkpoint and logging
the RL command. Start RL from them:

```bash
python -m examples.rl.async_rl_reverse_text.train \
  config=/path/to/config.json \
  init_source_job_id=SFT_JOB_ID \
  init_checkpoint_id=WEIGHTS_ONLY_CHECKPOINT_ID \
  init_trainer_checkpoint_id=RESUMABLE_CHECKPOINT_ID
```

`config=` is a JSON connection file; `connection=<profile>` uses a profile
instead. 

## Knobs

| Option | Default | prime-rl field |
| --- | --- | --- |
| `max_steps` | `20` | `max_steps` |
| `batch_size` | `128` rollouts, a multiple of `group_size` | `orchestrator.batch_size` |
| `group_size` | `16` | `orchestrator.group_size` |
| `max_completion_tokens` | `128` | `orchestrator.train.sampling.max_completion_tokens` |
| `max_off_policy_steps` | `8` | `orchestrator.max_off_policy_steps` |
| `initial_inflight` | `28` rollouts | `orchestrator.concurrency.initial_inflight`; unset in the example, so prime-rl derives 27-28 from vLLM's KV cache |
| `inflight_growth` | `1.2` for `ipo`, `1.25` for `dppo` and `grpo` | the concurrency controller's growth per turnover; `1.25` at `0c58f007b` |
| `max_inflight` | `1024` rollouts | `orchestrator.concurrency.max_inflight` |
| `loss` | `ipo` (or `dppo`, `grpo`) | `trainer.loss.type = "ipo"`; `"default"` at `0c58f007b` for `dppo`; prime-rl has no `grpo` |
| `ipo_eps` | `0.3`, `ipo` only | `trainer.loss.eps` |
| `dppo_mask_low`, `dppo_mask_high` | `0.2`, `dppo` only | `trainer.loss.dppo_mask_low/high` at `0c58f007b` |
| `eps_clip` | `0.2`, `grpo` only | none |
| `adv_tau` | `1.0` | `trainer.loss.adv_tau` |
| `kl_tau` | `0` for `ipo`, `1e-3` for `dppo` and `grpo` | `trainer.loss.kl_tau` |
| `constant_batch_size` | on for `ipo`, off for `dppo` and `grpo` | `orchestrator.constant_trainer_batch_size`; off is `orchestrator.train.filter_zero_advantages` at `0c58f007b` |
| `lr` | the job's (3e-6) | `trainer.optim.lr` (3e-6) |
| `eval_examples`, `eval_rollouts`, `eval_max_tokens` | `20`, `3`, `1024` | `eval -n 20 -r 3 --sampling.max-tokens 1024` |

The SFT stage takes `max_steps` (100), `batch_size` (32 packed rows),
`seq_len` (4096), `lr` (the job's, 2e-5), `shuffle`, and `seed`. Both stages
also take the job arguments every example takes: `model` (`Qwen/Qwen3-0.6B`),
`context` (`standard`), `hardware` (`h200`), `lora` (`false`), `job_config`,
`train_batch_size`, and `max_seq_len`. Only the defaults are
validated.

## Expected results

### Benchmark

The eval is prime-rl's reverse-text eval, run in the job as `eval/reward/mean`:
the first 20 prompts, up to 1024 tokens, temperature 1.0, and the LCS reward.
prime-rl reports about 0.05 for the base model and about 0.8 after SFT and RL.

| Checkpoint | This example on Cortex Training | prime-rl example                        |
| --- |--------------------------------|-----------------------------------------|
| Base Qwen3-0.6B | 0.090 ± 0.005                  | 0.093 ± 0.005                           |
| After SFT and 20 RL steps, `ipo` against `a1822f7a0` | 0.799-0.833, mean 0.816        | 0.801-0.826, mean 0.820                 |
| After SFT and 20 RL steps, `dppo` against `0c58f007b` | 0.807-0.832, mean 0.820        | 0.802-0.827, mean 0.815                 |

### Reward curves

![Training reward per RL step, one panel per loss: under ipo, this example on Cortex Training follows prime-rl's example at a1822f7a0 within the spread between runs; under dppo, it runs ahead of prime-rl at 0c58f007b over steps 4 to 8 and matches it from step 9](reward_curves.png)


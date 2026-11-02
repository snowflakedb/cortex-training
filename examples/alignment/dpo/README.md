# Direct Preference Optimization (DPO)

Train on preference data: a prompt with a **chosen** and a **rejected** answer.
DPO raises the model's probability of the chosen answer relative to the rejected
one, measured against a frozen reference model, with no reward model and no
sampling.

This example computes the DPO loss in this process with
[`forward_backward_custom`](../../../docs/reference/python-sdk.md#custom-losses).
`train.py` is the entry point, and `task.py` holds the data and the loss.

## How it works

1. **Items.** Each pair becomes two adjacent items, chosen then rejected. An
   item's `loss_mask` selects the log-probabilities of the final assistant answer
   only.
2. **Reference.** Before the first update, a no-gradient forward pass scores
   every item: `ref_logp` is its summed answer log-probability. The reference is
   the model the job starts from, so no second model is needed.
3. **Loss.** For each pair, with `s` the policy's and `r` the reference's score:

   ```
   margin = (s_chosen - r_chosen) - (s_rejected - r_rejected)
   loss   = -log sigmoid(beta * margin)
   ```

   `L` is the mean over the step's pairs.
4. **Step.** `forward_backward_custom(job_id, data, dpo_loss(beta))` applies the
   gradient of `L`, then `step` updates the weights.

## Run

The default is Qwen3-8B with LoRA at standard context on H200, on the small
built-in `smoke` set:

```bash
python -m examples.alignment.dpo.train connection=NAME
```

On UltraFeedback, with a larger learning rate and batch than the smoke defaults:

```bash
python -m examples.alignment.dpo.train connection=NAME \
  dataset=ultrafeedback max_steps=50 pairs_per_step=16 lr=5e-5 eval_pairs=32 eval_every=10
```

On Qwen3-8B LoRA this took held-out loss from 0.693 to about 0.56 and held-out
accuracy to about 0.72 in 50 steps. With `lr=1e-5` and 8 pairs per step,
learning was weak and noisy.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `model=`, `context=`, `hardware=`, `lora=` | `Qwen/Qwen3-8B`, `standard`, `h200`, `true` | Which catalog file to use |
| `lr=` | `1e-5` | Learning rate; `lr=None` keeps the job's |
| `dataset=` | `smoke` | `smoke` (12 training and 4 held-out pairs) or `ultrafeedback` (`HuggingFaceH4/ultrafeedback_binarized`) |
| `beta=` | `0.1` | How far the policy may move from the reference |
| `pairs_per_step=` | `4` | Pairs per step; a step sends twice as many items |
| `max_steps=`, `epochs=` | `20`, `1` | Training steps, and passes over the training pairs |
| `eval_pairs=`, `eval_every=` | `8`, `5` | Held-out pairs, and how often to evaluate them |
| `max_pair_tokens=` | `2048` | Skip pairs with a longer chosen or rejected conversation |
| `job_config=` | none | Use this job file instead of the catalog |
| `wandb_project=`, `sf_tracking=` | off | [Experiment tracking](../../../docs/guides/operations/wandb.md) |

## Expected results

- **`train_loss` starts at `log 2` ≈ 0.693.** The policy equals the reference at
  step 0, so every margin is 0; the example warns if it does not.
- **`reward_accuracy` rises** (the fraction of pairs where the chosen answer
  gains more than the rejected one), and **`reward_margin` grows.**
- **The `eval_` metrics** show the same on held-out pairs, before training,
  every `eval_every` steps, and at the end.
- **`logprob_max_abs_diff` stays near 0.** The two passes of each step saw the
  same weights.

## Troubleshooting

- `does not support custom losses yet`: the service doesn't have the server
  loss `weighted_cross_entropy` yet.
- A request needs at least as many items as the training sub-job has
  data-parallel ranks; raise `pairs_per_step=` if the server rejects a step.
- `first-step loss ... is not log 2`: the reference scores don't match the
  starting model, for example because `job_id=` attached a job that has already
  been trained.

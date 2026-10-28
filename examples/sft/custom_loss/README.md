# Supervised Fine-Tuning with a Custom Loss

Train on a loss you write in PyTorch, with
[`forward_backward_custom`](../../../docs/reference/python-sdk.md#custom-losses).
This example fine-tunes on the [conversational SFT](../README.md) datasets with
the focal loss, which no built-in loss provides: a token the model gives
probability `p` contributes `-(1 - p) ** gamma * log p`, so tokens it already
predicts well count less. Each sequence's loss is the mean over its assistant
tokens, and the custom loss `L` is the mean over sequences.

`train.py` is the entry point and `task.py` holds the loss. Each step calls
`forward_backward_custom(job_id, data, focal_loss(gamma))`: a forward pass
returns the per-token log-probabilities, `loss_fn` computes `L` in this process,
and a backward pass applies its gradient. Then `step` updates the weights. The
job is built like the SFT example's
([How an example builds its job](../../README.md#how-an-example-builds-its-job)).

## Run

The default is Qwen3-8B full-parameter at standard context on H200, on the
`who_trained_you` dataset, for 100 steps:

```bash
python -m examples.sft.custom_loss.train config=/path/to/config.json
```

## Options

| Argument | Default | Meaning |
|---|---|---|
| `model=`, `context=`, `hardware=`, `lora=` | `Qwen/Qwen3-8B`, `standard`, `h200`, `false` | Which catalog file to use |
| `lr=`, `train_batch_size=`, `max_seq_len=` | from the job | Override the job for this run |
| `dataset=` | `who_trained_you` | A conversational dataset; see [Datasets](../README.md#datasets) |
| `focal_gamma=` | `2.0` | The focal loss's `gamma`. `0` is the per-sequence mean negative log-likelihood |
| `max_steps=` | 100 | Training steps |
| `job_config=` | none | Use this job file instead of the catalog and `overrides.json` |
| `wandb_project=`, `sf_tracking=` | off | [Experiment tracking](../../../docs/guides/operations/wandb.md) |

To train with your own loss, replace `focal_loss` in `task.py`. A `loss_fn`
takes `(data, logprobs)` and returns `(loss, metrics)`: `data` is the list of
items passed to `forward_backward_custom`, and `logprobs[i][j]` is the
log-probability of token `j + 1` of item `i`.

## Expected results

Each step logs `train_loss` (the value of `L`), `train_nll` (the token-mean
negative log-likelihood over assistant tokens) and `loss_weight_sum`. On the
default task both losses drop quickly. On a mixture-of-experts model, also watch
`logprob_max_abs_diff`: the forward and backward passes can route tokens
differently.

After the final save, the example prints a generate command for the
checkpoint, as the [SFT example](../README.md#expected-results) does.

## Troubleshooting

- `does not support custom losses yet`: the service doesn't have the server
  loss `weighted_cross_entropy` yet.
- A batch needs at least as many sequences as the training sub-job has
  data-parallel ranks; raise `train_batch_size=` if the server rejects it.

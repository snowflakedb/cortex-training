# Snowflake Experiment Tracking

Training recipes can log metrics and parameters to Snowflake's native experiment
tracking. Results are viewable in Snowsight under **AI & ML > Experiments**.

## Prerequisites

Install `snowflake-ml-python` (>= 1.19.0):

```bash
uv pip install "snowflake-ml-python>=1.19.0"
```

## Usage

Set `sf_tracking=True` on the train command:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json sf_tracking=True
```

## How it works

When a recipe creates a Cortex Training job, the server associates it with a
Snowflake experiment and run. The recipe retrieves the experiment and run names,
opens a Snowpark session using the same client credentials, and logs training
hyperparameters and per-step metrics.

## Viewing results

When a run ends, URLs for the experiment and run are printed to stdout. You can
also browse experiments in Snowsight: **AI & ML > Experiments**.

Metrics are also logged locally (and to Weights & Biases if `wandb_project` is
set), so all backends receive the same data.

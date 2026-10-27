# Logs and Metrics

Download complete execution logs:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/logs
```

Download persisted stdout/stderr reconstructed as one file per sub-job:

```bash
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs --resume
```

`--resume` continues a stdout download already in that directory, including
one that was interrupted, and fetches only what is missing. It does not
apply to execution logs. On a terminal, stdout and GPU metrics progress is
one stderr line that updates in place. A redirected stderr gets one final line
per sub-job. The JSON result stays on stdout.

Download reconstructed GPU metrics:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

GPU and host metrics are written to `<output_dir>/<sub_job_id>/gpu.jsonl`. A
training sub-job's **training metrics** — the loss, gradient norm, learning
rate and whatever else its loss function returns — go to
`<output_dir>/<sub_job_id>/training.jsonl`, one row per `/fwd-bwd` call and
one per `/step`, each with the numbers that call returned. See
[GPU metrics download](../../reference/rest-api.md#124-gpu-metrics-download)
and
[Training metrics download](../../reference/rest-api.md#125-training-metrics-download)
for every row's fields, and `training_config.save_training_metrics` to opt out.

Open a job dashboard in the terminal:

```bash
cortex-training tui JOB_ID
```

The **Logs** tab tails a running job or loads its saved console after it
finishes. The **Resources** tab charts the selected sub-job's GPU use and lists
each GPU; `w` cycles 10 minutes (the default), 30 minutes, 1 hour, 6 hours, or
the whole job, and `r` refreshes. The **Training metrics** tab charts what the
training sub-job's calls returned, per step (`grad_norm`, `last_lr`, …) and,
below that, per `/fwd-bwd` call (`avg_loss`, …); `a` also shows the bookkeeping
numbers it hides, such as `rank` and `global_steps`. Run `cortex-training tui` without a job id
to browse and search your jobs first. See [Job TUI](../../reference/cli.md#job-tui).

Example-level metrics are also written under each example's `log_path`; see
[Experiment tracking](wandb.md) to mirror them elsewhere.

Snowflake profile and PAT clients also emit best-effort client-side operation
metrics over Snowflake's OTLP endpoint. Failures of essential SDK methods, the
outcome and timing of every download, and the outcome and timing of every call
the TUI makes are emitted automatically; other successful outcomes require
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1`. Each metric says whether the SDK,
the CLI or the TUI made the call (`client.surface`). See
[Client metrics](../../reference/python-sdk.md#client-metrics).

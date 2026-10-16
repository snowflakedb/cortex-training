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

`--resume` continues a stdout download already in that directory. It does not
apply to execution logs. On a terminal, stdout and GPU metrics progress is
one stderr line that updates in place. A redirected stderr gets one final line
per sub-job. The JSON result stays on stdout.

The TUI's **Logs** tab (below) keeps a sub-job's newest 20,000 lines (fewer
when long lines wrap to many rows) and says so once it drops older ones;
`download-log --log-type stdout` has the whole console. A job that fails while
it is created or starts up may have no saved console; the tab then shows the
error records from that sub-job's execution log, the file plain `download-log`
downloads. While a job is pending or starting, the tab says it is waiting.
Reading a running job's logs currently counts as activity on the job, so once
the job runs, the live tail pauses after 15 minutes with no input (`p` resumes
it). See [Logs](../../reference/cli.md#logs).

Download reconstructed GPU metrics:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

Metrics are written to `<output_dir>/<sub_job_id>/gpu.jsonl`.

Open a job dashboard in the terminal:

```bash
cortex-training tui JOB_ID
```

It opens on the **Logs** tab, which tails a running job or loads its saved
console after it finishes. The **Resources** tab shows the selected sub-job's
GPU type, requested and reporting GPUs, and two charts kept apart: GPU busy
(kernel-active %) and GPU memory used %, each with its min, average, max and
latest for the window and the whole job. It lists each GPU too; `w` cycles 10
minutes (the default), 30 minutes, 1 hour, 6 hours, or the whole job, and `r`
refreshes. Run `cortex-training tui`
without a job id to browse and search your jobs first: those in the configured
database and schema, and with `a` those in every database and schema you can
see where the service has the account-level list (`--schema-only` hides `a`). See
[Job TUI](../../reference/cli.md#job-tui).

Recipe-level metrics are also written under each recipe's `log_path`.

Snowflake profile and PAT clients also emit best-effort client-side operation
metrics over Snowflake's OTLP endpoint. Failures of essential SDK methods are
emitted automatically; successful outcomes require
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1`. See
[Client metrics](../../reference/python-sdk.md#client-metrics).

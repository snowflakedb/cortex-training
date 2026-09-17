# Cortex Training Client

Python SDK, command-line tools, runnable recipes, and documentation for the
Cortex Training REST API.

## Start Here

- [Follow the getting-started path](docs/getting-started/README.md)
- [Run a quick supervised fine-tuning job](docs/getting-started/first-sft-run.md)
- [Browse runnable recipes](recipes/README.md)
- [Check model and training-method compatibility](docs/reference/model-compatibility.md)
- [Use the CLI and Python client](docs/reference/cli.md)
- [Read the REST API reference](docs/reference/rest-api.md)

## Install

Requires Python 3.10 or later.

This project uses [uv](https://docs.astral.sh/uv/). Install it first if you
have not:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

For local development:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
uv pip install -e .
```

(`pip` works in place of `uv pip` throughout if you prefer.)

The package installs:

- `cortex-training`, for submitting and managing jobs
- `cortex-training tui`, for viewing job logs
- `cortex_training`, the Python SDK

Verify the command entry points:

```bash
cortex-training --help
cortex-training tui --help
```

## Usage

```bash
cortex-training list
cortex-training submit examples/api/training.json
cortex-training tui JOB_ID
```

```python
from cortex_training import CortexTrainingClient, SubJobConfig
```

Connection settings use `CORTEX_TRAINING_*` and `SNOWFLAKE_*` environment
variables. Login state is stored under `~/.config/cortex-training/`, and TUI
cache state is stored under `~/.cache/cortex-training/`, unless their existing
override variables are used.

See the [CLI reference](docs/reference/cli.md) for commands and configuration.

## Repository Map

| Path | Purpose |
|---|---|
| `model-catalog/` | Supported models, context limits, and recommended job profiles |
| `docs/` | Getting started material, concepts, guides, and reference |
| `recipes/` | End-to-end training, sampling, and evaluation workflows |
| `examples/api/` | Small JSON examples for individual API operations |
| `examples/config/` | Connection configuration templates |
| `src/cortex_training/` | Installable Python client |
| `tests/` | Client and CLI tests |
| `cluster-status.py` | Optional watch view of running jobs and GPU usage |

`cluster-status.py` is a standalone helper that shells out to `cortex-training
list`; it needs a working connection but is not imported by the package. It
refreshes every 5 seconds by default:

```bash
python cluster-status.py            # live view
python cluster-status.py --once     # print one summary and exit
```

## Plot GPU metrics

Use the same connection as the rest of this README (`cortex-training login`,
or `--host` / `--pat` / `--database`). Then:

```bash
cortex-training download-metrics JOB_ID --output-dir ./metrics
```

The command prints JSON. Each saved file is listed under `metrics` as
`saved_path`. An empty `metrics` array means the job has no GPU samples yet —
there will be nothing to plot.

Otherwise each sub-job writes `./metrics/<sub_job_id>/gpu.jsonl`. Directory
names include colons (`JOB_ID:training:0`, `JOB_ID:sampling:0`). One line is
one GPU during one sample interval. Metric columns are gauges (use them as-is;
do not difference consecutive rows). A JSON `null` means the value was not
measured, which is distinct from `0.0`. A column may be present and still
all-null.

Example line:

```json
{"record_version": 2, "sub_job_id": "JOB_ID:training:0", "sample_ts": "2026-09-16T00:00:00.000000000Z", "worker_num": 0, "local_gpu_index": 0, "global_gpu_index": 0, "gpu_uuid": "gpu-00000000-0000-0000-0000-000000000000", "gpu_util_pct": 87.5, "fb_used_bytes": 21474836480, "fb_free_bytes": 6442450944, "power_watts": 320.0, "gpu_temp_c": 62.0, "sm_active_pct": null, "sm_occupancy_pct": null, "tensor_active_pct": 40.0, "dram_active_pct": 30.0, "seq": 12, "seq_epoch": "00000000000000000000000000000000", "sample_interval_s": 5.0}
```

| Field | Meaning |
|---|---|
| `record_version` | JSONL schema version |
| `sub_job_id` | Sub-job that produced the sample |
| `sample_ts` | UTC scrape time (ISO-8601). Consecutive samples are `sample_interval_s` apart |
| `worker_num` | GPU worker index in the sub-job, `0..n-1` |
| `local_gpu_index` | GPU index inside that worker, `0..k-1` |
| `global_gpu_index` | Sub-job-wide GPU id. **Group and plot by this column** |
| `gpu_uuid` | Device UUID when the runtime provides it; otherwise omitted or `null` |
| `gpu_util_pct` | GPU utilization, percent |
| `fb_used_bytes` / `fb_free_bytes` | Frame-buffer used and free, bytes |
| `power_watts` | Board power, watts |
| `gpu_temp_c` | GPU temperature, Celsius |
| `sm_active_pct` | SM active, percent (may be null) |
| `sm_occupancy_pct` | SM occupancy, percent (may be null) |
| `tensor_active_pct` | Tensor pipeline active, percent (may be null) |
| `dram_active_pct` | DRAM active, percent (may be null) |
| `seq` | Monotonic sample counter within `seq_epoch` |
| `seq_epoch` | Opaque counter generation. `seq` restarts when this value changes |
| `sample_interval_s` | Sampling period, seconds |

Do not group by `local_gpu_index` alone: that id repeats on every worker, so
the series would mix distinct GPUs. If you load more than one `gpu.jsonl`,
also split by `sub_job_id` — `global_gpu_index` is unique per sub-job, not
across the whole job.

Optional plotting dependencies:

```bash
uv pip install pandas matplotlib
```

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

paths = sorted(Path("./metrics").glob("*/gpu.jsonl"))
if not paths:
    raise SystemExit("no gpu.jsonl under ./metrics")

frames = [pd.read_json(path, lines=True) for path in paths]
df = pd.concat(frames, ignore_index=True)
df["sample_ts"] = pd.to_datetime(df["sample_ts"], utc=True)

fig, ax = plt.subplots(figsize=(10, 4))
for (sub_job, gpu_id), series in df.groupby(["sub_job_id", "global_gpu_index"]):
    series = series.sort_values("sample_ts")
    ax.plot(
        series["sample_ts"],
        series["gpu_util_pct"],
        label=f"{sub_job} GPU {gpu_id}",
    )
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("GPU utilization (%)")
ax.legend()
plt.tight_layout()
plt.show()
```

Swap `gpu_util_pct` for another non-null metric column.

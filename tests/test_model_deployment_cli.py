from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

import cortex_training._cli as cli

JOB_ID = "12345678-1234-4234-8234-123456789abc"
DEPLOYMENT_ID = f"sf-ct-{JOB_ID}"
SUB_JOB_ID = f"{JOB_ID}:sampling:0"


class DeploymentClient:
    model_deployment_base_url = (
        "http://test.local/api/v2/databases/DB/schemas/SCH/"
        "cortex-training/inference/v1"
    )

    def __init__(self):
        self.created_spec = None
        self.waited_job_id = None
        self.described_id = None
        self.dropped_job_id = None
        self.log_request = None
        self.metrics_request = None

    def create_model_deployment(self, spec):
        self.created_spec = spec
        return {
            "deployment": DEPLOYMENT_ID,
            "deployment_class": "EVALUATION",
            "status": "PENDING",
            "model_source": spec["source"],
            "max_seq_len": spec["inference"]["max_seq_len"],
            "gpus_per_replica": spec["deployment"]["gpus_per_replica"],
            "replicas": spec["deployment"]["min_replicas"],
            "base_url": self.model_deployment_base_url,
        }

    def list_model_deployments(self, status=None):
        self.list_status = status
        return [self.ready_metadata()]

    def describe_model_deployment(self, deployment_id):
        self.described_id = deployment_id
        return {
            "deployment": deployment_id,
            "status": "READY",
            "base_url": self.model_deployment_base_url,
            "database": "DB",
            "schema": "SCH",
            "created_at": "2026-09-16T20:00:00Z",
            "creator": "CURRENT_USER",
            "owner_role": "CURRENT_OWNER_ROLE",
            "spec": {
                "deployment": {
                    "class": "EVALUATION",
                    "min_replicas": 3,
                    "gpus_per_replica": 2,
                    "gpu_type": "H200",
                },
                "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
                "inference": {"max_seq_len": 8192},
            },
        }

    def wait_for_model_deployment(self, deployment_id):
        self.waited_job_id = deployment_id
        return self.ready_metadata(include_base_url=True)

    def drop_model_deployment(self, deployment_id):
        self.dropped_job_id = deployment_id
        return {"deployment": deployment_id, "dropped": True}

    def download_model_deployment_logs(self, deployment_id, output_dir, *, log_type):
        self.log_request = (deployment_id, output_dir, log_type)
        path = Path(output_dir) / "execution.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return {
            "deployment": deployment_id,
            "logs": [{"filename": path.name, "saved_path": str(path)}],
        }

    def download_model_deployment_metrics(self, deployment_id, output_dir):
        self.metrics_request = (deployment_id, output_dir)
        path = Path(output_dir) / "gpu.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return {
            "deployment": deployment_id,
            "metrics": [
                {"filename": path.name, "saved_path": str(path), "chunk_count": 1}
            ],
        }

    def ready_metadata(self, *, include_base_url=False):
        metadata = {
            "deployment": DEPLOYMENT_ID,
            "deployment_class": "EVALUATION",
            "status": "READY",
            "model_source": {
                "type": "base_model",
                "model": "Qwen/Qwen3-1.7B",
            },
            "max_seq_len": 8192,
            "gpus_per_replica": 2,
            "replicas": 3,
            "created_at": "2026-09-16T20:00:00Z",
            "owner_role": "CURRENT_OWNER_ROLE",
        }
        if include_base_url:
            metadata["base_url"] = self.model_deployment_base_url
        return metadata


def write_spec(tmp_path: Path, *, checkpoint: bool = False) -> Path:
    source = (
        {
            "type": "checkpoint",
            "job_id": "source-training-job",
            "checkpoint_id": "cp_completed_weights_only",
        }
        if checkpoint
        else {"type": "base_model", "model": "Qwen/Qwen3-1.7B"}
    )
    path = tmp_path / "deployment.json"
    path.write_text(
        json.dumps(
            {
                "deployment": {
                    "class": "EVALUATION",
                    "min_replicas": 3,
                    "gpus_per_replica": 2,
                    "gpu_type": "H200",
                },
                "source": source,
                "inference": {"max_seq_len": 8192},
            }
        )
    )
    return path


def run_cli(client, *arguments):
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = cli.main(
        ["--base-url", "http://test.local", *arguments],
        client_factory=lambda _args: client,
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue()


def test_create_accepts_documented_flag_order_and_prints_only_public_metadata(tmp_path):
    client = DeploymentClient()
    spec_path = write_spec(tmp_path, checkpoint=True)

    result, stdout, stderr = run_cli(
        client,
        "model-deployment",
        "create",
        "--database",
        "DB",
        "--schema",
        "SCH",
        "--spec",
        str(spec_path),
    )

    assert result == 0
    assert stderr == ""
    assert client.created_spec["source"]["type"] == "checkpoint"
    assert client.created_spec["deployment"]["min_replicas"] == 3
    output = json.loads(stdout)
    assert output == {
        "deployment": DEPLOYMENT_ID,
        "status": "PENDING",
        "base_url": client.model_deployment_base_url,
    }
    assert JOB_ID not in stdout.replace(DEPLOYMENT_ID, "")
    assert SUB_JOB_ID not in stdout


def test_create_wait_supports_deprecated_spec_file(tmp_path):
    client = DeploymentClient()
    spec_path = write_spec(tmp_path)

    result, stdout, stderr = run_cli(
        client,
        "model-deployment",
        "create",
        "--database",
        "DB",
        "--schema",
        "SCH",
        "--spec-file",
        str(spec_path),
        "--wait",
    )

    assert result == 0
    assert stderr == ""
    assert client.waited_job_id == DEPLOYMENT_ID
    assert json.loads(stdout) == {
        "deployment": DEPLOYMENT_ID,
        "status": "READY",
        "base_url": client.model_deployment_base_url,
    }
    assert SUB_JOB_ID not in stdout
    assert "max_inflight_choices" not in stdout


def test_create_wait_prints_created_deployment_when_polling_fails(tmp_path):
    client = DeploymentClient()
    spec_path = write_spec(tmp_path)
    stderr = io.StringIO()

    class FlushTrackingOutput(io.StringIO):
        flushed = False

        def flush(self):
            self.flushed = True
            super().flush()

    stdout = FlushTrackingOutput()

    def fail_wait(deployment_id):
        client.waited_job_id = deployment_id
        raise RuntimeError("polling failed")

    client.wait_for_model_deployment = fail_wait
    result = cli.main(
        [
            "--base-url",
            "http://test.local",
            "model-deployment",
            "create",
            "--database",
            "DB",
            "--schema",
            "SCH",
            "--spec",
            str(spec_path),
            "--wait",
        ],
        client_factory=lambda _args: client,
        stdout=stdout,
        stderr=stderr,
    )

    assert result == 1
    assert client.waited_job_id == DEPLOYMENT_ID
    assert stdout.flushed
    assert json.loads(stdout.getvalue()) == {
        "deployment": DEPLOYMENT_ID,
        "status": "PENDING",
        "base_url": client.model_deployment_base_url,
    }
    assert stderr.getvalue() == "error: polling failed\n"


def test_create_wait_prints_created_deployment_when_interrupted(tmp_path):
    # Ctrl+C raises KeyboardInterrupt, which is not an Exception subclass.
    client = DeploymentClient()
    spec_path = write_spec(tmp_path)

    def interrupt_wait(deployment_id):
        client.waited_job_id = deployment_id
        raise KeyboardInterrupt

    client.wait_for_model_deployment = interrupt_wait
    result, stdout, stderr = run_cli(
        client,
        "model-deployment",
        "create",
        "--database",
        "DB",
        "--schema",
        "SCH",
        "--spec",
        str(spec_path),
        "--wait",
    )

    assert result == 130
    assert client.waited_job_id == DEPLOYMENT_ID
    assert json.loads(stdout)["deployment"] == DEPLOYMENT_ID
    assert stderr.startswith(f"interrupted; {DEPLOYMENT_ID} is still running")
    # The deployment is left running for the user to keep or drop.
    assert client.dropped_job_id is None


def test_wait_interrupted_names_the_running_deployment_and_exits_130():
    client = DeploymentClient()

    def interrupt_wait(_deployment_id):
        raise KeyboardInterrupt

    client.wait_for_model_deployment = interrupt_wait
    result, stdout, stderr = run_cli(
        client, "model-deployment", "wait", DEPLOYMENT_ID, "--database", "DB", "--schema", "SCH"
    )

    assert result == 130
    assert stdout == ""
    assert "Traceback" not in stderr
    assert stderr.startswith(f"interrupted; {DEPLOYMENT_ID} is still running")
    assert f"model-deployment drop {DEPLOYMENT_ID}" in stderr


def test_create_wait_failure_prints_the_last_observed_status(tmp_path):
    client = DeploymentClient()
    spec_path = write_spec(tmp_path)

    class WaitFailed(RuntimeError):
        deployment = {
            "deployment": DEPLOYMENT_ID,
            "status": "FAILED",
            "failure": {"code": "sub_job_failed", "message": "out of memory"},
        }

    def fail_wait(_deployment_id):
        raise WaitFailed(f"model deployment {DEPLOYMENT_ID} is FAILED (sub_job_failed)")

    client.wait_for_model_deployment = fail_wait
    result, stdout, stderr = run_cli(
        client, "model-deployment", "create", "--spec", str(spec_path), "--wait",
        "--database", "DB", "--schema", "SCH",
    )

    assert result == 1
    # Not the stale PENDING returned by create.
    assert json.loads(stdout) == WaitFailed.deployment
    assert "is FAILED (sub_job_failed)" in stderr


def test_create_dry_run_validates_and_prints_the_spec_without_a_client(tmp_path, monkeypatch):
    spec_path = tmp_path / "deployment.json"
    spec_path.write_text(
        json.dumps(
            {
                "deployment": {
                    "class": "EVALUATION",
                    "min_replicas": 1,
                    "gpus_per_replica": 1,
                    "gpu_type": "h200",
                },
                "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
                "inference": {"max_seq_len": 8192},
            }
        )
    )

    def no_client():
        raise AssertionError("dry-run must not build a client")

    monkeypatch.setattr(cli, "_load_cortex_training_client_class", no_client)
    stdout = io.StringIO()
    result = cli.main(
        ["model-deployment", "create", "--spec", str(spec_path), "--dry-run"],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert result == 0
    assert json.loads(stdout.getvalue())["deployment"]["gpu_type"] == "H200"


def test_create_dry_run_reports_spec_errors(tmp_path):
    spec_path = tmp_path / "deployment.json"
    spec_path.write_text(json.dumps({"deployment": {"class": "EVALUATION"}}))
    stderr = io.StringIO()

    result = cli.main(
        ["model-deployment", "create", "--spec", str(spec_path), "--dry-run"],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert result == 1
    assert "missing required field(s)" in stderr.getvalue()


def test_create_help_shows_an_example_spec(capsys):
    with pytest.raises(SystemExit):
        cli.main(["model-deployment", "create", "--help"])

    help_text = capsys.readouterr().out
    assert '"deployment": {"class": "EVALUATION", "min_replicas": 1,' in help_text
    assert '"base_model": "Qwen' not in help_text
    assert "--dry-run" in help_text


@pytest.mark.parametrize(
    "command", ["describe", "wait", "drop", "download-log", "download-metrics"]
)
def test_single_deployment_commands_name_the_positional_deployment(command, capsys):
    with pytest.raises(SystemExit):
        cli.main(["model-deployment", command, "--help"])

    help_text = capsys.readouterr().out
    assert "DEPLOYMENT" in help_text
    assert "DEPLOYMENT_ID" not in help_text
    assert "The deployment (currently an sf-ct-<UUID> ID)." in " ".join(help_text.split())


def test_poll_timeout_has_help_text(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    assert "Seconds to wait for a job or model deployment" in capsys.readouterr().out


def test_compact_is_accepted_after_the_model_deployment_subcommand():
    client = DeploymentClient()

    result, stdout, _stderr = run_cli(
        client, "model-deployment", "list", "--compact", "--database", "DB", "--schema", "SCH"
    )

    assert result == 0
    assert "\n" not in stdout.strip()


@pytest.mark.parametrize("command", ["wait", "create"])
def test_timeout_flag_sets_the_poll_timeout(command):
    extra = [DEPLOYMENT_ID] if command == "wait" else ["--spec", "spec.json"]
    args = cli.parse_args(["model-deployment", command, *extra, "--timeout", "90"])

    assert args.poll_timeout == 90


def test_list_status_filter_is_passed_to_the_client():
    client = DeploymentClient()

    result, _stdout, _stderr = run_cli(
        client, "model-deployment", "list", "--status", "ready", "--status", "PENDING",
        "--database", "DB", "--schema", "SCH",
    )

    assert result == 0
    assert client.list_status == ["ready", "PENDING"]


def test_list_returns_deployments_without_backing_job_or_sub_job_ids():
    client = DeploymentClient()

    result, stdout, stderr = run_cli(
        client,
        "model-deployment",
        "list",
        "--database",
        "DB",
        "--schema",
        "SCH",
    )

    assert result == 0
    assert stderr == ""
    output = json.loads(stdout)
    # list shows only the ID and status; describe has the rest.
    assert output == {"deployments": [{"deployment": DEPLOYMENT_ID, "status": "READY"}]}
    assert JOB_ID not in stdout.replace(DEPLOYMENT_ID, "")
    assert SUB_JOB_ID not in stdout
    assert "sub_jobs" not in stdout


def test_list_orders_by_creation_time_before_trimming_fields():
    client = DeploymentClient()
    newer = {**client.ready_metadata(), "deployment": "sf-ct-newer", "created_at": "2026-09-17T00:00:00Z"}
    older = {**client.ready_metadata(), "deployment": "sf-ct-older", "status": "PENDING",
             "created_at": "2026-09-16T00:00:00Z"}
    client.list_model_deployments = lambda status=None: [newer, older]

    result, stdout, _stderr = run_cli(
        client, "model-deployment", "list", "--database", "DB", "--schema", "SCH"
    )

    assert result == 0
    assert json.loads(stdout)["deployments"] == [
        {"deployment": "sf-ct-older", "status": "PENDING"},
        {"deployment": "sf-ct-newer", "status": "READY"},
    ]


def test_wait_prints_the_same_summary_as_create_wait(tmp_path):
    client = DeploymentClient()

    _, wait_stdout, _ = run_cli(
        client, "model-deployment", "wait", DEPLOYMENT_ID, "--database", "DB", "--schema", "SCH"
    )
    _, create_stdout, _ = run_cli(
        client, "model-deployment", "create", "--spec", str(write_spec(tmp_path)), "--wait",
        "--database", "DB", "--schema", "SCH",
    )

    expected = {
        "deployment": DEPLOYMENT_ID,
        "status": "READY",
        "base_url": client.model_deployment_base_url,
    }
    assert json.loads(wait_stdout) == expected
    assert json.loads(create_stdout) == expected


def test_wait_and_drop_pass_only_opaque_ids_to_client():
    client = DeploymentClient()

    wait_result, wait_stdout, _ = run_cli(
        client, "model-deployment", "wait", DEPLOYMENT_ID, "--database", "DB"
    )
    drop_result, drop_stdout, _ = run_cli(
        client, "model-deployment", "drop", DEPLOYMENT_ID, "--database", "DB"
    )

    assert wait_result == drop_result == 0
    assert client.waited_job_id == DEPLOYMENT_ID
    assert client.dropped_job_id == DEPLOYMENT_ID
    assert json.loads(wait_stdout)["deployment"] == DEPLOYMENT_ID
    assert json.loads(drop_stdout) == {
        "deployment": DEPLOYMENT_ID,
        "dropped": True,
    }


def test_drop_of_a_failed_deployment_prints_not_dropped_and_exits_zero():
    client = DeploymentClient()
    not_dropped = {"deployment": DEPLOYMENT_ID, "dropped": False, "status": "FAILED"}
    client.drop_model_deployment = lambda deployment_id: not_dropped

    result, stdout, _ = run_cli(
        client, "model-deployment", "drop", DEPLOYMENT_ID, "--database", "DB"
    )

    assert result == 0
    assert json.loads(stdout) == not_dropped


def test_download_commands_emit_flat_sanitized_metadata(tmp_path):
    client = DeploymentClient()
    log_dir = tmp_path / "logs"
    metrics_dir = tmp_path / "metrics"

    log_result, log_stdout, log_stderr = run_cli(
        client,
        "model-deployment",
        "download-log",
        DEPLOYMENT_ID,
        "--database",
        "DB",
        "--log-type",
        "execution",
        "--output-dir",
        str(log_dir),
    )
    metrics_result, metrics_stdout, metrics_stderr = run_cli(
        client,
        "model-deployment",
        "download-metrics",
        DEPLOYMENT_ID,
        "--database",
        "DB",
        "--output-dir",
        str(metrics_dir),
    )

    assert log_result == metrics_result == 0
    assert log_stderr == metrics_stderr == ""
    assert client.log_request == (DEPLOYMENT_ID, log_dir, "execution")
    assert client.metrics_request == (DEPLOYMENT_ID, metrics_dir)
    assert json.loads(log_stdout)["logs"][0]["saved_path"] == str(
        log_dir / "execution.jsonl"
    )
    assert json.loads(metrics_stdout)["metrics"][0]["saved_path"] == str(
        metrics_dir / "gpu.jsonl"
    )
    assert SUB_JOB_ID not in log_stdout + metrics_stdout


def test_create_help_calls_out_completed_weights_only_checkpoint(capsys):
    try:
        cli.build_parser().parse_args(["model-deployment", "create", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "completed Cortex Training weights-only checkpoint" in help_text


def test_create_summary_includes_failure_information():
    summary = cli._model_deployment_summary(
        {
            "deployment": DEPLOYMENT_ID,
            "status": "FAILED",
            "base_url": "https://x/inference/v1",
            "failure": {"code": "sub_job_failed", "message": "out of memory"},
            "model_source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
        }
    )
    assert summary == {
        "deployment": DEPLOYMENT_ID,
        "status": "FAILED",
        "base_url": "https://x/inference/v1",
        "failure": {"code": "sub_job_failed", "message": "out of memory"},
    }


def test_unknown_model_deployment_subcommand_is_rejected():
    with pytest.raises(ValueError, match="unknown model-deployment command: bogus"):
        cli._cmd_model_deployment(
            argparse.Namespace(model_deployment_command="bogus"),
            DeploymentClient(),
            io.StringIO(),
            io.StringIO(),
        )


def test_describe_prints_creation_spec_for_one_opaque_id():
    client = DeploymentClient()

    result, stdout, stderr = run_cli(
        client,
        "model-deployment",
        "describe",
        DEPLOYMENT_ID,
        "--database",
        "DB",
        "--schema",
        "SCH",
    )

    assert result == 0
    assert stderr == ""
    assert client.described_id == DEPLOYMENT_ID
    output = json.loads(stdout)
    assert output == client.describe_model_deployment(DEPLOYMENT_ID)

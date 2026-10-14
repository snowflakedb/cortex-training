from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cortex_training._cli as cli

JOB_ID = "12345678-1234-4234-8234-123456789abc"
DEPLOYMENT_ID = f"sf-ct-{JOB_ID}"
PREFIX = "/api/v2/databases/DB/schemas/SCH/cortex-training"


def marked_job() -> dict:
    return {
        "job_id": JOB_ID,
        "status": "JOB_STATE_RUNNING",
        "created_at": "2026-09-16T20:00:00Z",
        "owner_role": "OWNER_ROLE",
        "sub_jobs": [
            {
                "sub_job_id": f"{JOB_ID}:sampling:0",
                "job_type": "sampling",
                "model_name": "Qwen/Qwen3-1.7B",
                "inference_config": {
                    "deployment_class": "EVALUATION",
                    "num_replicas": 2,
                    "n_gpus": 4,
                    "max_seq_len": 8192,
                    "max_inflight_choices": 8,
                    "vllm_config": {
                        "tensor_parallel_size": 2,
                        "max_model_len": 8192,
                    },
                },
            }
        ],
        "hardware": "H200",
        "submitted_by": "CREATOR_USER",
    }


class ServerState:
    def __init__(self):
        self.create_body = None
        self.cancelled = False


def handler_type(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            return

        def send_json(self, status: int, body: dict):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == PREFIX:
                self.send_json(
                    200,
                    {
                        "jobs": [
                            {
                                "job_id": "ordinary-job",
                                "status": "running",
                                "sub_jobs": [],
                            },
                            marked_job(),
                        ]
                    },
                )
                return
            if self.path == f"{PREFIX}/{JOB_ID}":
                self.send_json(200, marked_job())
                return
            self.send_json(404, {"message": "not found"})

        def do_POST(self):
            if self.path == PREFIX:
                length = int(self.headers.get("Content-Length", "0"))
                state.create_body = json.loads(self.rfile.read(length))
                self.send_json(200, {"job_id": JOB_ID})
                return
            if self.path == f"{PREFIX}/{JOB_ID}:cancel":
                state.cancelled = True
                self.send_json(200, {})
                return
            self.send_json(404, {"message": "not found"})

    return Handler


def invoke_all(base_url: str, *arguments: str) -> list[dict]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = cli.main(
        ["--base-url", base_url, "--compact", *arguments],
        stdout=stdout,
        stderr=stderr,
    )
    assert result == 0, stderr.getvalue()
    assert stderr.getvalue() == ""
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def invoke(base_url: str, *arguments: str) -> dict:
    outputs = invoke_all(base_url, *arguments)
    assert len(outputs) == 1
    return outputs[0]


def test_model_deployment_lifecycle_over_real_http_mock_server(tmp_path):
    state = ServerState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    spec_path = Path(tmp_path) / "deployment.json"
    spec_path.write_text(
        json.dumps(
            {
                "deployment": {
                    "class": "EVALUATION",
                    "min_replicas": 2,
                    "gpus_per_replica": 2,
                    "gpu_type": "H200",
                },
                "source": {
                    "type": "base_model",
                    "model": "Qwen/Qwen3-1.7B",
                },
                "inference": {"max_seq_len": 8192},
            }
        )
    )

    try:
        ready = invoke(
            base_url,
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
        listed = invoke(
            base_url,
            "model-deployment",
            "list",
            "--database",
            "DB",
            "--schema",
            "SCH",
        )
        waited = invoke(
            base_url,
            "model-deployment",
            "wait",
            DEPLOYMENT_ID,
            "--database",
            "DB",
            "--schema",
            "SCH",
        )
        dropped = invoke(
            base_url,
            "model-deployment",
            "drop",
            DEPLOYMENT_ID,
            "--database",
            "DB",
            "--schema",
            "SCH",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert ready == {
        "deployment": DEPLOYMENT_ID,
        "status": "READY",
        "base_url": f"{base_url}{PREFIX}/inference/v1",
    }
    assert [item["deployment"] for item in listed["deployments"]] == [DEPLOYMENT_ID]
    assert waited["status"] == "READY"
    assert dropped == {"deployment": DEPLOYMENT_ID, "dropped": True}
    assert state.cancelled
    assert state.create_body == {
        "idle_timeout_seconds": 0,
        "hardware": "H200",
        "sub_job_configs": [
            {
                "job_type": "sampling",
                "model_name": "Qwen/Qwen3-1.7B",
                "inference_config": {
                    "max_seq_len": 8192,
                    "n_gpus": 4,
                    "deployment_class": "EVALUATION",
                    "num_replicas": 2,
                    "vllm_config": {
                        "tensor_parallel_size": 2,
                        "max_model_len": 8192,
                    },
                },
            }
        ],
    }
    combined = repr((ready, listed, waited, dropped))
    assert f"{JOB_ID}:sampling:0" not in combined
    assert "ordinary-job" not in combined
    assert "max_inflight_choices" not in combined

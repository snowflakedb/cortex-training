"""Tests for registered model versions: ``cortex-training model`` and registry deployments."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cortex_training.cli.commands as cli
from cortex_training import model_registry as registry
from cortex_training.client import CortexTrainingClient
from cortex_training.client import core as client_module
from cortex_training.model_deployment import ModelDeploymentSpec

JOB_ID = "12345678-1234-4234-8234-123456789abc"
OPERATION = "01c88339-0186-e087-0001-492a11f8505a"
DEPLOYMENT_JOB_ID = "87654321-4321-4321-8321-cba987654321"

FULL_SPEC_YAML = """---
version: "2.0"
model:
  type: "CORTEX_TRAINING"
  framework: "custom"
  details:
    kind: "FULL"
    base_model: "Qwen/Qwen3-0.6B"
    weights:
      path: "weights/"
---
"""
LORA_SPEC_YAML = FULL_SPEC_YAML.replace('kind: "FULL"', 'kind: "LORA"')


# ---------------------------------------------------------------- helpers


def test_split_model_and_version_honours_quotes():
    assert registry.split_model_and_version("DB.SCH.M/V1") == ("DB.SCH.M", "V1")
    assert registry.split_model_and_version("M") == ("M", None)
    assert registry.split_model_and_version('"a/b".M') == ('"a/b".M', None)
    assert registry.split_model_and_version('M/"v/2"') == ("M", '"v/2"')
    for bad in ("", "M/", "/V", '"open'):
        with pytest.raises(ValueError):
            registry.split_model_and_version(bad)


@pytest.mark.parametrize("name", ["M", "S.M", "DB.S.M", 'DB."My S".M', '"a""b"'])
def test_model_names_are_accepted(name):
    assert registry.require_model_name(name) == name


@pytest.mark.parametrize("name", ["A.B.C.D", "x;drop model y", "my-model", "", "DB..M"])
def test_model_names_that_are_not_identifiers_are_refused(name):
    with pytest.raises(ValueError, match="not a model name"):
        registry.require_model_name(name)


def test_parse_model_spec_reads_kind_and_base_model():
    assert registry.parse_model_spec(LORA_SPEC_YAML) == {"kind": "LORA", "base_model": "Qwen/Qwen3-0.6B"}
    assert registry.parse_model_spec("") == {"kind": None, "base_model": None}


@pytest.mark.parametrize(
    "status,state",
    [
        ("SUCCESS", "SUCCEEDED"),
        ("RUNNING", "RUNNING"),
        ("QUEUED", "RUNNING"),
        ("FAILED_WITH_ERROR", "FAILED"),
        ("ABORTED", "CANCELLED"),
    ],
)
def test_operation_state_maps_connector_statuses(status, state):
    assert registry.operation_state(status) == state


def test_default_model_name_drops_hyphens_and_upper_cases():
    assert registry.default_model_name(JOB_ID) == "JOB_12345678123442348234123456789ABC"


# ---------------------------------------------------------------- client


def client() -> CortexTrainingClient:
    return CortexTrainingClient(
        base_url="http://test.local",
        database="DB",
        schema="SCH",
        poll_interval=0.001,
        poll_timeout=0.05,
    )


def http_response(status_code: int, body) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.content = json.dumps(body).encode() if body is not None else b""
    response.raise_for_status.return_value = None
    return response


def sql_result(columns, rows) -> dict:
    return {"resultSetMetaData": {"rowType": [{"name": c} for c in columns]}, "data": rows}


def test_register_posts_the_names_and_returns_the_operation():
    instance = client()
    instance._session = MagicMock()
    instance._session.post.return_value = http_response(
        202, {"code": "392604", "job_id": OPERATION, "result_handler": OPERATION + "0"}
    )

    result = instance.register_checkpoint(JOB_ID, "cp-1", model="DB.SCH.M", version="V1", comment="c")

    assert result == {"operation": OPERATION, "status": "RUNNING"}
    url = instance._session.post.call_args.args[0]
    assert url.endswith(f"/{JOB_ID}/checkpoints/cp-1:register")
    assert instance._session.post.call_args.kwargs["json"] == {
        "model": "DB.SCH.M",
        "version": "V1",
        "comment": "c",
    }


def test_register_sends_an_empty_body_for_the_default_names():
    instance = client()
    instance._session = MagicMock()
    instance._session.post.return_value = http_response(202, {"job_id": OPERATION})

    instance.register_checkpoint(JOB_ID, "cp-1")

    assert instance._session.post.call_args.kwargs["json"] == {}


def test_register_200_means_already_registered():
    instance = client()
    instance._session = MagicMock()
    instance._session.post.return_value = http_response(
        200, {"status": "Version V1 of model DB.SCH.M is already registered from this checkpoint"}
    )

    result = instance.register_checkpoint(JOB_ID, "cp-1")

    assert result["status"] == "ALREADY_REGISTERED"
    assert "already registered" in result["message"]


class FakeConnection:
    def __init__(self, statuses, error=None):
        self.statuses = list(statuses)
        self.error = error
        self.closed = False

    def get_query_status(self, operation):
        name = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(name=name)

    def get_query_status_throw_if_error(self, operation):
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.closed = True


def test_wait_polls_until_the_registration_ends():
    instance = client()
    connection = FakeConnection(["RUNNING", "RUNNING", "SUCCESS"])
    instance._open_experiment_artifact_connection = lambda: connection

    assert instance.wait_for_registration(OPERATION) == {"operation": OPERATION, "status": "SUCCEEDED"}
    assert connection.closed


def test_wait_returns_a_failure_with_the_server_error():
    instance = client()
    connection = FakeConnection(["FAILED_WITH_ERROR"], error="517614: copied 5 of 6 checkpoint files")
    instance._open_experiment_artifact_connection = lambda: connection

    result = instance.wait_for_registration(OPERATION)

    assert result["status"] == "FAILED"
    assert "517614" in result["error"]


def test_wait_times_out_while_the_registration_keeps_running():
    instance = client()
    instance._open_experiment_artifact_connection = lambda: FakeConnection(["RUNNING"])

    with pytest.raises(TimeoutError, match="keeps running"):
        instance.wait_for_registration(OPERATION)


@pytest.mark.parametrize("operation", ["not-an-id", "../statements", ""])
def test_operation_ids_are_checked_before_any_request(operation):
    instance = client()
    instance._session = MagicMock()
    with pytest.raises(ValueError, match="not a registration operation ID"):
        instance.cancel_registration(operation)
    instance._session.post.assert_not_called()


def test_cancel_cancels_the_statement_then_reports_its_status():
    instance = client()
    instance._session = MagicMock()
    instance._session.post.return_value = http_response(200, {"code": "000604", "message": "SQL execution canceled"})
    instance._open_experiment_artifact_connection = lambda: FakeConnection(["ABORTED"], error="000604: canceled")

    result = instance.cancel_registration(OPERATION)

    assert instance._session.post.call_args.args[0] == f"http://test.local/api/v2/statements/{OPERATION}/cancel"
    assert result["status"] == "CANCELLED"


def sql_router(responses):
    """A session whose POST to /api/v2/statements answers by statement prefix."""
    statements = []

    def post(url, **kwargs):
        statement = kwargs["json"]["statement"]
        statements.append(statement)
        for prefix, body in responses.items():
            if statement.startswith(prefix):
                return http_response(200, body)
        raise AssertionError(f"unexpected statement {statement}")

    session = MagicMock()
    session.post.side_effect = post
    return session, statements


def versions_result(rows):
    return sql_result(["name", "created_on", "comment", "metadata"], rows)


def metadata(job=JOB_ID, checkpoint="cp-1", query=OPERATION):
    return json.dumps(
        {"source_job": job, "source_checkpoint": checkpoint, "registration_query_id": query}
    )


class SpecConnection:
    """A connector session whose GET writes a model.yaml chosen by version name."""

    def __init__(self, specs):
        self.specs = specs
        self.gets = []

    def cursor(self):
        outer = self

        class Cursor:
            def execute(self, statement):
                outer.gets.append(statement)
                version = statement.split("/versions/")[1].split("/")[0]
                folder = statement.split("'file://")[1].rstrip("/'")
                Path(folder, "model.yaml").write_text(outer.specs[version])

            def close(self):
                pass

        return Cursor()

    def close(self):
        pass


def test_list_shows_kind_base_model_and_source_and_filters():
    instance = client()
    instance._session, statements = sql_router(
        {
            "SHOW MODELS": sql_result(
                ["name", "model_type"], [["M", "CORTEX_TRAINING"], ["OTHER", "USER_MODEL"]]
            ),
            "SHOW VERSIONS IN MODEL M": versions_result(
                [
                    ["V1", "2026-11-19", None, metadata()],
                    ["V2", "2026-11-19", "lora", metadata(checkpoint="cp-2")],
                    ["V3", "2026-11-19", None, metadata(job="other-job")],
                ]
            ),
        }
    )
    connection = SpecConnection({"V1": FULL_SPEC_YAML, "V2": LORA_SPEC_YAML, "V3": FULL_SPEC_YAML})
    instance._open_experiment_artifact_connection = lambda: connection

    rows = instance.list_registered_versions(job_id=JOB_ID, kind="lora")

    assert rows == [
        {
            "model": "M",
            "version": "V2",
            "kind": "LORA",
            "base_model": "Qwen/Qwen3-0.6B",
            "source_job": JOB_ID,
            "source_checkpoint": "cp-2",
            "created_on": "2026-11-19",
            "comment": "lora",
        }
    ]
    # USER_MODEL models are not read; V3's job is filtered out before its spec is read.
    assert not any("OTHER" in s for s in statements)
    assert len(connection.gets) == 2


def test_list_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        client().list_registered_versions(kind="QLORA")


def deployments_result(rows):
    return sql_result(["name", "status", "model", "version"], rows)


def test_drop_is_held_back_while_a_named_deployment_serves_the_version():
    instance = client()
    instance._session, statements = sql_router(
        {"SHOW MODEL DEPLOYMENTS": deployments_result([["SUPPORT_BOT", "RUNNING", "DB.SCH.M", "V1"]])}
    )

    result = instance.drop_registered_model("M", "V1")

    assert result["dropped"] is False
    assert result["deployments"][0]["name"] == "SUPPORT_BOT"
    assert "drop those deployments first" in result["warning"]
    assert not any(s.startswith(("DROP", "ALTER")) for s in statements)


def test_drop_with_force_drops_and_still_warns():
    instance = client()
    instance._session, statements = sql_router(
        {
            "SHOW MODEL DEPLOYMENTS": deployments_result([["SUPPORT_BOT", "RUNNING", "M", "V1"]]),
            "ALTER MODEL": sql_result(["status"], [["ok"]]),
        }
    )

    result = instance.drop_registered_model("M", "V1", force=True)

    assert result["dropped"] is True
    assert statements[-1] == "ALTER MODEL M DROP VERSION V1"


def test_drop_ignores_ended_and_other_deployments():
    instance = client()
    instance._session, statements = sql_router(
        {
            "SHOW MODEL DEPLOYMENTS": deployments_result(
                [["OLD", "DROPPED", "M", "V1"], ["OTHER", "FAILED", "M", "V2"], ["X", "RUNNING", "N", "V1"]]
            ),
            "DROP MODEL": sql_result(["status"], [["ok"]]),
            "ALTER MODEL": sql_result(["status"], [["ok"]]),
        }
    )

    first = instance.drop_registered_model("M", "V1", force=False)
    assert first["deployments"] == [] and first["dropped"] is True
    result = instance.drop_registered_model("DB.SCH.M")
    assert result["dropped"] is False  # OTHER failed, but its record still names M
    assert statements[-1].startswith("SHOW MODEL DEPLOYMENTS")


def test_drop_refuses_a_name_that_is_not_an_identifier():
    instance = client()
    instance._session = MagicMock()
    with pytest.raises(ValueError, match="not a model name"):
        instance.drop_registered_model("M; DROP DATABASE DB")
    instance._session.post.assert_not_called()


def test_find_registered_version_matches_job_and_checkpoint():
    instance = client()
    instance._session, _ = sql_router(
        {
            "SHOW VERSIONS IN MODEL": versions_result(
                [["V1", "x", None, metadata(checkpoint="cp-0")], ["CP_1", "x", None, metadata()]]
            )
        }
    )
    assert instance.find_registered_version("JOB_X", JOB_ID, "cp-1") == "CP_1"
    assert instance.find_registered_version("JOB_X", JOB_ID, "cp-9") is None


# ---------------------------------------------------------------- registry deployments


def registry_spec(gpus: int = 1) -> dict:
    return {
        "deployment": {"class": "EVALUATION", "min_replicas": 2, "gpus_per_replica": gpus, "gpu_type": "h200"},
        "source": {"type": "registry_model", "model": "DB.SCH.M", "version": "V1"},
        "inference": {"max_seq_len": 4096},
    }


def test_registry_spec_round_trips_and_builds_the_deployments_body():
    spec = ModelDeploymentSpec.from_dict(registry_spec())
    assert spec.to_public_spec()["source"] == {"type": "registry_model", "model": "DB.SCH.M", "version": "V1"}
    assert spec._to_deployments_body() == {
        "deployment": {"class": "EVALUATION", "min_replicas": 2, "gpu_type": "H200"},
        "source": {"type": "registry_model", "registry_model": {"name": "DB.SCH.M", "version": "V1"}},
        "inference": {"max_seq_len": 4096},
    }


def test_registry_spec_needs_model_and_version_and_one_gpu_per_replica():
    missing = registry_spec()
    del missing["source"]["version"]
    with pytest.raises(ValueError, match="missing required field"):
        ModelDeploymentSpec.from_dict(missing)
    with pytest.raises(ValueError, match="must be 1 for a 'registry_model' source"):
        ModelDeploymentSpec.from_dict(registry_spec(gpus=2))


def test_create_from_a_registered_version_posts_to_the_deployments_route():
    instance = client()
    instance._session = MagicMock()
    instance._session.post.return_value = http_response(
        200, {"job_id": DEPLOYMENT_JOB_ID, "model_id": f"sf-ct-{DEPLOYMENT_JOB_ID}"}
    )
    instance.get_job = MagicMock()

    metadata_out = instance.create_model_deployment(registry_spec())

    assert instance._session.post.call_args.args[0].endswith("/cortex-training/deployments")
    sent = instance._session.post.call_args.kwargs["json"]["source"]["registry_model"]
    assert sent == {"name": "DB.SCH.M", "version": "V1"}
    assert metadata_out["deployment"] == f"sf-ct-{DEPLOYMENT_JOB_ID}"
    assert metadata_out["model_source"]["type"] == "registry_model"
    instance.get_job.assert_not_called()


def test_a_deployments_route_job_can_be_listed_and_described():
    """A deployment made through the deployments route stores no vllm_config."""
    job = {
        "job_id": DEPLOYMENT_JOB_ID,
        "status": "JOB_STATE_RUNNING",
        "hardware": "H200",
        "sub_jobs": [
            {
                "sub_job_id": f"{DEPLOYMENT_JOB_ID}:sampling:0",
                "job_type": "SAMPLING",
                "model_name": "Qwen/Qwen3-0.6B",
                "inference_config": {
                    "deployment_class": "EVALUATION",
                    "max_seq_len": 2048.0,
                    "n_gpus": 1.0,
                    "num_replicas": 1.0,
                },
                "source_checkpoint_info": {"model_registry": {"full": {"weights_path": "weights/"}}},
            }
        ],
    }
    out = client_module._model_deployment_metadata(job, f"sf-ct-{DEPLOYMENT_JOB_ID}")
    assert out["gpus_per_replica"] == 1
    assert out["model_source"] == {"type": "registry_model", "model": "Qwen/Qwen3-0.6B"}


# ---------------------------------------------------------------- CLI


class RegistryClient:
    model_deployment_base_url = "http://test.local/api/v2/databases/DB/schemas/SCH/cortex-training/inference/v1"

    def __init__(self, *, final="SUCCEEDED", deployments=None):
        self.final = final
        self.deployments = deployments or []
        self.calls = []

    def register_checkpoint(self, job_id, checkpoint_id, **names):
        self.calls.append(("register", job_id, checkpoint_id, names))
        return {"operation": OPERATION, "status": "RUNNING"}

    def wait_for_registration(self, operation):
        self.calls.append(("wait", operation))
        result = {"operation": operation, "status": self.final}
        if self.final != "SUCCEEDED":
            result["error"] = "517614: copied 5 of 6 checkpoint files"
        return result

    def cancel_registration(self, operation):
        self.calls.append(("cancel", operation))
        return {"operation": operation, "status": "CANCELLED"}

    def list_registered_versions(self, *, job_id=None, kind=None):
        self.calls.append(("list", job_id, kind))
        return [{"model": "M", "version": "V1", "kind": "FULL"}]

    def drop_registered_model(self, model, version=None, *, force=False):
        self.calls.append(("drop", model, version, force))
        dropped = force or not self.deployments
        return {"model": model, "version": version, "deployments": self.deployments, "dropped": dropped}

    def find_registered_version(self, model, job_id, checkpoint_id):
        self.calls.append(("find", model, job_id, checkpoint_id))
        return "CP_1"

    def create_model_deployment(self, spec, **registration):
        self.calls.append(("create", spec, registration))
        return {"deployment": "sf-ct-x", "status": "PENDING", "base_url": self.model_deployment_base_url}


def run_cli(fake, *arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        ["--base-url", "http://test.local", "--database", "DB", "--schema", "SCH", *arguments],
        client_factory=lambda _args: fake,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_register_wait_prints_success():
    fake = RegistryClient()
    code, out, _ = run_cli(
        fake, "model", "register", "--database", "DB", "--schema", "SCH",
        "--job", JOB_ID, "--checkpoint", "cp-1", "--model", "M", "--version", "V1", "--wait",
    )
    assert code == 0
    assert json.loads(out) == {"operation": OPERATION, "status": "SUCCEEDED"}
    assert fake.calls[0] == ("register", JOB_ID, "cp-1", {"model": "M", "version": "V1", "comment": None})


def test_cli_register_wait_exits_1_with_the_api_error():
    fake = RegistryClient(final="FAILED")
    code, out, _ = run_cli(fake, "model", "register", "--job", JOB_ID, "--checkpoint", "cp-1", "--wait")
    assert code == 1
    assert "517614" in json.loads(out)["error"]


def test_cli_wait_cancel_list():
    fake = RegistryClient()
    assert run_cli(fake, "model", "wait", OPERATION)[0] == 0
    code, out, _ = run_cli(fake, "model", "cancel", OPERATION)
    assert code == 0 and json.loads(out)["status"] == "CANCELLED"
    code, out, _ = run_cli(fake, "model", "list", "--job", JOB_ID, "--kind", "full")
    assert code == 0 and json.loads(out)["versions"][0]["kind"] == "FULL"
    assert ("list", JOB_ID, "FULL") in fake.calls


def test_cli_drop_is_held_back_without_force():
    fake = RegistryClient(deployments=[{"name": "SUPPORT_BOT", "status": "RUNNING"}])
    code, out, _ = run_cli(fake, "model", "drop", "DB.SCH.M/V1")
    assert code == 1 and json.loads(out)["dropped"] is False
    code, out, _ = run_cli(fake, "model", "drop", "DB.SCH.M/V1", "--force")
    assert code == 0 and json.loads(out)["dropped"] is True
    assert fake.calls[-1] == ("drop", "DB.SCH.M", "V1", True)


def checkpoint_spec_file(tmp_path: Path, deployment_class: str = "EVALUATION") -> Path:
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {
                "deployment": {"class": deployment_class, "min_replicas": 1, "gpus_per_replica": 1, "gpu_type": "H200"},
                "source": {"type": "checkpoint", "job_id": JOB_ID, "checkpoint_id": "cp-1"},
                "inference": {"max_seq_len": 2048},
            }
        )
    )
    return path


def test_cli_deployment_create_passes_register_and_names(tmp_path):
    fake = RegistryClient()
    code, _, _ = run_cli(
        fake, "model-deployment", "create", "--spec", str(checkpoint_spec_file(tmp_path)),
        "--register", "--model", "SUPPORT_BOT", "--version", "V3",
    )
    assert code == 0
    assert fake.calls[-1][2] == {"register": True, "model": "SUPPORT_BOT", "version": "V3"}


def test_cli_deployment_create_production_needs_no_flag(tmp_path):
    fake = RegistryClient()
    code, _, _ = run_cli(
        fake, "model-deployment", "create", "--spec", str(checkpoint_spec_file(tmp_path, "PRODUCTION"))
    )
    assert code == 0
    spec, registration = fake.calls[-1][1], fake.calls[-1][2]
    assert spec["deployment"]["class"] == "PRODUCTION" and registration == {}


def production_checkpoint_spec(**deployment) -> dict:
    return {
        "deployment": {"class": "PRODUCTION", "min_replicas": 2, "gpus_per_replica": 1, "gpu_type": "H200", **deployment},
        "source": {"type": "checkpoint", "job_id": JOB_ID, "checkpoint_id": "cp-1"},
        "inference": {"max_seq_len": 8192},
    }


def test_production_checkpoint_is_registered_then_deployed_from_the_version():
    instance = client()
    instance.register_checkpoint_for_deployment = MagicMock(return_value=("DB.SCH.JOB_X", "CP_1"))
    instance.create_job_from_body = MagicMock()
    instance._session = MagicMock()
    instance._session.post.return_value = http_response(200, {"job_id": DEPLOYMENT_JOB_ID})

    out = instance.create_model_deployment(production_checkpoint_spec(name="SUPPORT_BOT"), model="M", version="V3")

    instance.register_checkpoint_for_deployment.assert_called_once_with(JOB_ID, "cp-1", model="M", version="V3")
    instance.create_job_from_body.assert_not_called()
    body = instance._session.post.call_args.kwargs["json"]
    assert instance._session.post.call_args.args[0].endswith("/deployments")
    assert body["deployment"] == {"class": "PRODUCTION", "min_replicas": 2, "gpu_type": "H200", "name": "SUPPORT_BOT"}
    assert body["source"]["registry_model"] == {"name": "DB.SCH.JOB_X", "version": "CP_1"}
    assert out["deployment_class"] == "PRODUCTION"


def test_evaluation_checkpoint_still_deploys_straight_from_the_checkpoint():
    instance = client()
    instance.register_checkpoint_for_deployment = MagicMock()
    instance.get_job = MagicMock(
        return_value={"sub_jobs": [{"job_type": "JOB_TYPE_TRAINING", "model_name": "Qwen/Qwen3-1.7B"}]}
    )
    instance.create_job_from_body = MagicMock(return_value={"job_id": DEPLOYMENT_JOB_ID})
    spec = production_checkpoint_spec()
    spec["deployment"]["class"] = "EVALUATION"

    instance.create_model_deployment(spec)

    instance.register_checkpoint_for_deployment.assert_not_called()
    sub_job = instance.create_job_from_body.call_args.args[0]["sub_job_configs"][0]
    assert sub_job["source_checkpoint_info"] == {"source_job_id": JOB_ID, "checkpoint_id": "cp-1"}


def test_production_spec_rules():
    with pytest.raises(ValueError, match="needs a 'checkpoint' or 'registry_model' source"):
        ModelDeploymentSpec.from_dict({**production_checkpoint_spec(), "source": {"type": "base_model", "model": "Q"}})
    with pytest.raises(ValueError, match="must be 1"):
        ModelDeploymentSpec.from_dict(production_checkpoint_spec(gpus_per_replica=2))
    evaluation = production_checkpoint_spec(name="X")
    evaluation["deployment"]["class"] = "EVALUATION"
    with pytest.raises(ValueError, match="name is only for PRODUCTION"):
        ModelDeploymentSpec.from_dict(evaluation)
    assert ModelDeploymentSpec.from_dict(production_checkpoint_spec(name="SUPPORT_BOT")).to_public_spec()[
        "deployment"
    ]["name"] == "SUPPORT_BOT"


def test_register_for_deployment_waits_then_finds_the_version():
    instance = client()
    instance.register_checkpoint = MagicMock(return_value={"operation": OPERATION, "status": "RUNNING"})
    instance.wait_for_registration = MagicMock(return_value={"operation": OPERATION, "status": "SUCCEEDED"})
    instance.find_registered_version = MagicMock(return_value="CP_1")

    assert instance.register_checkpoint_for_deployment(JOB_ID, "cp-1") == (
        f"DB.SCH.{registry.default_model_name(JOB_ID)}",
        "CP_1",
    )


def test_register_for_deployment_stops_on_a_failed_registration():
    instance = client()
    instance.register_checkpoint = MagicMock(return_value={"operation": OPERATION, "status": "RUNNING"})
    instance.wait_for_registration = MagicMock(
        return_value={"operation": OPERATION, "status": "FAILED", "error": "517614: copied 5 of 6"}
    )
    with pytest.raises(RuntimeError, match="failed: 517614"):
        instance.register_checkpoint_for_deployment(JOB_ID, "cp-1")


def test_dropping_a_production_deployment_uses_the_deployments_route():
    instance = client()
    job = {
        "job_id": DEPLOYMENT_JOB_ID,
        "status": "JOB_STATE_RUNNING",
        "sub_jobs": [
            {
                "job_type": "sampling",
                "model_name": "Qwen/Qwen3-0.6B",
                "inference_config": {"deployment_class": "PRODUCTION", "num_replicas": 1.0, "n_gpus": 1.0, "max_seq_len": 2048.0},
            }
        ],
    }
    instance.get_job = MagicMock(return_value=job)
    instance.cancel_job = MagicMock()
    instance._session = MagicMock()
    instance._session.delete.return_value = http_response(200, {"message": "deleted"})

    assert instance.drop_model_deployment(f"sf-ct-{DEPLOYMENT_JOB_ID}")["dropped"] is True
    instance.cancel_job.assert_not_called()
    assert instance._session.delete.call_args.args[0].endswith(f"/deployments/sf-ct-{DEPLOYMENT_JOB_ID}")


def test_an_external_stage_deployment_can_be_listed():
    job = {
        "job_id": DEPLOYMENT_JOB_ID,
        "status": "JOB_STATE_CANCELLED",
        "sub_jobs": [
            {
                "job_type": "sampling",
                "model_name": "Qwen/Qwen3-1.7B",
                "inference_config": {
                    "deployment_class": "EVALUATION",
                    "num_replicas": 1.0,
                    "n_gpus": 1.0,
                    "max_seq_len": 4096.0,
                    "vllm_config": {"tensor_parallel_size": 1.0, "max_model_len": 4096.0},
                },
                "source_checkpoint_info": {"is_external": True},
            }
        ],
    }
    out = client_module._model_deployment_metadata(job, f"sf-ct-{DEPLOYMENT_JOB_ID}")
    assert out["model_source"] == {"type": "external", "model": "Qwen/Qwen3-1.7B"}


@pytest.mark.parametrize(
    "name,qualified",
    [("M", "DB.SCH.M"), ("S2.M", "DB.S2.M"), ("D2.S2.M", "D2.S2.M"), ('"my.m"', 'DB.SCH."my.m"')],
)
def test_deployment_model_names_are_sent_fully_qualified(name, qualified):
    assert registry.qualify_model_name(name, "DB", "SCH") == qualified


def test_a_cancelled_statement_is_reported_as_cancelled():
    """A cancelled statement ends as FAILED_WITH_ERROR with error 000604."""
    instance = client()
    instance._open_experiment_artifact_connection = lambda: FakeConnection(
        ["FAILED_WITH_ERROR"], error="000604: SQL execution canceled"
    )
    assert instance.wait_for_registration(OPERATION)["status"] == "CANCELLED"


def test_list_quotes_a_mixed_case_model_in_the_spec_path():
    instance = client()
    instance._session, _ = sql_router(
        {
            "SHOW MODELS": sql_result(["name", "model_type"], [["My_Model", "CORTEX_TRAINING"]]),
            'SHOW VERSIONS IN MODEL "My_Model"': versions_result([["v1", "x", None, metadata()]]),
        }
    )
    connection = SpecConnection({"v1": FULL_SPEC_YAML})
    instance._open_experiment_artifact_connection = lambda: connection

    rows = instance.list_registered_versions()

    assert rows[0]["kind"] == "FULL"
    assert 'DB.SCH."My_Model"/versions/v1/' in connection.gets[0]


def test_list_keeps_going_when_one_spec_cannot_be_read():
    instance = client()
    instance._session, _ = sql_router(
        {
            "SHOW MODELS": sql_result(["name", "model_type"], [["M", "CORTEX_TRAINING"]]),
            "SHOW VERSIONS IN MODEL M": versions_result([["V1", "x", None, metadata()], ["V2", "x", None, metadata()]]),
        }
    )
    connection = SpecConnection({"V2": FULL_SPEC_YAML})  # V1 has no spec: GET raises KeyError
    instance._open_experiment_artifact_connection = lambda: connection

    rows = instance.list_registered_versions()

    assert [(r["version"], r["kind"]) for r in rows] == [("V1", None), ("V2", "FULL")]


@pytest.mark.parametrize("status", ["JOB_STATE_FAILED", "JOB_STATE_TERMINATED", "JOB_STATE_CANCELLED"])
def test_dropping_an_ended_production_deployment_still_deletes_its_record(status):
    instance = client()
    instance.get_job = MagicMock(
        return_value={
            "job_id": DEPLOYMENT_JOB_ID,
            "status": status,
            "sub_jobs": [
                {
                    "job_type": "sampling",
                    "model_name": "Qwen/Qwen3-0.6B",
                    "inference_config": {"deployment_class": "PRODUCTION", "num_replicas": 1.0, "n_gpus": 1.0, "max_seq_len": 2048.0},
                }
            ],
        }
    )
    instance.cancel_job = MagicMock()
    instance._session = MagicMock()
    instance._session.delete.return_value = http_response(200, {"message": "Deployment has already been deleted"})

    assert instance.drop_model_deployment(f"sf-ct-{DEPLOYMENT_JOB_ID}")["dropped"] is True
    instance._session.delete.assert_called_once()
    instance.cancel_job.assert_not_called()

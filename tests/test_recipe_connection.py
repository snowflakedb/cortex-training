"""How recipes connect: the Snowflake profile, the deprecated config=, and the library check."""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest

from cortex_training import CortexTrainingClient
from recipes import utils


@pytest.fixture
def recorded(monkeypatch):
    """Record the client constructors make_client calls, instead of connecting."""
    calls = []
    monkeypatch.setattr(
        CortexTrainingClient, "from_configs", classmethod(lambda cls, *a, **k: calls.append(("profile", a, k)) or "client")
    )
    monkeypatch.setattr(
        CortexTrainingClient, "_from_pat", classmethod(lambda cls, *a, **k: calls.append(("pat", a, k)) or "client")
    )
    return calls


@pytest.fixture
def ran_as(monkeypatch):
    """Pretend the recipe was run as ``python -m MODULE ARGS``."""

    def set_command(module, *args):
        main = types.ModuleType("__main__")
        main.__spec__ = types.SimpleNamespace(name=module)
        monkeypatch.setitem(sys.modules, "__main__", main)
        monkeypatch.setattr(sys, "argv", [f"/repo/{module.replace('.', '/')}.py", *args])

    return set_command


def test_without_config_the_recipe_uses_the_profile(recorded, caplog):
    assert utils.make_client(None, connection="train", poll_timeout=60.0) == "client"
    assert recorded == [("profile", ("train",), {"poll_timeout": 60.0})]
    assert "deprecated" not in caplog.text


def test_without_config_or_connection_the_saved_profile_is_used(recorded):
    utils.make_client()
    assert recorded == [("profile", (None,), {})]


def test_config_still_works_with_a_warning_that_gives_the_new_command(recorded, ran_as, tmp_path, caplog):
    old = tmp_path / "config.json"
    old.write_text(json.dumps({"host": "acme.snowflakecomputing.com", "pat": "tok", "database": "MY_DB"}))
    ran_as("recipes.sft.conversational.train", f"config={old}", "job_config=configs/qwen3_8b_lora.json")
    with caplog.at_level(logging.WARNING, logger="recipes.utils"):
        utils.make_client(str(old))
    kind, _, kwargs = recorded[0]
    assert kind == "pat" and kwargs["host"] == "acme.snowflakecomputing.com" and kwargs["database"] == "MY_DB"
    text = caplog.text
    assert "config= is deprecated and stops working in cortex-training 0.2.0." in text
    assert f"cortex-training login --convert-old-config {old}" in text
    assert "python -m recipes.sft.conversational.train job_config=configs/qwen3_8b_lora.json" in text
    assert "add connection=NAME" in text
    assert "tok" not in text


def test_an_old_library_stops_the_recipe_before_anything_runs(monkeypatch, recorded):
    monkeypatch.delattr(CortexTrainingClient, "from_configs")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.0.3")
    with pytest.raises(SystemExit) as stopped:
        utils.make_client(None, connection="train")
    message = str(stopped.value)
    assert "these recipes need cortex-training 0.1.0 or newer, but cortex-training 0.0.3 is installed." in message
    assert "uv pip install -e ." in message
    assert recorded == []


@pytest.mark.parametrize(
    ("config_path", "connection", "expected"),
    [("/c.json", None, " config=/c.json"), (None, "train", " connection=train"), (None, None, "")],
)
def test_follow_up_commands_repeat_how_the_recipe_connected(config_path, connection, expected):
    line = utils._sampling_cli(module="recipes.inference.generate", config_path=config_path, connection=connection, job_id="J")
    assert line == f"  python -m recipes.inference.generate{expected} source_job_id=J"


def test_the_endpoint_examples_use_the_profile(caplog):
    from recipes.inference.endpoint import log_endpoint_ready

    with caplog.at_level(logging.INFO):
        log_endpoint_ready(None, "J", connection="train")
    assert "python -m recipes.inference.generate connection=train job_id=J" in caplog.text
    assert "config=" not in caplog.text


def test_the_dare_eval_command_uses_the_profile():
    from recipes.sft.tasks.dare import DareTask

    command = DareTask.eval_command(object.__new__(DareTask), config_path=None, job_id="J", checkpoint_id="C")
    assert command.startswith("python -m recipes.sft.dare.evaluate bench_dir=")
    assert "config=" not in command


@pytest.mark.parametrize("module", ["recipes.inference.serve", "recipes.inference.generate", "recipes.rl.math_grpo.train"])
def test_a_recipe_runs_without_config(module):
    import importlib

    Config = importlib.import_module(module).Config
    config = Config()
    assert (config.config, config.connection) == (None, None)
    assert Config(connection="train").connection == "train"

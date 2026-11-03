"""`python -m cortex_training.env`: export a profile's connection for tools that read ARCTIC_CORTEX_*."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cortex_training import env

PROFILES = """\
[train]
account = "my_acct"
host = "my_acct.us-west-2.snowflakecomputing.com"
user = "ME"
authenticator = "PROGRAMMATIC_ACCESS_TOKEN"
token = "tok-123"
database = "TRAIN_DB"
schema = "RUNS"

[other]
account = "OTHER"
user = "ME"
authenticator = "PROGRAMMATIC_ACCESS_TOKEN"
token = "it's-quoted"

[keyfile]
account = "ACME"
user = "ME"
authenticator = "SNOWFLAKE_JWT"
private_key_file = "/keys/me.p8"
"""


@pytest.fixture
def home(monkeypatch, tmp_path):
    for name in ("CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT",
                 "CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_HOST", "SNOWFLAKE_PAT",
                 "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA", "CORTEX_TRAINING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    snowflake = Path(os.environ["SNOWFLAKE_HOME"])
    (snowflake / "connections.toml").write_text(PROFILES)
    (snowflake / "cortex-training").mkdir(exist_ok=True)
    (snowflake / "cortex-training" / "config.json").write_text(json.dumps({"profile": "train"}))
    return snowflake


def run(argv):
    """Call main with a fake runner; return (code, the command and env it would run, stderr)."""
    seen = {}

    def fake_run(command, env):
        seen.update(command=list(command), env=dict(env))
        return 7

    stderr = io.StringIO()
    code = env.main(list(argv), stderr=stderr, run=fake_run)
    return code, seen, stderr.getvalue()


def test_the_command_runs_with_the_saved_profile_and_the_hyphen_host(home):
    code, seen, err = run(["--", "bash", "launch.sh", "--steps", "10"])
    assert code == 7  # the command's own exit code
    assert seen["command"] == ["bash", "launch.sh", "--steps", "10"]
    assert {k: v for k, v in seen["env"].items() if k.startswith("ARCTIC_CORTEX_")} == {
        "ARCTIC_CORTEX_HOST": "my-acct.us-west-2.snowflakecomputing.com",
        "ARCTIC_CORTEX_DATABASE": "TRAIN_DB",
        "ARCTIC_CORTEX_SCHEMA": "RUNS",
        "ARCTIC_CORTEX_PAT": "tok-123",
    }
    assert seen["env"]["SNOWFLAKE_HOME"] == os.environ["SNOWFLAKE_HOME"]  # the rest of the environment is kept
    assert "Running bash with ARCTIC_CORTEX_* from connection profile 'train' (saved by cortex-training login)." in err
    assert "tok-123" not in err


def test_another_profile_by_name(home):
    code, seen, _ = run(["connection=other", "--", "true"])
    assert code == 7
    assert seen["env"]["ARCTIC_CORTEX_HOST"] == "OTHER.snowflakecomputing.com"
    assert seen["env"]["ARCTIC_CORTEX_DATABASE"] == "CORTEX_TRAINING_DB"  # the default, like every command
    assert seen["env"]["ARCTIC_CORTEX_PAT"] == "it's-quoted"


@pytest.mark.parametrize("argv", [[], ["connection=train"], ["--"], ["connection=train", "--"]])
def test_without_a_command_it_shows_how_to_use_it(home, argv):
    code, seen, err = run(argv)
    assert code == 2 and seen == {}
    assert "usage: python -m cortex_training.env [connection=NAME] -- COMMAND" in err


def test_a_profile_without_a_pat_says_how_to_get_one(home):
    code, seen, err = run(["connection=keyfile", "--", "true"])
    assert code == 1 and seen == {}
    assert "connection profile 'keyfile' doesn't sign in with a programmatic access token (PAT)" in err
    assert "`cortex-training login` creates one" in err


def test_ci_variables_are_used_when_no_profile_exists(home, monkeypatch):
    (home / "connections.toml").write_text("")
    (home / "cortex-training" / "config.json").unlink()
    monkeypatch.setenv("CORTEX_TRAINING_HOST", "ci_acct.snowflakecomputing.com")
    monkeypatch.setenv("CORTEX_TRAINING_PAT", "ci-tok")
    code, seen, err = run(["--", "true"])
    assert code == 7, err
    assert seen["env"]["ARCTIC_CORTEX_HOST"] == "ci-acct.snowflakecomputing.com"
    assert seen["env"]["ARCTIC_CORTEX_PAT"] == "ci-tok"


def test_an_old_json_config_still_works_with_its_warning(home, monkeypatch, tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"host": "acme.snowflakecomputing.com", "pat": "old-tok", "database": "DB"}))
    monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(old))
    code, seen, err = run(["--", "true"])
    assert code == 7, err
    assert seen["env"]["ARCTIC_CORTEX_PAT"] == "old-tok" and seen["env"]["ARCTIC_CORTEX_SCHEMA"] == "PUBLIC"
    assert "warning: Signing in with an old JSON config" in err


def test_unknown_options_are_refused(home):
    code, seen, err = run(["--connection", "train", "--", "true"])
    assert code == 2 and seen == {} and "unknown argument '--connection'" in err


def test_a_missing_command_says_so(home):
    stderr = io.StringIO()
    code = env.main(["--", "no-such-command-xyz"], stderr=stderr, run=lambda c, e: subprocess.call(c, env=e))
    assert code == 127 and "command not found: no-such-command-xyz" in stderr.getvalue()


def test_the_child_gets_the_pat_and_the_wrapper_prints_none_of_it(home):
    # End to end, as the READMEs use it: only the child's own output has the token.
    child = f"{sys.executable} -c 'import os; print(os.environ[\"ARCTIC_CORTEX_PAT\"])'"
    result = subprocess.run(
        [sys.executable, "-m", "cortex_training.env", "connection=other", "--", "sh", "-c", child],
        capture_output=True, text=True, env=os.environ.copy(), check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "it's-quoted\n"
    assert "it's-quoted" not in result.stderr


def test_the_exit_code_is_the_commands(home):
    result = subprocess.run(
        [sys.executable, "-m", "cortex_training.env", "--", "sh", "-c", "exit 3"],
        capture_output=True, text=True, env=os.environ.copy(), check=False,
    )
    assert result.returncode == 3

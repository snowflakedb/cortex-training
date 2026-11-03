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


class Pipe(io.StringIO):
    def isatty(self):
        return False


class Terminal(io.StringIO):
    def isatty(self):
        return True


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


def run(argv=(), stdout=None):
    stdout = stdout if stdout is not None else Pipe()
    stderr = io.StringIO()
    code = env.main(list(argv), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_the_saved_profile_is_exported_with_the_hyphen_host(home):
    code, out, err = run()
    assert code == 0, err
    assert out == (
        "export ARCTIC_CORTEX_HOST=my-acct.us-west-2.snowflakecomputing.com\n"
        "export ARCTIC_CORTEX_DATABASE=TRAIN_DB\n"
        "export ARCTIC_CORTEX_SCHEMA=RUNS\n"
        "export ARCTIC_CORTEX_PAT=tok-123\n"
    )
    assert "connection profile 'train' (saved by cortex-training login)" in err
    assert "tok-123" not in err


def test_another_profile_by_name_and_quoting(home):
    code, out, _ = run(["connection=other"])
    assert code == 0
    assert "export ARCTIC_CORTEX_HOST=OTHER.snowflakecomputing.com\n" in out
    assert "export ARCTIC_CORTEX_DATABASE=CORTEX_TRAINING_DB\n" in out  # the default, like every command
    assert "export ARCTIC_CORTEX_PAT='it'\"'\"'s-quoted'\n" in out


def test_it_never_prints_the_pat_to_a_terminal(home):
    code, out, err = run(stdout=Terminal())
    assert code == 2 and out == ""
    assert "Prints your PAT, so it is not shown here." in err
    assert 'eval "$(python -m cortex_training.env)"' in err and "| source" in err
    assert "tok-123" not in err


def test_a_profile_without_a_pat_says_how_to_get_one(home):
    code, out, err = run(["connection=keyfile"])
    assert code == 1 and out == ""
    assert "connection profile 'keyfile' doesn't sign in with a programmatic access token (PAT)" in err
    assert "`cortex-training login` creates one" in err


def test_ci_variables_are_exported_when_no_profile_exists(home, monkeypatch):
    (home / "connections.toml").write_text("")
    (home / "cortex-training" / "config.json").unlink()
    monkeypatch.setenv("CORTEX_TRAINING_HOST", "ci_acct.snowflakecomputing.com")
    monkeypatch.setenv("CORTEX_TRAINING_PAT", "ci-tok")
    code, out, err = run()
    assert code == 0, err
    assert "export ARCTIC_CORTEX_HOST=ci-acct.snowflakecomputing.com\n" in out
    assert "export ARCTIC_CORTEX_PAT=ci-tok\n" in out


def test_an_old_json_config_still_works_with_its_warning(home, monkeypatch, tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"host": "acme.snowflakecomputing.com", "pat": "old-tok", "database": "DB"}))
    monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(old))
    code, out, err = run()
    assert code == 0, err
    assert "export ARCTIC_CORTEX_PAT=old-tok\n" in out and "export ARCTIC_CORTEX_SCHEMA=PUBLIC\n" in out
    assert "warning: Signing in with an old JSON config" in err


def test_unknown_arguments_are_refused(home):
    code, out, err = run(["--connection", "train"])
    assert code == 2 and out == "" and "unknown argument '--connection'" in err


def test_a_shell_can_eval_it(home):
    # The real use: bash evaluates the lines, and the tool it starts sees the values.
    script = f'eval "$({sys.executable} -m cortex_training.env connection=other)" && printf %s "$ARCTIC_CORTEX_PAT"'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=os.environ.copy(), check=True)
    assert result.stdout == "it's-quoted"

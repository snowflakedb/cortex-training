# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline tests for the interactive ``cortex-training login``."""

from __future__ import annotations

import io
import json
import logging
import stat
from pathlib import Path

import pytest
import tomlkit

import cortex_training._cli as cli
from cortex_training import _login
from cortex_training._connection import resolve_connection

HOST = "ORG-ACCOUNT.snowflakecomputing.com"
PAT = "test-pat-value-never-printed"


class TTYInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class FakeLiveClient:
    def __init__(self, owner: "FakeClientClass") -> None:
        self._owner = owner
        self.closed = False

    def get_capacity(self, hardware=None):
        if self._owner.on_capacity is not None:
            self._owner.on_capacity()
        if self._owner.capacity_error is not None:
            raise self._owner.capacity_error
        return {"gpus": 8}

    def close(self) -> None:
        self.closed = True


class FakeClientClass:
    def __init__(self, capacity_error=None, on_capacity=None) -> None:
        self.capacity_error = capacity_error
        self.on_capacity = on_capacity
        self.profile_calls: list[dict] = []
        self.pat_calls: list[dict] = []

    def from_connection_name(self, **kwargs):
        self.profile_calls.append(kwargs)
        return FakeLiveClient(self)

    def from_pat(self, **kwargs):
        self.pat_calls.append(kwargs)
        return FakeLiveClient(self)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in _login._ENV_VARS + ("CORTEX_TRAINING_ENDPOINT",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path / "snowflake"))
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "login.json"))


@pytest.fixture()
def connections_file(tmp_path) -> Path:
    return tmp_path / "snowflake" / "connections.toml"


@pytest.fixture()
def fake_client(monkeypatch) -> FakeClientClass:
    fake = FakeClientClass()
    monkeypatch.setattr(cli, "_load_cortex_training_client_class", lambda: fake)
    return fake


def _run(argv, stdin) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = cli.main(["login", *argv], stdin=stdin, stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


def _write_existing_profile(path: Path, name: str = "existing", **values) -> None:
    profile = {
        "account": "ORG-ACCOUNT",
        "host": HOST,
        "user": "EXISTING_USER",
        "authenticator": "programmatic_access_token",
        "token": "existing-token-never-printed",
        "database": "CORTEX_TRAINING_DB",
        "schema": "PUBLIC",
        **values,
    }
    _login.write_profile(
        path,
        name,
        host=profile["host"],
        user=profile["user"],
        pat=profile["token"],
        database=profile["database"],
        schema=profile["schema"],
    )


# ---------------------------------------------------------------------------
# Validation and file writing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        HOST,
        f"https://{HOST}/",
        f"https://{HOST}/console/login",
        f"  {HOST}  ",
        "https://app.snowflake.com/ORG/ACCOUNT/#/homepage",
    ],
)
def test_validate_host_accepts_and_normalizes(raw):
    assert _login.validate_host(raw) == HOST


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("ORG_ACCOUNT.snowflakecomputing.com", "Use 'ORG-ACCOUNT.snowflakecomputing.com'"),
        ("https://app.snowflake.com/", "no ORG/ACCOUNT part"),
        ("app.snowflake.com/org_name/acct", "Use 'org-name-acct.snowflakecomputing.com'"),
        ("ORG-ACCOUNT", "not an account host"),
        ("", "not an account host"),
    ],
)
def test_validate_host_rejects_with_fix(raw, message):
    with pytest.raises(ValueError, match=message):
        _login.validate_host(raw)


def test_write_profile_keeps_other_profiles_and_comments(connections_file):
    connections_file.parent.mkdir(parents=True)
    connections_file.write_text(
        '# my notes\n[other]\naccount = "X"\nuser = "U"  # keep me\n', encoding="utf-8"
    )

    _login.write_profile(
        connections_file, "cortex-training", host=HOST, user="ME", pat=PAT,
        database="CORTEX_TRAINING_DB", schema="PUBLIC",
    )

    text = connections_file.read_text(encoding="utf-8")
    assert "# my notes" in text and "# keep me" in text
    parsed = tomlkit.parse(text)
    assert parsed["other"]["account"] == "X"
    assert dict(parsed["cortex-training"]) == {
        "account": "ORG-ACCOUNT",
        "host": HOST,
        "user": "ME",
        "authenticator": "programmatic_access_token",
        "token": PAT,
        "database": "CORTEX_TRAINING_DB",
        "schema": "PUBLIC",
    }
    assert stat.S_IMODE(connections_file.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------


def test_interactive_login_creates_profile_and_verifies(connections_file, fake_client):
    # host, user, PAT, then Enter for database, schema, and profile name.
    stdin = TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\n")

    rc, stdout, stderr = _run([], stdin)

    assert rc == 0, stderr
    profile = tomlkit.parse(connections_file.read_text())["cortex-training"]
    assert profile["database"] == "CORTEX_TRAINING_DB"
    assert profile["schema"] == "PUBLIC"
    assert fake_client.profile_calls[0]["connection_name"] == "cortex-training"
    summary = json.loads(stdout)
    assert summary["verified"] is True
    assert summary["connection"] == "cortex-training"
    assert PAT not in stdout and PAT not in stderr
    assert "app.snowflake.com/ORG/ACCOUNT" in stderr
    assert "Settings > Authentication > Generate token" in stderr


def test_interactive_login_remembers_profile_for_later_commands(connections_file, fake_client):
    _run([], TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\n"))

    settings = resolve_connection()

    assert settings.use_connection_profile is True
    assert settings.connection_name == "cortex-training"
    assert "remembered by cortex-training login" in settings.source


def test_interactive_login_reprompts_on_bad_host(connections_file, fake_client):
    stdin = TTYInput(f"ORG_ACCOUNT.snowflakecomputing.com\n{HOST}\nME\n{PAT}\n\n\n\n")

    rc, _, stderr = _run([], stdin)

    assert rc == 0
    assert "Use 'ORG-ACCOUNT.snowflakecomputing.com' instead" in stderr


def test_interactive_login_prefills_from_env(connections_file, fake_client, monkeypatch):
    # A host alone is not a usable setup, so login prompts with it pre-filled.
    monkeypatch.setenv("CORTEX_TRAINING_HOST", HOST)
    stdin = TTYInput(f"\nME\n{PAT}\n\n\n\n")

    rc, stdout, stderr = _run([], stdin)

    assert rc == 0, stderr
    assert f"Account host [{HOST}]" in stderr
    assert json.loads(stdout)["host"] == HOST


def test_interactive_login_offers_pat_from_env(connections_file, fake_client, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_PAT", PAT)
    stdin = TTYInput(f"{HOST}\nME\n\n\n\n\n")  # Enter accepts the env PAT

    rc, stdout, stderr = _run([], stdin)

    assert rc == 0, stderr
    assert "Use the PAT from CORTEX_TRAINING_PAT?" in stderr
    assert tomlkit.parse(connections_file.read_text())["cortex-training"]["token"] == PAT
    assert PAT not in stdout and PAT not in stderr


def test_existing_profile_is_not_replaced_without_confirmation(connections_file, fake_client):
    _write_existing_profile(connections_file, "cortex-training")
    before = connections_file.read_text()
    stdin = TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\nn\n")

    rc, _, stderr = _run(["--connection", "cortex-training"], stdin)

    assert rc == 1
    assert "already exists" in stderr
    assert connections_file.read_text() == before


def test_force_replaces_existing_profile(connections_file, fake_client, monkeypatch):
    _write_existing_profile(connections_file, "cortex-training")
    monkeypatch.setenv("CORTEX_TRAINING_PAT", PAT)

    rc, _, stderr = _run(
        ["--host", HOST, "--user", "NEW_USER", "--connection", "cortex-training", "--force"],
        io.StringIO(),
    )

    assert rc == 0, stderr
    assert tomlkit.parse(connections_file.read_text())["cortex-training"]["user"] == "NEW_USER"


# ---------------------------------------------------------------------------
# Existing setup
# ---------------------------------------------------------------------------


def test_existing_setup_is_reported_and_verified(connections_file, fake_client):
    _write_existing_profile(connections_file, "default")
    before = connections_file.read_text()
    stdin = TTYInput("\n\n")  # check it: yes; create another anyway: no

    rc, stdout, stderr = _run([], stdin)

    assert rc == 0, stderr
    assert "source:    default connections.toml profile" in stderr
    assert f"{connections_file} [default]" in stderr
    assert "user:      EXISTING_USER" in stderr
    assert "token:     stored in connections.toml" in stderr
    assert "You're all set; nothing was changed." in stderr
    assert "existing-token-never-printed" not in stderr + stdout
    assert connections_file.read_text() == before
    assert json.loads(stdout) == {
        "logged_in": True,
        "profile_created": False,
        "source": "default connections.toml profile",
        "verified": True,
    }


def test_existing_setup_lists_unused_sources(connections_file, fake_client, monkeypatch):
    _write_existing_profile(connections_file, "default")
    _write_existing_profile(connections_file, "spare")
    monkeypatch.setenv("CORTEX_TRAINING_HOST", HOST)

    rc, _, stderr = _run([], TTYInput("\n\n"))

    assert rc == 0
    assert "Also found, but not used:" in stderr
    assert "connections.toml profile 'spare'" in stderr


def test_existing_env_setup_shows_where_the_token_comes_from(fake_client, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_HOST", HOST)
    monkeypatch.setenv("CORTEX_TRAINING_PAT", PAT)
    monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_DB")

    rc, stdout, stderr = _run([], TTYInput("\n\n"))

    assert rc == 0
    assert "source:    environment variables" in stderr
    assert "token:     from CORTEX_TRAINING_PAT" in stderr
    assert PAT not in stderr + stdout
    assert fake_client.pat_calls[0]["host"] == HOST


def test_failed_existing_setup_offers_new_profile(connections_file, fake_client):
    _write_existing_profile(connections_file, "default")
    fake_client.capacity_error = RuntimeError("rejected")
    # check: yes -> fails; set up a new profile: yes; then the prompts.
    stdin = TTYInput(f"\n\n{HOST}\nME\n{PAT}\n\n\nnew-profile\n")

    rc, stdout, stderr = _run([], stdin)

    assert "The connection check failed:" in stderr
    assert "Set up a new connections.toml profile instead?" in stderr
    assert "new-profile" in tomlkit.parse(connections_file.read_text())
    assert rc == 1  # the new profile's check fails with the same fake error
    assert json.loads(stdout)["verified"] is False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_failed_check_keeps_profile_and_explains(connections_file, fake_client):
    fake_client.capacity_error = RuntimeError("Programmatic access token is invalid")

    rc, stdout, stderr = _run([], TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\n"))

    assert rc == 1
    assert "cortex-training" in tomlkit.parse(connections_file.read_text())
    assert "Programmatic access token is invalid" in stderr
    assert "The profile was kept." in stderr
    assert json.loads(stdout)["verified"] is False


def test_database_creation_is_explained(connections_file, fake_client):
    client_logger = logging.getLogger("cortex_training.client")

    def create_database():
        client_logger.info("Database '%s' not found. Attempting to create it...", "CORTEX_TRAINING_DB")
        client_logger.info("Database '%s' created successfully.", "CORTEX_TRAINING_DB")

    fake_client.on_capacity = create_database

    rc, _, stderr = _run([], TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\n"))

    assert rc == 0, stderr
    assert "Database CORTEX_TRAINING_DB does not exist yet." in stderr
    assert "CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB" in stderr
    assert "Database 'CORTEX_TRAINING_DB' created successfully." in stderr
    assert client_logger.handlers == []


def test_warns_when_env_overrides_new_profile(connections_file, fake_client, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_PAT", PAT)
    monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "someone-else")

    rc, _, stderr = _run(["--host", HOST, "--user", "ME"], io.StringIO())

    assert rc == 0, stderr
    assert "commands without --connection will still use connection profile 'someone-else'" in stderr


# ---------------------------------------------------------------------------
# Scripted mode
# ---------------------------------------------------------------------------


def test_scripted_login_with_env_pat(connections_file, fake_client, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_PAT", PAT)

    rc, stdout, stderr = _run(
        ["--host", f"https://{HOST}/", "--user", "ME", "--database", "DB", "--schema", "S"],
        io.StringIO(),
    )

    assert rc == 0, stderr
    profile = tomlkit.parse(connections_file.read_text())["cortex-training"]
    assert (profile["host"], profile["database"], profile["schema"]) == (HOST, "DB", "S")
    assert PAT not in stdout + stderr


def test_scripted_login_with_pat_stdin(connections_file, fake_client):
    rc, _, stderr = _run(
        ["--host", HOST, "--user", "ME", "--pat-stdin"], io.StringIO(f"{PAT}\n")
    )

    assert rc == 0, stderr
    assert tomlkit.parse(connections_file.read_text())["cortex-training"]["token"] == PAT


def test_scripted_login_requires_missing_values(connections_file, fake_client, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_PAT", PAT)

    rc, _, stderr = _run(["--host", HOST], io.StringIO())

    assert rc == 1
    assert "Snowflake user is required: pass --user" in stderr
    assert not connections_file.exists()


def test_scripted_login_requires_pat(connections_file, fake_client):
    rc, _, stderr = _run(["--host", HOST, "--user", "ME"], io.StringIO())

    assert rc == 1
    assert "set CORTEX_TRAINING_PAT or pass --pat-stdin" in stderr


def test_non_interactive_login_without_setup_explains(fake_client):
    rc, _, stderr = _run([], io.StringIO())

    assert rc == 1
    assert "no Cortex Training connection is configured" in stderr


def test_legacy_login_still_remembers_config(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"base_url": "http://localhost:8084", "database": "DB"}))

    rc, stdout, _ = _run([str(config)], io.StringIO())

    assert rc == 0
    assert json.loads(stdout)["config_path"] == str(config.resolve())
    assert json.loads((tmp_path / "login.json").read_text()) == {
        "config_path": str(config.resolve())
    }


# ---------------------------------------------------------------------------
# Saved login validation
# ---------------------------------------------------------------------------


def _remember(tmp_path: Path, state: dict | str) -> Path:
    path = tmp_path / "login.json"
    path.write_text(state if isinstance(state, str) else json.dumps(state), encoding="utf-8")
    return path


def test_prompts_link_to_the_setup_guide(connections_file, fake_client):
    _, _, stderr = _run([], TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\n"))

    assert "authentication.md#step-1-create-a-pat" in stderr
    assert "screenshots" in stderr
    assert "authentication.md#finding-your-account-and-host" in stderr


def test_working_saved_login_is_verified_without_asking(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    _remember(tmp_path, {"connection": "saved"})

    rc, stdout, stderr = _run([], TTYInput("\n"))  # only: create another anyway? no

    assert rc == 0, stderr
    assert "Check this setup now?" not in stderr
    assert "Checking your saved login with a capacity request..." in stderr
    assert "You're all set" in stderr
    assert fake_client.profile_calls[0]["connection_name"] == "saved"
    assert (tmp_path / "login.json").exists()


def test_failing_saved_login_is_removed_and_reset(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    login_file = _remember(tmp_path, {"connection": "saved"})
    calls = {"n": 0}

    def fail_first_check():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Programmatic access token is invalid")

    fake_client.on_capacity = fail_first_check
    stdin = TTYInput(f"{HOST}\nME\n{PAT}\n\n\nfresh\n")

    rc, stdout, stderr = _run([], stdin)

    assert "Your saved cortex-training login no longer works: the connection check failed." in stderr
    assert "Programmatic access token is invalid" in stderr
    assert f"Removed the saved login ({login_file})." in stderr
    assert "Let's set up a new one." in stderr
    assert rc == 0, stderr
    assert json.loads(login_file.read_text()) == {"connection": "fresh"}


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        ({"connection": "gone"}, "profile 'gone' is no longer in"),
        ("{not json", "invalid cortex-training login state"),
        ({"config_path": "/nonexistent/config.json"}, "config file /nonexistent/config.json no longer exists"),
    ],
    ids=["missing-profile", "corrupt-file", "missing-config"],
)
def test_stale_saved_login_is_removed_before_prompting(
    tmp_path, connections_file, fake_client, state, reason
):
    login_file = _remember(tmp_path, state)

    rc, _, stderr = _run([], TTYInput(f"{HOST}\nME\n{PAT}\n\n\n\n"))

    assert rc == 0, stderr
    assert reason in stderr
    assert "Let's set up a new one." in stderr
    assert json.loads(login_file.read_text()) == {"connection": "cortex-training"}


def test_stale_saved_login_does_not_hide_a_working_setup(tmp_path, connections_file, fake_client, monkeypatch):
    _write_existing_profile(connections_file, "team")
    monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "team")
    login_file = _remember(tmp_path, {"connection": "gone"})

    rc, _, stderr = _run([], TTYInput("\n\n"))  # check it: yes; create another: no

    assert rc == 0, stderr
    assert "profile 'gone' is no longer in" in stderr
    assert not login_file.exists()
    assert "connection profile 'team' from CORTEX_TRAINING_CONNECTION" in stderr
    assert "Let's set up a new one." not in stderr


def test_stale_saved_login_without_terminal_explains(tmp_path, connections_file, fake_client):
    login_file = _remember(tmp_path, {"connection": "gone"})

    rc, _, stderr = _run([], io.StringIO())

    assert rc == 1
    assert not login_file.exists()
    assert "Run 'cortex-training login' in a terminal to set up a new one" in stderr


# ---------------------------------------------------------------------------
# --reset and help
# ---------------------------------------------------------------------------


def test_reset_keeps_profile_unless_confirmed_and_starts_fresh(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    _write_existing_profile(connections_file, "teammate")
    login_file = _remember(tmp_path, {"connection": "saved"})
    # keep the profile (Enter = No), then host, user, PAT, defaults, new name.
    stdin = TTYInput(f"\n{HOST}\nME\n{PAT}\n\n\nfresh\n")

    rc, _, stderr = _run(["--reset"], stdin)

    assert rc == 0, stderr
    assert f"Removed the saved login ({login_file})." in stderr
    assert f"Kept profile [saved] in {connections_file}." in stderr
    assert "Other profiles were not touched: teammate." in stderr
    assert "Snowflake user [EXISTING_USER]" not in stderr  # no defaults from the reset profile
    profiles = tomlkit.parse(connections_file.read_text())
    assert {"saved", "teammate", "fresh"} <= set(profiles)
    assert json.loads(login_file.read_text()) == {"connection": "fresh"}


def test_reset_removes_profile_when_confirmed(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    _write_existing_profile(connections_file, "teammate")
    _remember(tmp_path, {"connection": "saved"})

    rc, _, stderr = _run(["--reset"], TTYInput(f"y\n{HOST}\nME\n{PAT}\n\n\n\n"))

    assert rc == 0, stderr
    profiles = tomlkit.parse(connections_file.read_text())
    assert "saved" not in profiles
    assert "teammate" in profiles
    assert stat.S_IMODE(connections_file.stat().st_mode) == 0o600


def test_reset_without_terminal_only_clears(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    login_file = _remember(tmp_path, {"connection": "saved"})

    rc, stdout, stderr = _run(["--reset"], io.StringIO())

    assert rc == 0
    assert not login_file.exists()
    assert "saved" in tomlkit.parse(connections_file.read_text())  # never removed without consent
    assert json.loads(stdout) == {"logged_in": False, "reset": True}


def test_reset_with_force_removes_saved_profile(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    _remember(tmp_path, {"connection": "saved"})

    rc, _, _ = _run(["--reset", "--force"], io.StringIO())

    assert rc == 0
    assert "saved" not in tomlkit.parse(connections_file.read_text())


def test_reset_keeps_a_remembered_json_config(tmp_path, fake_client):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"base_url": "http://localhost:8084", "database": "DB"}))
    login_file = _remember(tmp_path, {"config_path": str(config)})

    rc, _, stderr = _run(["--reset"], io.StringIO())

    assert rc == 0
    assert not login_file.exists()
    assert config.exists()
    assert f"Kept {config}; login only remembered its path." in stderr


def test_reset_with_nothing_saved(fake_client):
    rc, _, stderr = _run(["--reset"], io.StringIO())

    assert rc == 0
    assert "There was no saved login to remove." in stderr


def test_reset_with_corrupt_saved_login(tmp_path, fake_client):
    login_file = _remember(tmp_path, "{not json")

    rc, _, _ = _run(["--reset"], io.StringIO())

    assert rc == 0
    assert not login_file.exists()


def test_failed_saved_login_is_not_offered_as_defaults(tmp_path, connections_file, fake_client):
    _write_existing_profile(connections_file, "saved")
    _remember(tmp_path, {"connection": "saved"})
    calls = {"n": 0}

    def fail_first_check():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rejected")

    fake_client.on_capacity = fail_first_check

    rc, _, stderr = _run([], TTYInput(f"{HOST}\nME\n{PAT}\n\n\nfresh\n"))

    assert rc == 0, stderr
    assert "Snowflake user [EXISTING_USER]" not in stderr
    assert f"Account host [{HOST}]" not in stderr


def test_reset_cannot_be_combined_with_config_path():
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["login", "config.json", "--reset"])
    assert exc.value.code == 2


def test_login_help_explains_what_it_changes(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["login", "--help"])
    help_text = capsys.readouterr().out

    for expected in (
        "Nothing is changed if your current setup works",
        "connections.toml (other profiles kept, mode 600)",
        "CREATE DATABASE IF NOT EXISTS <database>",
        "it never stores credentials",
        "The PAT is never accepted as a",
        "--reset",
        "cli.md#login",
    ):
        assert expected in help_text
    assert "LOGIN_HOST" not in help_text

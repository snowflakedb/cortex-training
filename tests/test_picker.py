from __future__ import annotations

import json
import logging

import pytest

from cortex_training import _picker
from cortex_training.client import CortexTrainingClient

PROFILES = {
    "default": {"account": "ACME-MAIN", "database": "MAIN_DB", "schema": "TRAIN"},
    "prod": {"account": "ACME-PROD", "database": "PROD_DB"},
    "bare": {"account": "ACME-BARE"},
}


@pytest.fixture(autouse=True)
def _no_picker_env(monkeypatch):
    for name in ("CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT", "CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_SCHEMA"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def profiles(monkeypatch):
    """Snowflake profiles as the Connector would report them; mutate to change them."""
    state = {"default": "default", "profiles": dict(PROFILES)}
    monkeypatch.setattr(_picker, "snowflake_profiles", lambda: (state["default"], state["profiles"]))
    return state


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path / "cortex-training"


def write_config(config_home, data):
    config_home.mkdir(parents=True, exist_ok=True)
    path = config_home / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ─── Which profile ──────────────────────────────────────────────────────────


def test_passed_profile_wins_over_config_and_default(profiles, config_home):
    write_config(config_home, {"profile": "bare"})
    r = _picker.resolve(connection="prod", environ={})
    assert (r.profile, r.sources["profile"]) == ("prod", "passed in")


def test_config_profile_wins_over_the_snowflake_default(profiles, config_home):
    write_config(config_home, {"profile": "prod"})
    r = _picker.resolve(environ={})
    assert (r.profile, r.sources["profile"]) == ("prod", "from config.json")


def test_snowflake_default_profile_is_used_last(profiles, config_home):
    profiles["default"] = "bare"
    r = _picker.resolve(environ={})
    assert (r.profile, r.sources["profile"]) == ("bare", "Snowflake default profile")


def test_unknown_profile_names_the_known_ones(profiles, config_home):
    with pytest.raises(_picker.PickerError, match=r"'nope' \(passed in\) does not exist; known profiles: bare, default, prod"):
        _picker.resolve(connection="nope", environ={})


def test_unknown_config_profile_says_where_it_came_from(profiles, config_home):
    write_config(config_home, {"profile": "gone"})
    with pytest.raises(_picker.PickerError, match=r"'gone' \(from config.json\) does not exist"):
        _picker.resolve(environ={})


def test_ci_credentials_are_used_when_no_profile_exists(profiles, config_home):
    profiles["profiles"] = {}
    env = {"CORTEX_TRAINING_HOST": "acme.snowflakecomputing.com", "CORTEX_TRAINING_PAT": "secret"}
    r = _picker.resolve(environ=env)
    assert (r.profile, r.host, r.pat) == (None, "acme.snowflakecomputing.com", "secret")
    assert r.sources["host"] == "from CORTEX_TRAINING_HOST"
    assert (r.database, r.schema) == ("CORTEX_TRAINING_DB", "PUBLIC")


def test_ci_credentials_are_ignored_with_a_warning_when_a_profile_exists(profiles, config_home, caplog):
    env = {"CORTEX_TRAINING_HOST": "acme.snowflakecomputing.com", "CORTEX_TRAINING_PAT": "secret"}
    with caplog.at_level(logging.WARNING, logger=_picker.__name__):
        r = _picker.resolve(environ=env)
    assert (r.profile, r.host, r.pat) == ("default", None, None)
    assert "CORTEX_TRAINING_HOST/CORTEX_TRAINING_PAT are set but ignored: profile default is used" in caplog.text


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"CORTEX_TRAINING_HOST": "acme.snowflakecomputing.com"}, "CORTEX_TRAINING_PAT is not set"),
        ({"CORTEX_TRAINING_PAT": "secret"}, "CORTEX_TRAINING_HOST is not set"),
        ({}, "No Snowflake connection profile is set up"),
    ],
)
def test_no_profile_and_incomplete_ci_credentials_fail_clearly(profiles, config_home, env, message):
    profiles["profiles"] = {}
    with pytest.raises(_picker.PickerError, match=message):
        _picker.resolve(environ=env)


# ─── Which database and schema ──────────────────────────────────────────────


def test_database_and_schema_passed_in_win(profiles, config_home):
    env = {"CORTEX_TRAINING_DATABASE": "ENV_DB", "CORTEX_TRAINING_SCHEMA": "ENV_SCH"}
    r = _picker.resolve(database="ARG_DB", schema="ARG_SCH", environ=env)
    assert (r.database, r.schema) == ("ARG_DB", "ARG_SCH")
    assert (r.sources["database"], r.sources["schema"]) == ("passed in", "passed in")


def test_env_database_beats_the_profile_and_is_logged(profiles, config_home, caplog):
    with caplog.at_level(logging.WARNING, logger=_picker.__name__):
        r = _picker.resolve(environ={"CORTEX_TRAINING_DATABASE": "ENV_DB"})
    assert (r.database, r.sources["database"]) == ("ENV_DB", "from CORTEX_TRAINING_DATABASE")
    assert (r.schema, r.sources["schema"]) == ("TRAIN", "from profile default")
    assert "Using database ENV_DB from CORTEX_TRAINING_DATABASE" in caplog.text


def test_profile_database_beats_the_default(profiles, config_home):
    r = _picker.resolve(connection="prod", environ={})
    assert (r.database, r.sources["database"]) == ("PROD_DB", "from profile prod")
    assert (r.schema, r.sources["schema"]) == ("PUBLIC", "default")


def test_defaults_when_nothing_names_a_database(profiles, config_home):
    r = _picker.resolve(connection="bare", environ={})
    assert (r.database, r.schema) == ("CORTEX_TRAINING_DB", "PUBLIC")
    assert (r.sources["database"], r.sources["schema"]) == ("default", "default")


def test_blank_values_are_treated_as_unset(profiles, config_home):
    r = _picker.resolve(connection="prod", database="  ", environ={"CORTEX_TRAINING_DATABASE": " "})
    assert r.database == "PROD_DB"


# ─── config.json ────────────────────────────────────────────────────────────


def test_default_config_path_honors_xdg_config_home(config_home):
    assert _picker.config_path() == config_home / "config.json"


def test_default_config_path_without_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _picker.config_path() == tmp_path / ".config" / "cortex-training" / "config.json"


def test_missing_default_config_means_no_settings(config_home):
    assert _picker.load_config() == ({}, config_home / "config.json")


def test_explicit_config_must_exist(tmp_path):
    with pytest.raises(_picker.PickerError, match="config.json not found"):
        _picker.load_config(tmp_path / "missing.json")


def test_explicit_config_replaces_the_default(profiles, config_home, tmp_path):
    write_config(config_home, {"profile": "prod"})
    other = tmp_path / "team.json"
    other.write_text(json.dumps({"profile": "bare", "poll_interval": 2}), encoding="utf-8")
    r = _picker.resolve(ct_config=other, environ={})
    assert (r.profile, r.sources["profile"]) == ("bare", f"from {other}")
    assert r.settings == {"poll_interval": 2}


def test_settings_are_returned_without_the_profile_key(profiles, config_home):
    write_config(config_home, {"profile": "prod", "endpoint": "ct", "poll_timeout": 60, "verify_ssl": False})
    r = _picker.resolve(environ={})
    assert r.settings == {"endpoint": "ct", "poll_timeout": 60, "verify_ssl": False}


@pytest.mark.parametrize("key", ["pat", "host", "database", "schema", "token", "password", "base_url"])
def test_credentials_and_database_are_rejected(config_home, key):
    write_config(config_home, {key: "x"})
    with pytest.raises(_picker.PickerError, match=f"sets {key}, which belong in your Snowflake connection profile"):
        _picker.load_config()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"poll_intervall": 1}, "unknown keys poll_intervall"),
        ({"poll_interval": "1"}, "poll_interval must be int or float"),
        ({"poll_interval": True}, "poll_interval must be int or float"),
        ({"poll_timeout": 0}, "poll_timeout must be greater than 0"),
        ({"verify_ssl": "yes"}, "verify_ssl must be bool"),
        ({"profile": " "}, "profile must not be empty"),
        (["not", "an", "object"], "must hold a JSON object"),
    ],
)
def test_invalid_settings_fail_clearly(config_home, data, message):
    write_config(config_home, data)
    with pytest.raises(_picker.PickerError, match=message):
        _picker.load_config()


def test_invalid_json_fails_clearly(config_home):
    config_home.mkdir(parents=True)
    (config_home / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(_picker.PickerError, match="invalid JSON"):
        _picker.load_config()


# ─── Resolution ─────────────────────────────────────────────────────────────


def test_describe_names_each_source(profiles, config_home):
    write_config(config_home, {"profile": "prod"})
    r = _picker.resolve(environ={"CORTEX_TRAINING_DATABASE": "EXP_DB"})
    assert r.describe() == (
        "profile prod (from config.json), db EXP_DB (from CORTEX_TRAINING_DATABASE), schema PUBLIC (default)"
    )


def test_the_pat_is_not_in_the_repr(profiles, config_home):
    profiles["profiles"] = {}
    r = _picker.resolve(environ={"CORTEX_TRAINING_HOST": "acme.snowflakecomputing.com", "CORTEX_TRAINING_PAT": "secret"})
    assert "secret" not in repr(r)
    assert r.describe().startswith("host acme.snowflakecomputing.com (from CORTEX_TRAINING_HOST)")


# ─── from_configs and the old constructors ─────────────────────────────────


@pytest.fixture
def constructors(monkeypatch):
    calls = []
    monkeypatch.setattr(CortexTrainingClient, "_from_profile", classmethod(lambda cls, *a, **k: calls.append(("profile", a, k)) or "client"))
    monkeypatch.setattr(CortexTrainingClient, "_from_pat", classmethod(lambda cls, *a, **k: calls.append(("pat", a, k)) or "client"))
    return calls


def test_from_configs_builds_a_profile_client(profiles, config_home, constructors):
    write_config(config_home, {"profile": "prod", "poll_interval": 2, "verify_ssl": False})
    assert CortexTrainingClient.from_configs(max_retries=3) == "client"
    assert constructors == [
        ("profile", ("prod",), {"database": "PROD_DB", "schema": "PUBLIC", "verify_ssl": False, "poll_interval": 2, "max_retries": 3})
    ]


def test_from_configs_keyword_arguments_win_over_config(profiles, config_home, constructors):
    write_config(config_home, {"poll_interval": 2, "verify_ssl": False})
    CortexTrainingClient.from_configs("bare", poll_interval=5, verify_ssl=True)
    _, _, kwargs = constructors[0]
    assert (kwargs["poll_interval"], kwargs["verify_ssl"]) == (5, True)


def test_from_configs_builds_a_ci_client_when_no_profile_exists(profiles, config_home, constructors, monkeypatch):
    profiles["profiles"] = {}
    monkeypatch.setenv("CORTEX_TRAINING_HOST", "acme.snowflakecomputing.com")
    monkeypatch.setenv("CORTEX_TRAINING_PAT", "secret")
    CortexTrainingClient.from_configs(database="CI_DB")
    assert constructors == [
        ("pat", (), {"host": "acme.snowflakecomputing.com", "pat": "secret", "database": "CI_DB", "schema": "PUBLIC", "verify_ssl": True})
    ]


@pytest.mark.parametrize(("name", "kind"), [("from_pat", "pat"), ("from_connection_name", "profile")])
def test_old_constructors_warn_and_still_work(constructors, name, kind):
    with pytest.warns(FutureWarning, match=rf"CortexTrainingClient\.{name}\(\) is deprecated.*from_configs\(\)"):
        assert getattr(CortexTrainingClient, name)("arg", key="value") == "client"
    assert constructors == [(kind, ("arg",), {"key": "value"})]

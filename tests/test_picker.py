from __future__ import annotations

import inspect
import json
import logging

import pytest

from cortex_training import _picker
from cortex_training.client import CortexTrainingClient
from cortex_training.client.auth import SnowflakeProfileError

CONNECTIONS_TOML = """\
[default]
account = "ACME-MAIN"
database = "MAIN_DB"
schema = "TRAIN"

[prod]
account = "ACME-PROD"
database = "PROD_DB"

[bare]
account = "ACME-BARE"
"""

CI_ENV = {"CORTEX_TRAINING_HOST": "acme.snowflakecomputing.com", "CORTEX_TRAINING_PAT": "secret"}


@pytest.fixture(autouse=True)
def _no_picker_env(monkeypatch):
    for name in (
        "CORTEX_TRAINING_HOST",
        "CORTEX_TRAINING_PAT",
        "CORTEX_TRAINING_DATABASE",
        "CORTEX_TRAINING_SCHEMA",
        "SNOWFLAKE_DEFAULT_CONNECTION_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def snowflake_home(tmp_path, monkeypatch):
    """A throwaway Snowflake folder holding three profiles; it must exist, or the real one is used."""
    home = tmp_path / "snowflake"
    home.mkdir()
    (home / "connections.toml").write_text(CONNECTIONS_TOML, encoding="utf-8")
    monkeypatch.setenv("SNOWFLAKE_HOME", str(home))
    return home


def no_profiles(snowflake_home):
    (snowflake_home / "connections.toml").unlink()


def write_config(snowflake_home, data):
    folder = snowflake_home / "cortex-training"
    folder.mkdir(exist_ok=True)
    path = folder / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ─── Which profile ──────────────────────────────────────────────────────────


def test_passed_profile_wins_over_config_and_default(snowflake_home):
    write_config(snowflake_home, {"profile": "bare"})
    r = _picker.resolve(connection="prod", environ={})
    assert (r.profile, r.sources["profile"]) == ("prod", "passed in")


def test_config_profile_wins_over_the_snowflake_default(snowflake_home):
    path = write_config(snowflake_home, {"profile": "prod"})
    r = _picker.resolve(environ={})
    assert (r.profile, r.sources["profile"]) == ("prod", f"saved in {path}")


def test_snowflake_default_profile_is_used_last(snowflake_home, monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "bare")
    r = _picker.resolve(environ={})
    assert (r.profile, r.sources["profile"]) == ("bare", "the Snowflake default profile")


def test_profiles_are_read_fresh_on_every_call(snowflake_home):
    assert _picker.resolve(connection="prod", environ={}).database == "PROD_DB"
    (snowflake_home / "connections.toml").write_text('[prod]\naccount = "A"\ndatabase = "NEW_DB"\n')
    assert _picker.resolve(connection="prod", environ={}).database == "NEW_DB"


def test_unknown_profile_names_the_known_ones(snowflake_home):
    with pytest.raises(_picker.PickerError, match=r"'nope' \(passed in\) does not exist; known profiles: bare, default, prod"):
        _picker.resolve(connection="nope", environ={})


def test_unknown_saved_profile_says_where_it_was_saved(snowflake_home):
    path = write_config(snowflake_home, {"profile": "gone"})
    with pytest.raises(_picker.PickerError, match=rf"'gone' \(saved in {path}\) does not exist.*connection=NAME"):
        _picker.resolve(environ={})


@pytest.mark.parametrize(
    ("setup", "source"),
    [
        ("env", "SNOWFLAKE_DEFAULT_CONNECTION_NAME"),
        ("config.toml", "default_connection_name in .*config.toml"),
        ("none", "no default is set, so 'default'"),
    ],
)
def test_a_missing_default_profile_is_an_error_not_a_fall_through(snowflake_home, monkeypatch, setup, source):
    (snowflake_home / "connections.toml").write_text('[x]\naccount = "A"\n')
    if setup == "env":
        monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "gone")
    elif setup == "config.toml":
        (snowflake_home / "config.toml").write_text('default_connection_name = "gone"\n')
    name = "default" if setup == "none" else "gone"
    with pytest.raises(_picker.PickerError, match=rf"default profile '{name}' \({source}\) does not exist; known profiles: x\."):
        _picker.resolve(environ=CI_ENV)


def test_an_unreadable_profile_file_is_a_picker_error(snowflake_home):
    (snowflake_home / "connections.toml").write_text("[broken\n")
    with pytest.raises(_picker.PickerError):
        _picker.resolve(environ={})


def test_ci_credentials_are_used_when_no_profile_exists(snowflake_home):
    no_profiles(snowflake_home)
    r = _picker.resolve(environ=CI_ENV)
    assert (r.profile, r.host, r.pat) == (None, "acme.snowflakecomputing.com", "secret")
    assert r.sources["host"] == "from CORTEX_TRAINING_HOST"
    assert (r.database, r.schema) == ("CORTEX_TRAINING_DB", "PUBLIC")


def test_ci_credentials_are_ignored_with_a_warning_when_a_profile_exists(snowflake_home, caplog):
    with caplog.at_level(logging.WARNING, logger=_picker.__name__):
        r = _picker.resolve(environ=CI_ENV)
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
def test_no_profile_and_incomplete_ci_credentials_fail_clearly(snowflake_home, env, message):
    no_profiles(snowflake_home)
    with pytest.raises(_picker.PickerError, match=message):
        _picker.resolve(environ=env)


# ─── The CI host ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "acme.snowflakecomputing.com",
        "https://acme.snowflakecomputing.com",
        "https://acme.snowflakecomputing.com/",
        " HTTPS://acme.snowflakecomputing.com// ",
    ],
)
def test_ci_host_accepts_a_scheme_and_trailing_slash(snowflake_home, value):
    no_profiles(snowflake_home)
    r = _picker.resolve(environ={**CI_ENV, "CORTEX_TRAINING_HOST": value})
    assert r.host == "acme.snowflakecomputing.com"


def test_ci_host_keeps_a_port(snowflake_home):
    no_profiles(snowflake_home)
    r = _picker.resolve(environ={**CI_ENV, "CORTEX_TRAINING_HOST": "acme.snowflakecomputing.com:8443"})
    assert r.host == "acme.snowflakecomputing.com:8443"


@pytest.mark.parametrize(
    "value",
    [
        "http://acme.snowflakecomputing.com",
        "myorg-myacct",
        "https://app.snowflake.com/myorg/myacct",
        "acme.snowflakecomputing.com/some/path",
        "acme snowflakecomputing.com",
        "https://",
    ],
)
def test_ci_host_that_is_not_a_host_fails_clearly(snowflake_home, value):
    no_profiles(snowflake_home)
    with pytest.raises(_picker.PickerError, match=r"CORTEX_TRAINING_HOST must be your account host, like ACCOUNT\.snowflakecomputing\.com; got "):
        _picker.resolve(environ={**CI_ENV, "CORTEX_TRAINING_HOST": value})


def test_from_configs_never_builds_a_doubled_scheme(snowflake_home, monkeypatch):
    no_profiles(snowflake_home)
    monkeypatch.setenv("CORTEX_TRAINING_HOST", "https://acme.snowflakecomputing.com/")
    monkeypatch.setenv("CORTEX_TRAINING_PAT", "secret")
    monkeypatch.setenv("CORTEX_TRAINING_DISABLE_TELEMETRY", "1")
    client = CortexTrainingClient.from_configs()
    assert client.base_url == "https://acme.snowflakecomputing.com"


# ─── Which database and schema ──────────────────────────────────────────────


def test_database_and_schema_passed_in_win(snowflake_home):
    env = {"CORTEX_TRAINING_DATABASE": "ENV_DB", "CORTEX_TRAINING_SCHEMA": "ENV_SCH"}
    r = _picker.resolve(database="ARG_DB", schema="ARG_SCH", environ=env)
    assert (r.database, r.schema) == ("ARG_DB", "ARG_SCH")
    assert (r.sources["database"], r.sources["schema"]) == ("passed in", "passed in")


def test_env_database_beats_the_profile_and_is_logged(snowflake_home, caplog):
    with caplog.at_level(logging.WARNING, logger=_picker.__name__):
        r = _picker.resolve(environ={"CORTEX_TRAINING_DATABASE": "ENV_DB"})
    assert (r.database, r.sources["database"]) == ("ENV_DB", "from CORTEX_TRAINING_DATABASE")
    assert (r.schema, r.sources["schema"]) == ("TRAIN", "from profile default")
    assert "Using database ENV_DB from CORTEX_TRAINING_DATABASE" in caplog.text


def test_profile_database_beats_the_default(snowflake_home):
    r = _picker.resolve(connection="prod", environ={})
    assert (r.database, r.sources["database"]) == ("PROD_DB", "from profile prod")
    assert (r.schema, r.sources["schema"]) == ("PUBLIC", "default")


def test_defaults_when_nothing_names_a_database(snowflake_home):
    r = _picker.resolve(connection="bare", environ={})
    assert (r.database, r.schema) == ("CORTEX_TRAINING_DB", "PUBLIC")
    assert (r.sources["database"], r.sources["schema"]) == ("default", "default")


def test_blank_values_are_treated_as_unset(snowflake_home):
    r = _picker.resolve(connection="prod", database="  ", environ={"CORTEX_TRAINING_DATABASE": " "})
    assert r.database == "PROD_DB"


# ─── config.json ────────────────────────────────────────────────────────────


def test_config_path_is_in_the_snowflake_folder(snowflake_home):
    assert _picker.config_path() == snowflake_home / "cortex-training" / "config.json"


def test_config_path_follows_snowflake_home(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("SNOWFLAKE_HOME", str(other))
    assert _picker.config_path() == other / "cortex-training" / "config.json"


def test_xdg_config_home_no_longer_applies(snowflake_home, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert _picker.config_path() == snowflake_home / "cortex-training" / "config.json"


def test_missing_config_means_no_saved_profile(snowflake_home):
    assert _picker.load_config() == ({}, snowflake_home / "cortex-training" / "config.json")


@pytest.mark.parametrize("key", ["pat", "host", "database", "schema", "token", "password", "base_url"])
def test_credentials_and_database_are_rejected(snowflake_home, key):
    write_config(snowflake_home, {key: "x"})
    with pytest.raises(_picker.PickerError, match=f'sets {key}, which belong in your Snowflake connection profile.*holds only "profile"'):
        _picker.load_config()


@pytest.mark.parametrize("key", ["endpoint", "poll_interval", "poll_timeout", "verify_ssl", "poll_intervall"])
def test_settings_other_than_profile_are_rejected(snowflake_home, key):
    write_config(snowflake_home, {"profile": "prod", key: 1})
    with pytest.raises(_picker.PickerError, match=rf'unknown keys {key}; it holds only "profile"'):
        _picker.load_config()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"profile": " "}, "profile must be a non-empty string"),
        ({"profile": 3}, "profile must be a non-empty string"),
        (["not", "an", "object"], "must hold a JSON object"),
    ],
)
def test_invalid_config_fails_clearly(snowflake_home, data, message):
    write_config(snowflake_home, data)
    with pytest.raises(_picker.PickerError, match=message):
        _picker.load_config()


def test_invalid_json_fails_clearly(snowflake_home):
    (snowflake_home / "cortex-training").mkdir()
    (snowflake_home / "cortex-training" / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(_picker.PickerError, match="invalid JSON"):
        _picker.load_config()


# ─── Resolution ─────────────────────────────────────────────────────────────


def test_describe_names_each_source(snowflake_home):
    path = write_config(snowflake_home, {"profile": "prod"})
    r = _picker.resolve(environ={"CORTEX_TRAINING_DATABASE": "EXP_DB"})
    assert r.describe() == (
        f"profile prod (saved in {path}), db EXP_DB (from CORTEX_TRAINING_DATABASE), schema PUBLIC (default)"
    )


def test_the_pat_is_not_in_the_repr(snowflake_home):
    no_profiles(snowflake_home)
    r = _picker.resolve(environ=CI_ENV)
    assert "secret" not in repr(r)
    assert r.describe().startswith("host acme.snowflakecomputing.com (from CORTEX_TRAINING_HOST)")


# ─── from_configs and the old constructors ─────────────────────────────────


@pytest.fixture
def constructors(monkeypatch):
    calls = []
    monkeypatch.setattr(CortexTrainingClient, "_from_profile", classmethod(lambda cls, *a, **k: calls.append(("profile", a, k)) or "client"))
    monkeypatch.setattr(CortexTrainingClient, "_from_pat", classmethod(lambda cls, *a, **k: calls.append(("pat", a, k)) or "client"))
    return calls


def test_from_configs_builds_a_profile_client(snowflake_home, constructors):
    write_config(snowflake_home, {"profile": "prod"})
    assert CortexTrainingClient.from_configs(max_retries=3, verify_ssl=False) == "client"
    assert constructors == [
        ("profile", ("prod",), {"database": "PROD_DB", "schema": "PUBLIC", "max_retries": 3, "verify_ssl": False})
    ]


def test_from_configs_has_no_ct_config():
    assert "ct_config" not in inspect.signature(CortexTrainingClient.from_configs).parameters
    assert "ct_config" not in inspect.signature(_picker.resolve).parameters


def test_from_configs_builds_a_ci_client_when_no_profile_exists(snowflake_home, constructors, monkeypatch):
    no_profiles(snowflake_home)
    for name, value in CI_ENV.items():
        monkeypatch.setenv(name, value)
    CortexTrainingClient.from_configs(database="CI_DB")
    assert constructors == [
        ("pat", (), {"host": "acme.snowflakecomputing.com", "pat": "secret", "database": "CI_DB", "schema": "PUBLIC"})
    ]


def test_a_failed_sign_in_names_the_profile_and_where_it_came_from(snowflake_home, monkeypatch):
    path = write_config(snowflake_home, {"profile": "prod"})

    def fail(cls, *args, **kwargs):
        raise SnowflakeProfileError("could not open Snowflake connection 'prod': bad token")

    monkeypatch.setattr(CortexTrainingClient, "_from_profile", classmethod(fail))
    with pytest.raises(SnowflakeProfileError) as raised:
        CortexTrainingClient.from_configs()
    message = str(raised.value)
    assert message.startswith("could not open Snowflake connection 'prod': bad token\n")
    assert f"Profile 'prod' was used because it was saved in {path}. Pass connection=NAME" in message
    assert isinstance(raised.value.__cause__, SnowflakeProfileError)


@pytest.mark.parametrize(("name", "kind"), [("from_pat", "pat"), ("from_connection_name", "profile")])
def test_old_constructors_warn_and_still_work(constructors, name, kind):
    with pytest.warns(FutureWarning, match=rf"CortexTrainingClient\.{name}\(\) is deprecated.*from_configs\(\)"):
        assert getattr(CortexTrainingClient, name)("arg", key="value") == "client"
    assert constructors == [(kind, ("arg",), {"key": "value"})]

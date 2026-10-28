from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap

import threading
from pathlib import Path

import filelock
import pytest

from cortex_training import _profiles


@pytest.fixture(autouse=True)
def snowflake_home(tmp_path, monkeypatch):
    """Point every test at an empty, existing Snowflake folder, never the real one."""
    home = tmp_path / "snowflake"
    home.mkdir()
    monkeypatch.setenv("SNOWFLAKE_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(_profiles.DEFAULT_PROFILE_ENV, raising=False)
    return home


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


CONNECTIONS_TOML = textwrap.dedent(
    """\
    # my profiles
    [default]
    account = "ACME-MAIN"  # keep this comment
    user = "ME"

    [other]
    account = "ACME-OTHER"
    """
)

CONFIG_TOML = textwrap.dedent(
    """\
    default_connection_name = "work"

    [cli]
    ignore_new_version_warning = true

    [connections.work]
    account = "ACME-WORK"
    """
)


# ─── Which file ─────────────────────────────────────────────────────────────


def test_connections_toml_is_read_when_it_exists_and_hides_config_toml(snowflake_home):
    (snowflake_home / "connections.toml").write_text(CONNECTIONS_TOML)
    (snowflake_home / "config.toml").write_text(CONFIG_TOML)
    assert _profiles.profile_file() == snowflake_home / "connections.toml"
    assert sorted(_profiles.read_profiles()) == ["default", "other"]


def test_config_toml_is_read_when_connections_toml_is_missing(snowflake_home):
    (snowflake_home / "config.toml").write_text(CONFIG_TOML)
    assert _profiles.profile_file() == snowflake_home / "config.toml"
    assert _profiles.read_profiles() == {"work": {"account": "ACME-WORK"}}


def test_no_files_means_no_profiles_and_a_new_connections_toml(snowflake_home):
    assert _profiles.read_profiles() == {}
    assert _profiles.profile_file() == snowflake_home / "connections.toml"


def test_config_toml_without_profiles_still_gets_a_new_connections_toml(snowflake_home):
    (snowflake_home / "config.toml").write_text("[cli]\nignore_new_version_warning = true\n")
    assert _profiles.profile_file() == snowflake_home / "connections.toml"


def test_missing_snowflake_home_falls_back_to_the_platform_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path / "does-not-exist"))
    from snowflake.connector.sf_dirs import _resolve_platform_dirs

    assert _profiles.snowflake_dir() == _resolve_platform_dirs().user_config_path


# ─── Writing ────────────────────────────────────────────────────────────────


def test_writing_into_connections_toml_keeps_other_profiles_and_comments(snowflake_home):
    path = snowflake_home / "connections.toml"
    path.write_text(CONNECTIONS_TOML)
    assert _profiles.write_profile("training", {"account": "ACME-TRAIN", "database": "TRAIN_DB"}) == path
    text = path.read_text()
    assert text.startswith(CONNECTIONS_TOML)
    assert '[training]\naccount = "ACME-TRAIN"\ndatabase = "TRAIN_DB"\n' in text
    assert sorted(_profiles.read_profiles()) == ["default", "other", "training"]


def test_writing_into_config_toml_never_creates_connections_toml(snowflake_home):
    config = snowflake_home / "config.toml"
    config.write_text(CONFIG_TOML)
    assert _profiles.write_profile("training", {"account": "ACME-TRAIN"}) == config
    assert not (snowflake_home / "connections.toml").exists()
    text = config.read_text()
    assert text.startswith(CONFIG_TOML)
    assert '[connections.training]\naccount = "ACME-TRAIN"\n' in text
    assert sorted(_profiles.read_profiles()) == ["training", "work"]


def test_writing_with_both_files_leaves_config_toml_alone(snowflake_home):
    (snowflake_home / "connections.toml").write_text(CONNECTIONS_TOML)
    config = snowflake_home / "config.toml"
    config.write_text(CONFIG_TOML)
    _profiles.write_profile("training", {"account": "ACME-TRAIN"})
    assert config.read_text() == CONFIG_TOML


def test_writing_with_no_files_creates_a_private_connections_toml(snowflake_home):
    path = _profiles.write_profile("training", {"account": "ACME-TRAIN", "token": "secret"})
    assert path == snowflake_home / "connections.toml"
    assert mode(path) == 0o600
    assert _profiles.read_profiles() == {"training": {"account": "ACME-TRAIN", "token": "secret"}}


def test_writing_creates_a_missing_folder(tmp_path, monkeypatch):
    folder = tmp_path / "platform" / "snowflake"
    monkeypatch.setattr(_profiles, "snowflake_dir", lambda: folder)
    path = _profiles.write_profile("training", {"account": "ACME-TRAIN"})
    assert path == folder / "connections.toml"
    assert mode(folder) == 0o700


def test_writing_replaces_an_existing_profile_and_makes_the_file_private(snowflake_home):
    path = snowflake_home / "connections.toml"
    path.write_text(CONNECTIONS_TOML)
    path.chmod(0o644)
    _profiles.write_profile("other", {"account": "ACME-NEW", "role": None})
    assert _profiles.read_profiles()["other"] == {"account": "ACME-NEW"}
    assert mode(path) == 0o600


def test_a_failed_write_leaves_the_file_and_no_temp_file(snowflake_home, monkeypatch):
    path = snowflake_home / "connections.toml"
    path.write_text(CONNECTIONS_TOML)

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(_profiles.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        _profiles.write_profile("training", {"account": "ACME-TRAIN"})
    assert path.read_text() == CONNECTIONS_TOML
    assert not [p.name for p in snowflake_home.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize("name", ["", "my profile", "a.b", "x]", "émile"])
def test_invalid_profile_names_are_rejected(name):
    with pytest.raises(_profiles.ProfileFileError, match="invalid profile name"):
        _profiles.write_profile(name, {"account": "ACME"})


def test_a_profile_needs_settings():
    with pytest.raises(_profiles.ProfileFileError, match="needs at least one setting"):
        _profiles.write_profile("training", {})


def test_unreadable_toml_fails_clearly(snowflake_home):
    (snowflake_home / "connections.toml").write_text("[broken\n")
    with pytest.raises(_profiles.ProfileFileError, match="could not read"):
        _profiles.read_profiles()


# ─── Removing ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("file_name", "content", "name", "left"), [
    ("connections.toml", CONNECTIONS_TOML, "other", ["default"]),
    ("config.toml", CONFIG_TOML, "work", []),
])
def test_removing_a_profile_keeps_the_rest(snowflake_home, file_name, content, name, left):
    (snowflake_home / file_name).write_text(content)
    assert _profiles.remove_profile(name) is True
    assert sorted(_profiles.read_profiles()) == left
    assert _profiles.remove_profile(name) is False


def test_removing_from_config_toml_keeps_its_other_settings(snowflake_home):
    config = snowflake_home / "config.toml"
    config.write_text(CONFIG_TOML)
    _profiles.remove_profile("work")
    text = config.read_text()
    assert 'default_connection_name = "work"' in text and "[cli]" in text


def test_removing_with_no_file_is_a_no_op(snowflake_home):
    assert _profiles.remove_profile("training") is False
    assert list(snowflake_home.iterdir()) == []


# ─── Default profile ────────────────────────────────────────────────────────


def test_default_profile_name_order(snowflake_home, monkeypatch):
    assert _profiles.default_profile_name() == "default"
    (snowflake_home / "config.toml").write_text(CONFIG_TOML)
    assert _profiles.default_profile_name() == "work"
    monkeypatch.setenv(_profiles.DEFAULT_PROFILE_ENV, "terminal")
    assert _profiles.default_profile_name() == "terminal"


# ─── Parity with the Connector ──────────────────────────────────────────────

_CONNECTOR_VIEW = textwrap.dedent(
    """\
    import json
    from snowflake.connector.config_manager import CONFIG_MANAGER
    print(json.dumps({
        "default": CONFIG_MANAGER["default_connection_name"],
        "profiles": sorted(CONFIG_MANAGER["connections"]),
    }))
    """
)


@pytest.mark.parametrize("files", [
    {"connections.toml": CONNECTIONS_TOML},
    {"config.toml": CONFIG_TOML},
    {"connections.toml": CONNECTIONS_TOML, "config.toml": CONFIG_TOML},
])
def test_the_connector_sees_the_same_profiles_and_default(snowflake_home, files):
    for name, content in files.items():
        path = snowflake_home / name
        path.write_text(content)
        path.chmod(0o600)
    _profiles.write_profile("training", {"account": "ACME-TRAIN"})
    env = {**os.environ, "SNOWFLAKE_HOME": str(snowflake_home)}
    result = subprocess.run(
        [sys.executable, "-c", _CONNECTOR_VIEW], env=env, capture_output=True, text=True, check=True
    )
    connector = json.loads(result.stdout.strip().splitlines()[-1])
    assert connector == {"default": _profiles.default_profile_name(), "profiles": sorted(_profiles.read_profiles())}


def test_profile_tables_are_found_by_file_name(tmp_path):
    connections = {"default": {"account": "A"}}
    config = {"default_connection_name": "work", "connections": {"work": {"account": "B"}}}
    assert _profiles._profile_tables(tmp_path / "connections.toml", connections) is connections
    assert _profiles._profile_tables(tmp_path / "config.toml", config) is config["connections"]
    with pytest.raises(_profiles.ProfileFileError, match="not a Snowflake profile file"):
        _profiles._profile_tables(tmp_path / "other.toml", connections)


# ─── Review follow-ups: symlinks, concurrency, durability, names, values ────


def test_a_symlinked_connections_toml_stays_linked(snowflake_home, tmp_path):
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real = dotfiles / "connections.toml"
    real.write_text(CONNECTIONS_TOML)
    link = snowflake_home / "connections.toml"
    link.symlink_to(real)

    _profiles.write_profile("training", {"account": "ACME-TRAIN"})

    assert link.is_symlink()
    assert '[training]\naccount = "ACME-TRAIN"\n' in real.read_text()
    assert sorted(_profiles.read_profiles()) == ["default", "other", "training"]
    assert _profiles.remove_profile("training") is True
    assert link.is_symlink() and "[training]" not in real.read_text()


def test_concurrent_writers_keep_every_profile(snowflake_home):
    (snowflake_home / "connections.toml").write_text(CONNECTIONS_TOML)
    errors = []

    def write(index):
        try:
            _profiles.write_profile(f"p{index}", {"account": f"ACME-{index}"})
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(_profiles.read_profiles()) == 22  # default, other, and 20 new
    assert not [p.name for p in snowflake_home.iterdir() if p.name.endswith(".tmp")]


def test_the_lock_is_held_while_the_file_is_written(snowflake_home, monkeypatch):
    (snowflake_home / "connections.toml").write_text(CONNECTIONS_TOML)
    lock_file = snowflake_home / ".connections.toml.lock"
    real_replace = _profiles._replace
    held = []

    def replace(path, text):
        try:
            filelock.FileLock(lock_file, timeout=0).acquire()
        except filelock.Timeout:
            held.append(True)
        real_replace(path, text)

    monkeypatch.setattr(_profiles, "_replace", replace)
    _profiles.write_profile("training", {"account": "ACME"})

    assert held == [True]
    assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600


def test_the_file_is_flushed_to_disk_before_it_replaces_the_old_one(snowflake_home, monkeypatch):
    events = []
    real_fsync, real_replace = _profiles.os.fsync, _profiles.os.replace
    monkeypatch.setattr(_profiles.os, "fsync", lambda fd: events.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(_profiles.os, "replace", lambda a, b: events.append("replace") or real_replace(a, b))

    _profiles.write_profile("training", {"account": "ACME-TRAIN"})

    assert events.index("fsync") < events.index("replace")


def test_existing_profiles_with_any_name_can_be_replaced_and_removed(snowflake_home):
    (snowflake_home / "connections.toml").write_text('["prod.us"]\naccount = "ACME-US"\n')
    assert sorted(_profiles.read_profiles()) == ["prod.us"]

    _profiles.write_profile("prod.us", {"account": "ACME-US2"})
    assert _profiles.read_profiles()["prod.us"] == {"account": "ACME-US2"}
    assert _profiles.remove_profile("prod.us") is True
    assert _profiles.read_profiles() == {}


def test_new_profiles_still_need_a_simple_name(snowflake_home):
    with pytest.raises(_profiles.ProfileFileError, match="invalid profile name"):
        _profiles.write_profile("prod.eu", {"account": "ACME-EU"})


def test_path_values_are_written_as_strings(snowflake_home):
    _profiles.write_profile("keys", {"account": "ACME", "private_key_file": Path("/keys/me.p8")})
    assert _profiles.read_profiles()["keys"]["private_key_file"] == "/keys/me.p8"


def test_values_toml_can_not_hold_fail_clearly(snowflake_home):
    with pytest.raises(_profiles.ProfileFileError, match="can't write when"):
        _profiles.write_profile("odd", {"account": "ACME", "when": object()})
    assert not (snowflake_home / "connections.toml").exists()


def test_a_value_that_fails_is_never_in_the_error(snowflake_home):
    class Secret:
        def __repr__(self):
            return "pat-SECRET"

        __str__ = __repr__

    with pytest.raises(_profiles.ProfileFileError) as raised:
        _profiles.write_profile("odd", {"account": "ACME", "token": Secret()})
    assert str(raised.value) == "profile 'odd': can't write token: a Secret value can't be stored in TOML"
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


def test_replacing_a_profile_drops_keys_not_given(snowflake_home):
    _profiles.write_profile("keys", {"account": "ACME", "private_key_file": "/k.p8"})
    _profiles.write_profile("keys", {"account": "ACME"})
    assert _profiles.read_profiles()["keys"] == {"account": "ACME"}

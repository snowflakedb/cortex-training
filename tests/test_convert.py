from __future__ import annotations

import io
import json
import re

import pytest
import requests

import cortex_training.cli.commands as _cli
from cortex_training import _convert
from cortex_training import _picker
from cortex_training import _profiles
from tests.test_login import ApiError
from tests.test_login import write_profiles

HOST = "acme-main.snowflakecomputing.com"
PAT = "old-secret-token"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Empty Snowflake folder and config home for every test; never the real ones."""
    snowflake = tmp_path / "snowflake"
    snowflake.mkdir()
    monkeypatch.setenv("SNOWFLAKE_HOME", str(snowflake))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "CORTEX_TRAINING_LOGIN_FILE", *(n for names in _convert._OLD_ENV.values() for n in names)):
        monkeypatch.delenv(name, raising=False)
    return snowflake


@pytest.fixture
def old_file(tmp_path):
    """Write an old JSON config and return its path."""

    def write(content=None, name="ct-config.json"):
        path = tmp_path / name
        if content is None:
            content = {"host": HOST, "pat": PAT, "database": "CORTEX_TRAINING_DB", "schema": "PUBLIC"}
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        return path

    return write


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


class World:
    """Answers the PAT client's requests like Snowflake would."""

    def __init__(self, databases=("CORTEX_TRAINING_DB",)):
        self.databases = set(databases)
        self.schemas = {"PUBLIC"}
        self.user, self.role = "ME", "TRAINER"
        self.sql_error = None
        self.capacity_error = None
        self.create_error = None
        self.statements = []
        self.opened = 0
        self.profile_error = None
        self.profiles_opened = []

    def open(self, old):
        self.opened += 1
        return FakePatClient(self, old)

    def open_profile(self, profile, database, schema):
        # Commands sign in from the profile file, so it must be written by now.
        assert profile in _profiles.read_profiles()
        self.profiles_opened.append((profile, database, schema))
        return FakeProfileClient(self)


class FakeProfileClient:
    def __init__(self, world):
        self.world = world

    def get_capacity(self):
        if self.world.profile_error:
            raise self.world.profile_error
        return {}

    def close(self):
        pass


class FakePatClient:
    """A PAT client: SQL goes through ``_send`` to the SQL API, as in the real client."""

    _auth_provider = None
    base_url = f"https://{HOST}"

    def __init__(self, world, old):
        self.world, self.database, self.schema = world, old.database, old.schema
        self.closed = False

    def _send(self, method, url, **kwargs):
        assert url == f"{self.base_url}/api/v2/statements"
        statement = kwargs["json"]["statement"]
        self.world.statements.append(statement)
        if self.world.sql_error:
            raise self.world.sql_error
        if statement.startswith("SELECT CURRENT_USER()"):
            return _rows(["USER_NAME", "ROLE_NAME"], [[self.world.user, self.world.role]])
        if statement.startswith("SHOW DATABASES LIKE"):
            name = re.search(r"'(.*)'", statement).group(1)
            return _rows(["name"], [[name]] if name in self.world.databases else [])
        if statement.startswith("SHOW SCHEMAS LIKE"):
            name = re.search(r"'(.*)'", statement).group(1)
            return _rows(["name"], [[name]] if name in self.world.schemas else [])
        if statement.startswith("CREATE DATABASE"):
            if self.world.create_error:
                raise self.world.create_error
            self.world.databases.add(statement.split()[-1])
            return _rows(["status"], [["created"]])
        raise AssertionError(f"unexpected statement: {statement}")

    def get_capacity(self):
        if self.world.capacity_error:
            raise self.world.capacity_error
        if self.database not in self.world.databases:
            raise ApiError(400, "517602", f"Schema {self.database}.PUBLIC is not found or not authorized")
        return {}

    def _current_role(self):
        return None

    def close(self):
        self.closed = True


def _rows(columns, data):
    return FakeResponse({"resultSetMetaData": {"rowType": [{"name": c} for c in columns]}, "data": data})


def convert(world, path, answers=()):
    stdin, stdout, stderr = io.StringIO("".join(f"{a}\n" for a in answers)), io.StringIO(), io.StringIO()
    code = _convert.run_convert(
        str(path), stdin=stdin, stdout=stdout, stderr=stderr, interactive=True, open_client=world.open, open_profile=world.open_profile
    )
    summary = json.loads(stdout.getvalue()) if code == 0 else None
    return code, summary, stderr.getvalue()


def config():
    path = _picker.config_path()
    return json.loads(path.read_text()) if path.exists() else None


def nothing_written(path, snowflake):
    """The old file is untouched, and no profile, settings, or backup was written."""
    return (
        path.exists()
        and not (snowflake / "connections.toml").exists()
        and config() is None
        and not path.with_name(path.name + ".bak").exists()
    )


# ─── The happy path ─────────────────────────────────────────────────────────

# Check now? [Y] -> profile name [cortex-training] -> go ahead? y (No is the default)
YES = ["", "", "y"]


def test_an_old_config_becomes_a_profile_and_settings(old_file, isolated):
    path = old_file({"host": HOST, "pat": PAT, "database": "MY_DB", "schema": "RUNS", "poll_timeout": 3600, "endpoint": "ct"})
    world = World(databases={"MY_DB"})
    world.schemas = {"RUNS"}
    code, summary, err = convert(world, path, YES)
    assert code == 0, err
    assert _profiles.read_profiles() == {
        "cortex-training": {
            "account": "acme-main",
            "host": HOST,
            "user": "ME",
            "role": "TRAINER",
            "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
            "token": PAT,
            "database": "MY_DB",
            "schema": "RUNS",
        }
    }
    # config.json holds only the profile; the plan and the ending say what wasn't kept.
    assert config() == {"profile": "cortex-training"}
    assert err.count("Not kept: endpoint ct, poll_timeout 3600.") == 2
    assert "--endpoint ct, --poll-timeout 3600" in err
    assert not path.exists()
    backup = json.loads(path.with_name("ct-config.json.bak").read_text())
    assert backup == {"host": HOST, "database": "MY_DB", "schema": "RUNS", "poll_timeout": 3600, "endpoint": "ct"}
    assert "  ✓ Database MY_DB is ready" in err
    assert "SHOW SCHEMAS LIKE 'RUNS' IN DATABASE MY_DB" in world.statements
    assert "You're all set. Cortex Training will use profile cortex-training." in err
    assert "Try it:  cortex-training --connection cortex-training capacity" in err
    assert summary["profile"] == "cortex-training" and summary["deleted"] == [str(path)]
    assert PAT not in err and PAT not in json.dumps(summary)


def test_it_shows_the_file_and_asks_before_checking(old_file):
    path = old_file()
    code, _, err = convert(World(), path, YES)
    assert code == 0
    assert f"host      {HOST}" in err and "pat       ••••••••" in err
    assert "This only reads; nothing is changed.\nCheck now? [Y/n]" in err
    assert "  ✓ Signed in as ME, role TRAINER" in err
    assert "  ✓ Cortex Training is enabled on this account" in err
    assert "  ✓ Database CORTEX_TRAINING_DB is ready" in err
    assert "This will:" in err and "Go ahead? [y/N]" in err


def test_the_connection_wrapper_and_old_spellings_are_read(old_file):
    path = old_file({"connection": {"host": HOST, "pat": PAT, "db": "CORTEX_TRAINING_DB", "no_verify_ssl": True}})
    code, _, err = convert(World(), path, YES)
    assert code == 0, err
    assert _profiles.read_profiles()["cortex-training"]["database"] == "CORTEX_TRAINING_DB"
    assert config() == {"profile": "cortex-training"}
    assert "Not kept: verify_ssl False." in err and "--no-verify-ssl" in err
    assert json.loads(path.with_name("ct-config.json.bak").read_text()) == {
        "connection": {"host": HOST, "db": "CORTEX_TRAINING_DB", "no_verify_ssl": True}
    }


def test_an_underscore_host_is_kept_as_written(old_file):
    path = old_file({"host": "my_acct.us-west-2.snowflakecomputing.com", "pat": PAT})
    code, _, _ = convert(World(), path, YES)
    assert code == 0
    profile = _profiles.read_profiles()["cortex-training"]
    assert (profile["host"], profile["account"]) == ("my_acct.us-west-2.snowflakecomputing.com", "my_acct")


def test_login_json_is_deleted_only_when_it_points_to_the_old_file(old_file):
    path = old_file()
    login_file = _cli._login_state_path()
    login_file.parent.mkdir(parents=True)
    login_file.write_text(json.dumps({"config_path": str(path)}))
    code, summary, _ = convert(World(), path, YES)
    assert code == 0
    assert not login_file.exists()
    assert summary["deleted"] == [str(path), str(login_file)]


def test_login_json_pointing_elsewhere_is_kept(old_file, tmp_path):
    path = old_file()
    login_file = _cli._login_state_path()
    login_file.parent.mkdir(parents=True)
    login_file.write_text(json.dumps({"config_path": str(tmp_path / "other.json")}))
    code, _, _ = convert(World(), path, YES)
    assert code == 0
    assert login_file.exists()


# ─── Files that can't be converted ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "isn't valid JSON (line 1: Expecting property name enclosed in double quotes). Fix the file"),
        ({"host": HOST, "database": "DB"}, "has no pat, and neither CORTEX_TRAINING_PAT nor SNOWFLAKE_PAT is set, so there's no sign-in to convert."),
        ({"base_url": "http://localhost:8080", "pat": PAT}, "points to a test server (base_url), which can't become a Snowflake profile."),
        ({"host": HOST, "pat": PAT, "password": "x"}, "has keys an old config can't have: password."),
        ([1, 2], "must hold a JSON object."),
    ],
)
def test_a_file_that_cant_be_converted_says_what_to_do_and_writes_nothing(old_file, isolated, content, message):
    path = old_file(content)
    world = World()
    code, _, err = convert(world, path)
    assert code == 1
    assert message in err
    assert "cortex-training login" in err
    assert world.opened == 0
    assert nothing_written(path, isolated)


def test_a_missing_file_says_so(tmp_path):
    code, _, err = convert(World(), tmp_path / "nope.json")
    assert code == 1
    assert "can't be used: there's no file there." in err


# ─── The check ──────────────────────────────────────────────────────────────


def test_declining_the_check_changes_nothing(old_file, isolated):
    path = old_file()
    world = World()
    code, _, err = convert(world, path, ["n"])
    assert code == 1
    assert "Nothing was changed." in err
    assert world.opened == 0
    assert nothing_written(path, isolated)


def test_a_rejected_pat_is_explained(old_file, isolated):
    path = old_file()
    world = World()
    world.sql_error = requests.exceptions.HTTPError("401 Client Error", response=FakeResponse({}, status_code=401))
    code, _, err = convert(world, path, [""])
    assert code == 1
    assert re.search(r"Snowflake rejected the PAT in \S+ct-config\.json\. The most common causes:", err)
    assert "The PAT was rejected." not in err
    assert "Your user has no network policy." in err
    assert "The token is expired or mistyped" in err
    assert err.rstrip().endswith("Nothing was changed.")
    assert nothing_written(path, isolated)


def test_an_unreachable_host_is_explained(old_file, isolated):
    path = old_file()
    world = World()
    world.sql_error = requests.exceptions.ConnectionError("Max retries exceeded (Connection refused)")
    code, _, err = convert(world, path, [""])
    assert code == 1
    assert f"Couldn't reach {HOST}. Check the address in" in err and "your VPN" in err
    assert err.rstrip().endswith("Nothing was changed.")
    assert nothing_written(path, isolated)


def test_a_host_missing_from_private_dns_points_to_the_privatelink_fix(old_file, isolated):
    path = old_file({"host": "my_acct.us-west-2.privatelink.snowflakecomputing.com", "pat": PAT})
    world = World()
    world.sql_error = requests.exceptions.ConnectionError("NameResolutionError: Failed to resolve 'my-acct...'")
    code, _, err = convert(world, path, [""])
    assert code == 1
    assert "Couldn't find my-acct.us-west-2.privatelink.snowflakecomputing.com on the network." in err
    assert "add a DNS record for this hyphen form" in err
    assert nothing_written(path, isolated)


def test_an_account_without_cortex_training_is_turned_away(old_file, isolated):
    path = old_file()
    world = World()
    world.capacity_error = ApiError(403, "517600", "Cortex Training API is not enabled for this account")
    code, _, err = convert(world, path, [""])
    assert code == 1
    assert "Cortex Training isn't enabled on account acme-main" in err
    assert nothing_written(path, isolated)


def test_a_missing_database_is_created_only_after_asking(old_file):
    path = old_file()
    world = World(databases=set())
    # Check [Y] -> run CREATE [y] -> name -> go ahead
    code, summary, err = convert(world, path, ["", "y", "", "y"])
    assert code == 0, err
    assert "CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB;" in err and "Run it? [y/N]" in err
    assert "CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB" in world.statements
    assert summary["created_database"] is True


def test_declining_the_create_changes_nothing(old_file, isolated):
    path = old_file()
    world = World(databases=set())
    code, _, err = convert(world, path, ["", "n"])
    assert code == 1
    assert "To create it yourself, run in Snowsight: CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB;" in err
    assert not any(statement.startswith("CREATE") for statement in world.statements)
    assert nothing_written(path, isolated)


# ─── The plan ───────────────────────────────────────────────────────────────


def test_a_taken_name_gets_the_next_free_one_and_is_never_replaced(old_file, isolated):
    write_profiles(isolated, '[cortex-training]\naccount = "OTHER"\nuser = "SOMEONE"\n')
    path = old_file()
    # Check -> name: typing the taken name is refused -> accept the suggestion -> go ahead
    code, summary, err = convert(World(), path, ["", "cortex-training", "", "y"])
    assert code == 0, err
    assert "Profile cortex-training already exists, so the new one will be called cortex-training-2." in err
    assert "Profile cortex-training already exists; choose another name." in err
    profiles = _profiles.read_profiles()
    assert profiles["cortex-training"] == {"account": "OTHER", "user": "SOMEONE"}
    assert profiles["cortex-training-2"]["token"] == PAT
    assert summary["profile"] == "cortex-training-2"


def test_settings_at_their_defaults_are_not_reported(old_file):
    # The 3.1 bug bash file lists every setting at its default; dropping them changes nothing.
    path = old_file({"host": HOST, "pat": PAT, "endpoint": "cortex-training", "poll_interval": 0.5,
                     "poll_timeout": 1800.0, "verify_ssl": True})
    code, _, err = convert(World(), path, YES)
    assert code == 0, err
    assert "Not kept" not in err
    assert config() == {"profile": "cortex-training"}


def test_a_config_json_that_can_t_be_read_back_stops_before_anything_changes(old_file, isolated):
    _picker.config_path().parent.mkdir(parents=True)
    _picker.config_path().write_text(json.dumps({"poll_timeout": 60}))
    path = old_file()
    code, _, err = convert(World(), path, ["", ""])
    assert code == 1
    assert 'it holds only "profile"' in err
    assert json.loads(_picker.config_path().read_text()) == {"poll_timeout": 60}
    assert path.exists() and "cortex-training" not in _profiles.read_profiles()


def test_declining_the_plan_changes_nothing(old_file, isolated):
    path = old_file()
    code, _, err = convert(World(), path, ["", "", "n"])
    assert code == 1
    assert "Nothing was changed." in err
    assert nothing_written(path, isolated)


# ─── All or nothing ─────────────────────────────────────────────────────────


def test_a_failed_write_puts_every_file_back(old_file, isolated, monkeypatch):
    write_profiles(isolated, '[other]\naccount = "OTHER"\nuser = "SOMEONE"\n')
    profiles_before = (isolated / "connections.toml").read_bytes()
    _picker.config_path().parent.mkdir(parents=True)
    _picker.config_path().write_text(json.dumps({"profile": "other"}))
    path = old_file()
    real_write = _profiles.write_file

    def fail_on_backup(target, text):
        if target.name.endswith(".bak"):
            raise OSError(28, "No space left on device")
        real_write(target, text)

    monkeypatch.setattr(_profiles, "write_file", fail_on_backup)
    code, _, err = convert(World(), path, YES)
    assert code == 1
    assert "so every file was put back as it was." in err
    assert (isolated / "connections.toml").read_bytes() == profiles_before
    assert config() == {"profile": "other"}
    assert path.exists() and not path.with_name("ct-config.json.bak").exists()


def test_a_failed_delete_puts_every_file_back(old_file, isolated, monkeypatch):
    path = old_file()
    login_file = _cli._login_state_path()
    login_file.parent.mkdir(parents=True)
    login_file.write_text(json.dumps({"config_path": str(path)}))
    real_unlink = type(path).unlink

    def fail_on_login_file(self, *args, **kwargs):
        if self == login_file:
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    # Only this patch is undone before the checks; monkeypatch.undo() would also undo
    # the isolated folders, and the checks would then read the real ones.
    with monkeypatch.context() as patch:
        patch.setattr(type(path), "unlink", fail_on_login_file)
        code, _, err = convert(World(), path, YES)
    assert code == 1
    assert "so every file was put back as it was." in err
    assert path.exists() and json.loads(path.read_text())["pat"] == PAT
    assert login_file.exists()
    assert "cortex-training" not in _profiles.read_profiles()
    assert config() is None


# ─── The command ────────────────────────────────────────────────────────────


def test_it_needs_a_terminal(old_file):
    stderr = io.StringIO()
    code = _convert.run_convert(str(old_file()), stdin=io.StringIO(), stdout=io.StringIO(), stderr=stderr, interactive=False)
    assert code == 2
    assert "cortex-training login is interactive, so it needs a terminal." in stderr.getvalue()


def test_the_flag_takes_no_profile_name():
    stderr = io.StringIO()
    code = _cli.main(["login", "train", "--convert-legacy-config", "ct.json"], stdout=io.StringIO(), stderr=stderr)
    assert code == 2
    assert "takes no profile name" in stderr.getvalue()


# ─── Review follow-ups (Da, #166) ───────────────────────────────────────────


def test_a_pat_left_out_of_the_file_comes_from_the_env_var_the_old_cli_used(old_file, monkeypatch):
    path = old_file({"host": HOST, "database": "CORTEX_TRAINING_DB"})
    monkeypatch.setenv("CORTEX_TRAINING_PAT", "env-secret-token")
    code, summary, err = convert(World(), path, YES)
    assert code == 0, err
    assert _profiles.read_profiles()["cortex-training"]["token"] == "env-secret-token"
    assert "pat       •••••••• (from CORTEX_TRAINING_PAT)" in err
    assert "env-secret-token" not in err and "env-secret-token" not in json.dumps(summary)


def test_a_database_left_out_of_the_file_comes_from_the_env_var_not_the_default(old_file, monkeypatch):
    path = old_file({"host": HOST, "pat": PAT})
    monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "TEAM_DB")
    world = World(databases={"TEAM_DB"})
    code, summary, err = convert(world, path, YES)
    assert code == 0, err
    profile = _profiles.read_profiles()["cortex-training"]
    assert (profile["database"], profile["schema"]) == ("TEAM_DB", "PUBLIC")
    assert "database  TEAM_DB (from CORTEX_TRAINING_DATABASE)    schema PUBLIC (the default)" in err
    assert "check that database TEAM_DB and schema PUBLIC are visible" in err


def test_the_file_beats_the_env_vars_and_cortex_training_beats_snowflake(old_file, monkeypatch):
    path = old_file({"pat": PAT, "database": "MY_DB"})
    monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "TEAM_DB")
    monkeypatch.setenv("SNOWFLAKE_HOST", "acme-other.snowflakecomputing.com")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "SF_SCHEMA")
    monkeypatch.setenv("CORTEX_TRAINING_SCHEMA", "CT_SCHEMA")
    world = World(databases={"MY_DB"})
    world.schemas = {"CT_SCHEMA"}
    code, _, err = convert(world, path, YES)
    assert code == 0, err
    profile = _profiles.read_profiles()["cortex-training"]
    assert (profile["host"], profile["database"], profile["schema"]) == ("acme-other.snowflakecomputing.com", "MY_DB", "CT_SCHEMA")
    assert "host      acme-other.snowflakecomputing.com (from SNOWFLAKE_HOST)" in err
    assert "database  MY_DB    schema CT_SCHEMA (from CORTEX_TRAINING_SCHEMA)" in err


def test_a_database_name_from_an_env_var_that_isnt_an_identifier_stops(old_file, isolated, monkeypatch):
    path = old_file({"host": HOST, "pat": PAT})
    monkeypatch.setenv("CORTEX_TRAINING_DATABASE", "team db")
    world = World()
    code, _, err = convert(world, path)
    assert code == 1
    assert "CORTEX_TRAINING_DATABASE: 'team db' isn't a valid database name." in err
    assert world.opened == 0 and nothing_written(path, isolated)


def test_enter_at_go_ahead_changes_nothing(old_file, isolated):
    # Converting deletes the old file, so No is the default.
    path = old_file()
    code, _, err = convert(World(), path, ["", "", ""])
    assert code == 1
    assert "Go ahead? [y/N]" in err and "Nothing was changed." in err
    assert nothing_written(path, isolated)


def test_a_symlinked_config_json_keeps_its_link(old_file, tmp_path):
    target = tmp_path / "dotfiles" / "ct-config.json"
    target.parent.mkdir()
    target.write_text("{}")
    link = _picker.config_path()
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    code, _, err = convert(World(), old_file(), YES)
    assert code == 0, err
    assert link.is_symlink()
    assert json.loads(target.read_text()) == {"profile": "cortex-training"}


def test_an_old_file_at_the_new_config_json_path_says_so(old_file, isolated):
    path = _picker.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"host": HOST, "pat": PAT, "database": "CORTEX_TRAINING_DB"}))
    world = World()
    code, _, err = convert(world, path)
    assert code == 1
    assert "is where the new config.json goes, and it can hold only \"profile\". Move the old config" in err
    assert world.opened == 0 and path.exists()


# ─── Review follow-ups (Da, #202) ───────────────────────────────────────────


def _point_login_json_at(path):
    login_file = _cli._login_state_path()
    login_file.parent.mkdir(parents=True, exist_ok=True)
    login_file.write_text(json.dumps({"config_path": str(path)}))
    return login_file


def test_the_plan_says_it_signs_in_with_the_new_profile_before_deleting(old_file):
    path = old_file()
    _point_login_json_at(path)
    world = World()
    code, _, err = convert(world, path, YES)
    assert code == 0, err
    assert "  4. Sign in with profile cortex-training to check that it works" in err
    assert "  5. Only if it works, delete " in err
    assert "  ✓ Signed in with profile cortex-training" in err
    assert world.profiles_opened == [("cortex-training", "CORTEX_TRAINING_DB", "PUBLIC")]


def test_old_files_are_removed_only_when_the_new_profile_signs_in(old_file):
    path = old_file()
    login_file = _point_login_json_at(path)
    code, _, err = convert(World(), path, YES)
    assert code == 0, err
    assert not path.exists() and not login_file.exists()


def test_a_new_profile_that_cant_sign_in_puts_every_file_back(old_file, isolated):
    path = old_file()
    login_file = _point_login_json_at(path)
    login_before = login_file.read_text()
    world = World()
    world.profile_error = ApiError(403, "390100", "Programmatic access token is invalid")
    code, _, err = convert(world, path, YES)
    assert code == 1
    assert "Signing in with profile cortex-training didn't work:" in err
    assert "Every file was put back as it was;" in err and "are kept." in err
    assert "run the command again in a few minutes." in err
    assert path.exists() and login_file.read_text() == login_before
    assert not (isolated / "connections.toml").exists() and config() is None
    assert not path.with_name("ct-config.json.bak").exists()

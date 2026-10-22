from __future__ import annotations

import io
import json
import re

import pytest
import requests

from cortex_training import _cli
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
    for name in ("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT", "CORTEX_TRAINING_LOGIN_FILE"):
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

    def open(self, old):
        self.opened += 1
        return FakePatClient(self, old)


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
    code = _convert.run_convert(str(path), stdin=stdin, stdout=stdout, stderr=stderr, interactive=True, open_client=world.open)
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

# Check now? [Y] -> profile name [cortex-training] -> go ahead? [Y]
YES = ["", "", ""]


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
    assert config() == {"poll_timeout": 3600, "endpoint": "ct", "profile": "cortex-training"}
    assert not path.exists()
    backup = json.loads(path.with_name("ct-config.json.bak").read_text())
    assert backup == {"host": HOST, "database": "MY_DB", "schema": "RUNS", "poll_timeout": 3600, "endpoint": "ct"}
    assert "  ✓ Database MY_DB is ready" in err
    assert "SHOW SCHEMAS LIKE 'RUNS' IN DATABASE MY_DB" in world.statements
    assert "You're all set. Cortex Training will use profile cortex-training." in err
    assert "Run your command again." in err
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
    assert "This will:" in err and "Go ahead? [Y/n]" in err


def test_the_connection_wrapper_and_old_spellings_are_read(old_file):
    path = old_file({"connection": {"host": HOST, "pat": PAT, "db": "CORTEX_TRAINING_DB", "no_verify_ssl": True}})
    code, _, err = convert(World(), path, YES)
    assert code == 0, err
    assert _profiles.read_profiles()["cortex-training"]["database"] == "CORTEX_TRAINING_DB"
    assert config() == {"verify_ssl": False, "profile": "cortex-training"}
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
        ({"host": HOST, "database": "DB"}, "has no pat, so there's no sign-in to convert. Run `cortex-training login`"),
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
    code, summary, err = convert(world, path, ["", "y", "", ""])
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
    code, summary, err = convert(World(), path, ["", "cortex-training", "", ""])
    assert code == 0, err
    assert "Profile cortex-training already exists, so the new one will be called cortex-training-2." in err
    assert "Profile cortex-training already exists; choose another name." in err
    profiles = _profiles.read_profiles()
    assert profiles["cortex-training"] == {"account": "OTHER", "user": "SOMEONE"}
    assert profiles["cortex-training-2"]["token"] == PAT
    assert summary["profile"] == "cortex-training-2"


def test_a_setting_that_differs_from_config_json_is_asked_about(old_file):
    _picker.config_path().parent.mkdir(parents=True)
    _picker.config_path().write_text(json.dumps({"poll_timeout": 60}))
    path = old_file({"host": HOST, "pat": PAT, "poll_timeout": 3600})
    # Check -> name -> keep config.json's value [1] -> go ahead
    code, _, err = convert(World(), path, ["", "", "", ""])
    assert code == 0, err
    explained = err.index("config.json already sets poll_timeout to 60; the old config sets 3600.\n")
    # The explanation, then the choices, then a short question.
    assert explained < err.index("1  Keep 60 (config.json)") < err.index("Which should Cortex Training use? [1]")
    assert config() == {"poll_timeout": 60, "profile": "cortex-training"}


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
    real_write = _convert._write_atomic

    def fail_on_backup(target, data):
        if target.name.endswith(".bak"):
            raise OSError(28, "No space left on device")
        real_write(target, data)

    monkeypatch.setattr(_convert, "_write_atomic", fail_on_backup)
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

    monkeypatch.setattr(type(path), "unlink", fail_on_login_file)
    code, _, err = convert(World(), path, YES)
    monkeypatch.undo()
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
    code = _cli.main(["login", "train", "--convert-old-config", "ct.json"], stdout=io.StringIO(), stderr=stderr)
    assert code == 2
    assert "takes no profile name" in stderr.getvalue()

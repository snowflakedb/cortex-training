from __future__ import annotations

import io
import json
import textwrap

import pytest

from cortex_training import _cli
from cortex_training import _database_setup
from cortex_training import _login
from cortex_training import _picker
from cortex_training import _profiles


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Empty Snowflake folder and config home for every test; never the real ones."""
    snowflake = tmp_path / "snowflake"
    snowflake.mkdir()
    monkeypatch.setenv("SNOWFLAKE_HOME", str(snowflake))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT"):
        monkeypatch.delenv(name, raising=False)
    # The fake client answers the database checks itself, instead of running SQL.
    monkeypatch.setattr(_database_setup, "missing", lambda client: client.missing())
    monkeypatch.setattr(_database_setup, "is_listed", lambda client, kind, name: client.is_listed(kind, name))
    monkeypatch.setattr(_database_setup, "create", lambda client, objects: client.create(objects))
    return snowflake


def write_profiles(snowflake, text):
    (snowflake / "connections.toml").write_text(textwrap.dedent(text))


def config():
    path = _picker.config_path()
    return json.loads(path.read_text()) if path.exists() else None


class FakeClient:
    """Stands in for a signed-in profile client."""

    def __init__(self, world, profile, database, schema):
        self.world, self.profile, self.database, self.schema = world, profile, database, schema
        self.closed = False

    def _current_role(self):
        return self.world.role

    def missing(self):
        if self.world.check_fails:
            return None
        return [] if self.database in self.world.databases else [("database", self.database)]

    def is_listed(self, kind, name):
        return name in self.world.databases

    def create(self, objects):
        if self.world.create_error:
            raise _database_setup.DatabaseSetupError(self.world.create_error)
        self.world.created.extend(objects)
        self.world.databases.update(name for _, name in objects)

    def get_capacity(self):
        # Answers like the Cortex Training API: 403/517600 when it isn't enabled on
        # the account, 400/517602 when the database in the URL can't be resolved.
        self.world.capacity_calls.append(self.profile)
        if self.world.service_error:
            raise RuntimeError(self.world.service_error)
        if self.profile in self.world.not_enabled:
            raise ApiError(403, "517600", "Cortex Training API is not enabled for this account")
        if self.database not in self.world.databases:
            raise ApiError(400, "517602", f"Schema {self.database}.PUBLIC is not found or not authorized")
        return {"capacity_by_hardware": {}}

    def close(self):
        self.closed = True


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(f"{status} Client Error: {message}")
        self.response = type("Response", (), {"status_code": status, "json": lambda _self: {"code": code, "message": message}})()


class World:
    def __init__(self, databases=(), role="TRAINER"):
        self.databases = set(databases)
        self.role = role
        self.check_fails = False
        self.service_error = None
        self.not_enabled = set()
        self.capacity_calls = []
        self.sign_in_error = {}
        self.created = []
        self.create_error = None
        self.opened = []

    def open(self, profile, database, schema):
        if profile in self.sign_in_error:
            raise RuntimeError(self.sign_in_error[profile])
        client = FakeClient(self, profile, database, schema)
        self.opened.append(client)
        return client


def status(err, name):
    """The status shown next to profile ``name`` in the list (the last list shown)."""
    found = None
    for line in err.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0].isdigit() and parts[1] == name:
            rest = parts[3:] if parts[2] == "(current)" else parts[2:]
            found = " ".join(rest)
    return found


def login(world, answers=(), name=None):
    stdin, stdout, stderr = io.StringIO("".join(f"{a}\n" for a in answers)), io.StringIO(), io.StringIO()
    code = _login.run_login(name, stdin=stdin, stdout=stdout, stderr=stderr, interactive=True, open_client=world.open)
    summary = json.loads(stdout.getvalue()) if code == 0 else None
    return code, summary, stderr.getvalue()


PROFILES = """\
    [prod]
    account = "ACME-PROD"
    user = "ME"
    role = "TRAINER"
    authenticator = "SNOWFLAKE_JWT"
    private_key_file = "/keys/me.p8"
    database = "ANALYTICS_DB"

    [train]
    account = "ACME-PROD"
    user = "ME"
    role = "TRAINER"
    authenticator = "SNOWFLAKE_JWT"
    private_key_file = "/keys/me.p8"
    database = "TRAIN_DB"

    [other]
    account = "ACME-OTHER"
    user = "SOMEONE"
    database = "OTHER_DB"
"""


# ─── New users ──────────────────────────────────────────────────────────────


def test_a_new_user_creates_a_pat_profile(isolated):
    world = World(databases={"CORTEX_TRAINING_DB"})
    code, summary, err = login(
        world,
        ["acme-main.snowflakecomputing.com", "ME", "", "my-secret-token", "", "", ""],
    )
    assert code == 0
    assert _profiles.read_profiles() == {
        "cortex-training": {
            "account": "acme-main",
            "host": "acme-main.snowflakecomputing.com",
            "user": "ME",
            "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
            "token": "my-secret-token",
            "database": "CORTEX_TRAINING_DB",
            "schema": "PUBLIC",
        }
    }
    assert config() == {"profile": "cortex-training"}
    assert summary["profile"] == "cortex-training"
    assert summary["created_profiles"] == ["cortex-training"]
    assert "my-secret-token" not in err and "my-secret-token" not in json.dumps(summary)
    assert all(client.closed for client in world.opened)


def test_a_new_user_can_choose_the_profile_and_database_names(isolated):
    world = World()
    code, summary, err = login(
        world,
        ["https://app.snowflake.com/acme/main", "ME", "TRAINER", "tok", "mine", "MY_DB", "", "y"],
    )
    assert code == 0
    profile = _profiles.read_profiles()["mine"]
    assert (profile["host"], profile["role"], profile["database"]) == ("acme-main.snowflakecomputing.com", "TRAINER", "MY_DB")
    assert "CREATE DATABASE IF NOT EXISTS MY_DB;" in err
    assert world.created == [("database", "MY_DB")]
    assert summary["created_database"] is True


def test_a_create_that_fails_explains_why_without_a_traceback(isolated):
    world = World()
    world.create_error = "Role TRAINER can't find database MY_DB and can't create it."
    code, _, err = login(
        world,
        ["https://app.snowflake.com/acme/main", "ME", "TRAINER", "tok", "mine", "MY_DB", "", "y"],
    )
    assert code == 1
    assert "Role TRAINER can't find database MY_DB and can't create it." in err
    assert "Traceback" not in err
    assert world.created == []
    assert config() is None


def test_without_a_terminal_login_explains_the_ci_route(isolated):
    stderr = io.StringIO()
    code = _login.run_login(None, stdin=io.StringIO(), stdout=io.StringIO(), stderr=stderr, interactive=False)
    assert code == 2
    assert "CORTEX_TRAINING_HOST and CORTEX_TRAINING_PAT" in stderr.getvalue()


# ─── Existing setups ────────────────────────────────────────────────────────


def test_a_working_saved_profile_is_kept_by_default(isolated):
    write_profiles(isolated, PROFILES)
    _picker.save_profile("train")
    world = World(databases={"TRAIN_DB"})
    code, summary, err = login(world, [""])
    assert code == 0
    assert "Cortex Training is set up to use profile train" in err and "It works. Keep using train?" in err
    assert summary["profile"] == "train" and summary["created_profiles"] == []
    assert config() == {"profile": "train"}


def test_declining_the_saved_profile_lets_the_user_pick_another(isolated):
    write_profiles(isolated, PROFILES)
    _picker.save_profile("train")
    world = World(databases={"TRAIN_DB", "ANALYTICS_DB"})
    # Keep? no. Pick 2 (prod, sorted: other, prod, train). Use its database.
    code, summary, _ = login(world, ["n", "n", "2", "1"])
    assert code == 0
    assert config() == {"profile": "prod"}


def test_a_saved_profile_that_no_longer_exists_is_reported(isolated):
    write_profiles(isolated, PROFILES)
    _picker.save_profile("gone")
    world = World(databases={"TRAIN_DB"})
    code, _, err = login(world, ["n", "3", "1"])
    assert code == 0
    assert "set up to use profile gone, which doesn't exist any more" in err
    assert config() == {"profile": "train"}


def test_other_settings_in_config_json_are_kept(isolated):
    write_profiles(isolated, PROFILES)
    path = _picker.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"poll_interval": 2}))
    code, _, _ = login(World(databases={"TRAIN_DB"}), ["n", "3", "1"])
    assert code == 0
    assert config() == {"poll_interval": 2, "profile": "train"}


# ─── Picking a profile and its database ─────────────────────────────────────


def test_picking_a_profile_says_which_database_is_used(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"ANALYTICS_DB"})
    code, summary, err = login(world, ["n", "2", "1"])
    assert code == 0
    assert "Profile prod uses database ANALYTICS_DB." in err
    assert (summary["profile"], summary["database"]) == ("prod", "ANALYTICS_DB")


def test_no_database_choice_when_the_profile_already_uses_the_default(isolated):
    _profiles.write_profile("ct", {"account": "ACME", "user": "ME", "database": "cortex_training_db"})
    code, summary, err = login(World(databases={"cortex_training_db"}), ["1"])
    assert code == 0
    assert "Which database" not in err
    assert summary["profile"] == "ct"


def test_a_separate_database_copies_the_sign_in_into_a_new_profile(isolated):
    write_profiles(isolated, PROFILES)
    before = dict(_profiles.read_profiles()["prod"])
    world = World(databases={"ANALYTICS_DB"})
    # prod -> separate database -> name [prod-ct] -> database [CORTEX_TRAINING_DB] -> check [Y] -> run CREATE
    code, summary, err = login(world, ["n", "2", "2", "", "", "", "y"])
    assert code == 0
    profiles = _profiles.read_profiles()
    assert profiles["prod"] == before
    assert profiles["prod-ct"] == {**{k: v for k, v in before.items() if k != "database"}, "database": "CORTEX_TRAINING_DB", "schema": "PUBLIC"}
    assert world.created == [("database", "CORTEX_TRAINING_DB")]
    assert config() == {"profile": "prod-ct"}


def test_the_default_choice_is_the_snowflake_default_profile(isolated, monkeypatch):
    write_profiles(isolated, PROFILES)
    monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "train")
    code, summary, err = login(World(databases={"TRAIN_DB"}), ["n", "", "1"])
    assert code == 0
    assert "Which profile should Cortex Training use? [3]" in err
    assert summary["profile"] == "train"


def test_login_name_uses_an_existing_profile(isolated):
    write_profiles(isolated, PROFILES)
    code, summary, err = login(World(databases={"TRAIN_DB"}), ["1"], name="train")
    assert code == 0
    assert "Your Snowflake connection profiles" not in err
    assert summary["profile"] == "train"


def test_login_name_creates_that_profile_when_it_does_not_exist(isolated):
    write_profiles(isolated, PROFILES)
    code, summary, _ = login(World(databases={"CORTEX_TRAINING_DB"}), ["acme-new.snowflakecomputing.com", "ME", "", "tok", "", ""], name="fresh")
    assert code == 0
    assert _profiles.read_profiles()["fresh"]["host"] == "acme-new.snowflakecomputing.com"
    assert summary["profile"] == "fresh"


# Answers for `login fresh`: host, user, role, PAT, database.
NEW_PROFILE = ["acme-new.snowflakecomputing.com", "ME", "", "tok", ""]


def test_a_new_profile_is_checked_only_after_asking(isolated):
    world = World(databases={"CORTEX_TRAINING_DB"})
    code, summary, err = login(world, [*NEW_PROFILE, "n"], name="fresh")
    assert code == 0
    assert "Check it now? [Y/n]" in err
    assert world.opened == [] and world.capacity_calls == []
    assert "Saved, but not checked. Cortex Training will use profile fresh." in err
    assert "You're all set" not in err
    assert config() == {"profile": "fresh"}
    assert summary["profile"] == "fresh" and summary["checked"] is False


def test_a_new_profile_that_cannot_sign_in_can_be_deleted(isolated):
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["fresh"] = "Incorrect username or password was specified."
    code, _, err = login(world, [*NEW_PROFILE, "", ""], name="fresh")
    assert code == 1
    assert "Please check the account, user, and PAT you entered." in err
    assert "Delete profile fresh for now? [Y/n]" in err
    assert "fresh" not in _profiles.read_profiles()
    assert "Saved profile(s)" not in err
    assert config() is None


def test_a_new_profile_that_cannot_sign_in_is_kept_if_the_user_says_no(isolated):
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["fresh"] = "Incorrect username or password was specified."
    code, _, err = login(world, [*NEW_PROFILE, "", "n"], name="fresh")
    assert code == 1
    assert "fresh" in _profiles.read_profiles()
    assert "Saved profile(s) fresh; run cortex-training login again to finish." in err
    assert config() is None


def test_a_sign_in_error_is_shown_in_full(isolated):
    # Snowflake's reason, and the next steps after it, come after the first 160 characters.
    reason = (
        "could not open Snowflake connection 'fresh': 394400 (08001): Failed to connect to DB: "
        "acme-new.snowflakecomputing.com:443. Programmatic access token is invalid.\n"
        "\n"
        "Snowflake rejected your programmatic access token (PAT). The most common causes:"
    )
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["fresh"] = reason
    code, _, err = login(world, [*NEW_PROFILE, "", ""], name="fresh")
    assert code == 1
    assert "Profile fresh couldn't sign in:" in err
    assert "Programmatic access token is invalid." in err
    assert "Snowflake rejected your programmatic access token (PAT). The most common causes:" in err


def test_a_profile_that_just_failed_is_not_used_to_check_its_account(isolated):
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["cortex-training"] = "Programmatic access token is invalid."
    # No profiles -> create one -> check [Y] -> it fails -> keep it (n) -> back to the list.
    code, _, err = login(world, ["acme-new.snowflakecomputing.com", "ME", "", "tok", "", "", "", "n"])
    assert code != 0
    assert "Profile cortex-training can't be used." in err
    assert "Which of your accounts have Cortex Training?" not in err
    assert [client.profile for client in world.opened] == []


def test_a_new_profile_on_an_account_without_cortex_training_can_be_deleted(isolated):
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.not_enabled.add("fresh")
    code, _, err = login(world, [*NEW_PROFILE, "", ""], name="fresh")
    assert code == 1
    assert "Cortex Training isn't enabled on account acme-new" in err
    # It signed in, so the typed sign-in details aren't the problem.
    assert "Please check the account, user, and PAT you entered." not in err
    assert "fresh" not in _profiles.read_profiles()


def test_an_existing_profile_is_never_offered_for_deletion(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"TRAIN_DB"})
    world.sign_in_error["train"] = "Incorrect username or password was specified."
    code, _, err = login(world, name="train")
    assert code == 1
    assert "Delete profile" not in err
    assert "train" in _profiles.read_profiles()


def test_login_rejects_an_invalid_profile_name(isolated):
    code, _, err = login(World(), name="my profile")
    assert code == 1 and "isn't a valid profile name" in err


def test_a_new_profile_name_must_be_unused(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"CORTEX_TRAINING_DB"})
    code, summary, err = login(world, ["n", "2", "2", "train", "prod-ct", "", ""])
    assert code == 0
    assert "Profile train already exists; choose another name." in err
    assert summary["profile"] == "prod-ct"


# ─── An unusable database ───────────────────────────────────────────────────


def test_a_profile_with_the_same_identity_and_a_working_database_is_offered(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"TRAIN_DB"})
    # prod -> use ANALYTICS_DB -> it's missing -> use train instead -> train's database
    code, summary, err = login(world, ["n", "2", "1", "y", "1"])
    assert code == 0
    assert "Database ANALYTICS_DB is missing, or role TRAINER cannot access it" in err
    assert "Profile train signs in as the same user and role on the same account, and its database TRAIN_DB works" in err
    assert summary["profile"] == "train"


def test_without_a_matching_profile_a_new_profile_is_offered(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases=set())
    # other -> use OTHER_DB -> missing; no same-identity profile -> create new? yes -> names -> CREATE? yes
    code, summary, err = login(world, ["n", "1", "1", "y", "", "", "", "y"])
    assert code == 0
    assert "Profile train signs in" not in err
    assert summary["profile"] == "other-ct"
    assert world.created == [("database", "CORTEX_TRAINING_DB")]


def test_declining_a_new_profile_changes_nothing(isolated):
    write_profiles(isolated, PROFILES)
    before = (isolated / "connections.toml").read_text()
    code, _, err = login(World(databases=set()), ["n", "1", "1", "n"])
    assert code == 1
    assert "Profile other has no database Cortex Training can use, so nothing was saved" in err
    assert (isolated / "connections.toml").read_text() == before
    assert config() is None


def test_declining_create_database_keeps_the_new_profile_and_says_so(isolated):
    write_profiles(isolated, PROFILES)
    code, _, err = login(World(databases=set()), ["n", "1", "1", "y", "", "", "", "n"])
    assert code == 1
    assert "To create it yourself, run in Snowsight: CREATE DATABASE IF NOT EXISTS CORTEX_TRAINING_DB;" in err
    assert "Saved profile(s) other-ct; run cortex-training login again to finish." in err
    assert "other-ct" in _profiles.read_profiles()
    assert config() is None


def test_a_database_check_that_cannot_run_continues_with_a_note(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"TRAIN_DB"})
    world.check_fails = True
    code, summary, err = login(world, ["n", "3", "1"])
    assert code == 0
    assert "Could not check database TRAIN_DB; continuing" in err
    assert summary["profile"] == "train"


# ─── Failures ───────────────────────────────────────────────────────────────


def test_a_sign_in_failure_returns_to_the_list(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"ANALYTICS_DB"})
    world.sign_in_error["train"] = "Network policy is required."
    # train fails -> back to the list -> prod -> use its database
    code, summary, err = login(world, ["n", "3", "2", "1"])
    assert code == 0
    assert "Profile train couldn't sign in: Network policy is required." in err
    assert "can't be used. Pick another one, or create a new one." in err
    assert status(err, "train") == "✗ Didn't work just now: Profile train couldn't sign in: Network policy is required."
    assert summary["profile"] == "prod"


def test_a_failed_profile_is_no_longer_the_default(isolated, monkeypatch):
    write_profiles(isolated, PROFILES)
    monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "train")
    world = World(databases={"ANALYTICS_DB"})
    world.sign_in_error["train"] = "denied"
    code, summary, err = login(world, ["n", "", "", "1"])
    assert code == 0
    assert "Which profile should Cortex Training use? [3]" in err
    assert "Which profile should Cortex Training use? [2]" in err
    assert summary["profile"] == "prod"


def test_a_service_failure_returns_to_the_list(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases={"TRAIN_DB"})
    world.service_error = "Cortex Training is not enabled"
    code, _, err = login(world, ["n", "3", "1"])
    assert code == 1  # then input runs out
    assert "Profile train signs in, but Cortex Training didn't answer: Cortex Training is not enabled" in err
    assert "can't be used. Pick another one, or create a new one." in err
    assert config() is None


def test_login_name_does_not_retry_another_profile(isolated):
    write_profiles(isolated, PROFILES)
    world = World()
    world.sign_in_error["train"] = "denied"
    code, _, err = login(world, [], name="train")
    assert code == 1
    assert "Profile train couldn't sign in: denied" in err
    assert "Your Snowflake connection profiles" not in err


def test_running_out_of_input_cancels_cleanly(isolated):
    write_profiles(isolated, PROFILES)
    code, _, err = login(World(databases={"TRAIN_DB"}), [])
    assert code == 1 and "no input" in err


def test_ctrl_c_cancels_and_closes_clients(isolated, monkeypatch):
    write_profiles(isolated, PROFILES)
    world = World(databases={"TRAIN_DB"})

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_login._Login, "verify_service", interrupt)
    code, _, err = login(world, ["n", "3", "1"])
    assert code == 130 and "login cancelled." in err
    assert world.opened and all(client.closed for client in world.opened)


# ─── Same identity ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("candidate", "matches"),
    [
        ({"account": "acme-prod", "user": "me", "role": "trainer", "database": "X"}, True),
        ({"account": "ACME-PROD", "user": "ME", "database": "X"}, True),  # no role: default role, same as base's session
        ({"account": "ACME-PROD", "user": "ME", "role": "ADMIN", "database": "X"}, False),
        ({"account": "ACME-PROD", "user": "YOU", "role": "TRAINER", "database": "X"}, False),
        ({"account": "ACME-OTHER", "user": "ME", "role": "TRAINER", "database": "X"}, False),
        ({"account": "ACME-PROD", "user": "ME", "role": "TRAINER"}, False),  # no database
    ],
)
def test_same_identity_rules(isolated, candidate, matches):
    _profiles.write_profile("base", {"account": "ACME-PROD", "user": "ME", "role": "TRAINER", "database": "A"})
    _profiles.write_profile("candidate", candidate)
    flow = _login._Login(_login._Prompter(io.StringIO(), io.StringIO()), World().open)
    assert (flow.same_identity("base", "TRAINER") == ["candidate"]) is matches


# ─── Hosts ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "host"),
    [
        ("acme-main.snowflakecomputing.com", "acme-main.snowflakecomputing.com"),
        ("https://acme-main.snowflakecomputing.com/", "acme-main.snowflakecomputing.com"),
        ("https://app.snowflake.com/acme/main", "acme-main.snowflakecomputing.com"),
        ("https://app.snowflake.com/us-west-2/xy12345#/homepage", "xy12345.us-west-2.snowflakecomputing.com"),
        ("my_acct.region.snowflakecomputing.com", "my_acct.region.snowflakecomputing.com"),
    ],
)
def test_validate_host(raw, host):
    assert _login.validate_host(raw) == host


@pytest.mark.parametrize("raw", ["app.snowflake.com/acme", "not a host", "localhost"])
def test_validate_host_rejects(raw):
    with pytest.raises(ValueError):
        _login.validate_host(raw)


# ─── CLI ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [["login", "config.json"], ["login", "~/ct/config.json"], ["login", "--config", "ct.json"]])
def test_the_old_login_forms_explain_what_replaced_them(argv, isolated):
    stderr = io.StringIO()
    code = _cli.main(argv, stdout=io.StringIO(), stderr=stderr, stdin=io.StringIO())
    assert code == 2
    assert "is no longer supported. Run `cortex-training login`" in stderr.getvalue()
    assert not (_picker.config_path().parent / "login.json").exists()


def test_the_cli_runs_the_new_login(monkeypatch, isolated):
    calls = []
    monkeypatch.setattr(_login, "run_login", lambda name, **kw: calls.append(name) or 0)
    assert _cli.main(["login", "training"], stdout=io.StringIO(), stderr=io.StringIO(), stdin=io.StringIO()) == 0
    assert _cli.main(["login"], stdout=io.StringIO(), stderr=io.StringIO(), stdin=io.StringIO()) == 0
    assert calls == ["training", None]


# ─── Profile hints ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("values", "verdict", "hint"),
    [
        ({"account": "A", "user": "U", "authenticator": "PROGRAMMATIC_ACCESS_TOKEN", "token": "t"}, "Signs in without prompts", "a programmatic access token (no prompts)"),
        ({"account": "A", "user": "U", "authenticator": "SNOWFLAKE_JWT"}, "Signs in without prompts", "a key file (no prompts)"),
        ({"account": "A", "user": "U", "private_key_file": "/k.p8"}, "Signs in without prompts", "a key file"),
        ({"account": "A", "user": "U", "authenticator": "externalbrowser"}, "Signs in with a prompt", "opens a browser window"),
        ({"account": "A", "user": "U", "authenticator": "username_password_mfa", "password": "p"}, "Signs in with a prompt", "approve each sign-in"),
        ({"account": "A", "user": "U", "password": "p"}, "Sign-in unclear", "saved by `snow connection add`"),
        ({"account": "A", "user": "U", "token": "t"}, "Likely won't sign in", "Snowflake will ignore it"),
        ({"account": "A", "user": "U", "authenticator": "oauth"}, "Sign-in unclear", "until it expires"),
    ],
)
def test_describe_profile_sign_in(values, verdict, hint):
    described = _login.describe_profile(values)
    assert described.verdict == verdict
    assert hint in described.sign_in


def test_describe_profile_summary():
    assert _login.describe_profile({"account": "A", "user": "U", "database": "cortex_training_db", "role": "R"}).summary == (
        "account A · user U · role R · database cortex_training_db (recommended)"
    )
    assert _login.describe_profile({"account": "A", "user": "U", "database": "WORK"}).summary == (
        "account A · user U · your default role · database WORK"
    )
    assert _login.describe_profile({"account": "A", "user": "U"}).summary.endswith("no database set")


def test_a_profile_without_account_or_user_cannot_be_picked(isolated):
    write_profiles(isolated, """\
        [broken]
        password = "p"

        [good]
        account = "ACME"
        user = "ME"
        authenticator = "SNOWFLAKE_JWT"
        database = "CORTEX_TRAINING_DB"
    """)
    described = _login.describe_profile({"password": "p"})
    assert (described.verdict, described.usable, described.summary) == ("Can't be used", False, "no account or user set")
    code, summary, err = login(World(databases={"CORTEX_TRAINING_DB"}), ["n", "1", "2"])
    assert code == 0
    assert "Profile broken can't be used: no account or user set. Pick another." in err
    assert summary["profile"] == "good"


def test_the_list_explains_what_makes_a_good_profile(isolated):
    write_profiles(isolated, PROFILES)
    _, _, err = login(World(databases={"TRAIN_DB"}), ["n", "3", "1"])
    assert "Cortex Training needs three things:" in err
    assert "Let's see what you already have." in err
    assert status(err, "train") == "? Cortex Training not checked (you skipped the check); checked when you pick it"
    assert "account ACME-PROD · user ME · role TRAINER · database TRAIN_DB" in err
    assert "signs in with a key file (no prompts)" in err


# ─── Cortex Training is checked first ───────────────────────────────────────


def test_the_list_says_to_pick_an_account_with_cortex_training(isolated):
    write_profiles(isolated, PROFILES)
    _, _, err = login(World(databases={"TRAIN_DB"}), ["n", "3", "1"])
    assert "MAKE SURE YOU PICK A PROFILE ON AN ACCOUNT WHERE CORTEX TRAINING IS ENABLED." in err


def test_an_account_without_cortex_training_returns_to_the_list_before_anything_is_created(isolated):
    write_profiles(isolated, PROFILES)
    before = (isolated / "connections.toml").read_text()
    world = World(databases={"TRAIN_DB"})
    world.not_enabled = {"other"}
    # other: CT not enabled, and its database is missing too -> back to the list -> train
    code, summary, err = login(world, ["n", "1", "3", "1"])
    assert code == 0
    assert "Cortex Training isn't enabled on account ACME-OTHER" in err
    assert "Create a new profile with the same sign-in" not in err
    assert "CREATE DATABASE" not in err
    assert (isolated / "connections.toml").read_text() == before
    assert summary["profile"] == "train"


def test_a_missing_database_still_means_cortex_training_answered(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases=set())
    code, summary, err = login(world, ["n", "1", "1", "y", "", "", "", "y"])
    assert code == 0
    assert "isn't enabled" not in err
    assert world.created == [("database", "CORTEX_TRAINING_DB")]
    assert world.capacity_calls[0] == "other"  # asked before the database steps


def test_a_database_that_is_still_missing_at_the_end_is_named(isolated):
    write_profiles(isolated, PROFILES)
    world = World(databases=set())
    world.check_fails = True
    code, _, err = login(world, ["n", "3", "1"])
    assert code == 1  # back to the list, then input runs out
    assert "Cortex Training can't find database TRAIN_DB or schema PUBLIC with profile train" in err


# ─── Checking which accounts have Cortex Training ───────────────────────────

ACCOUNTS = """\
    [alpha]
    account = "ACME-ALPHA"
    user = "ME"
    authenticator = "PROGRAMMATIC_ACCESS_TOKEN"
    token = "t"
    database = "CORTEX_TRAINING_DB"

    [alpha2]
    account = "acme-alpha"
    user = "ME"
    authenticator = "SNOWFLAKE_JWT"
    database = "CORTEX_TRAINING_DB"

    [beta]
    account = "ACME-BETA"
    user = "ME"
    authenticator = "SNOWFLAKE_JWT"
    database = "CORTEX_TRAINING_DB"

    [desk]
    account = "ACME-DESK"
    user = "ME"
    authenticator = "externalbrowser"
"""


def test_the_check_is_asked_first_and_lists_what_it_will_contact(isolated):
    write_profiles(isolated, ACCOUNTS)
    _, _, err = login(World(databases={"CORTEX_TRAINING_DB"}), ["n", "1"])
    assert "can sign in once to each account below to find out. This only reads; nothing is changed." in err
    assert "  - ACME-ALPHA  (using profiles alpha, alpha2)" in err
    assert "  - ACME-BETA  (using profile beta)" in err
    assert "Not checked now: desk." in err
    assert "Check these accounts now? [Y/n]" in err


def test_declining_the_check_signs_in_to_nothing_before_the_pick(isolated):
    write_profiles(isolated, ACCOUNTS)
    world = World(databases={"CORTEX_TRAINING_DB"})
    login(world, ["n", "3"])
    assert [client.profile for client in world.opened] == ["beta"]  # only the profile picked


def test_checking_shows_each_accounts_status_and_signs_in_once_per_account(isolated):
    write_profiles(isolated, ACCOUNTS)
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.not_enabled = {"beta"}
    code, summary, err = login(world, ["y", "1"])
    assert code == 0
    assert status(err, "alpha") == "✓ Cortex Training is enabled on ACME-ALPHA"
    assert status(err, "alpha2") == "✓ Cortex Training is enabled on ACME-ALPHA"
    assert status(err, "beta") == "✗ Cortex Training is not enabled on ACME-BETA"
    assert status(err, "desk") == "? Cortex Training not checked yet; checked when you pick it"
    probed = [client.profile for client in world.opened[:2]]
    assert sorted(probed) == ["alpha", "beta"]  # one per account; desk never signed in
    assert all(client.closed for client in world.opened)
    assert summary["profile"] == "alpha"


def test_a_profile_on_an_account_without_cortex_training_cannot_be_picked(isolated):
    write_profiles(isolated, ACCOUNTS)
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.not_enabled = {"beta"}
    code, summary, err = login(world, ["y", "3", "1"])
    assert code == 0
    assert "Profile beta can't be used: Cortex Training isn't enabled on account ACME-BETA. Pick another." in err
    assert summary["profile"] == "alpha"


def test_the_default_choice_is_a_profile_on_an_enabled_account(isolated, monkeypatch):
    write_profiles(isolated, ACCOUNTS)
    monkeypatch.setenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "beta")
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.not_enabled = {"beta"}
    _, summary, err = login(world, ["y", ""])
    assert "Which profile should Cortex Training use? [1]" in err
    assert summary["profile"] == "alpha"


def test_a_check_that_fails_to_sign_in_is_shown_and_still_pickable(isolated):
    write_profiles(isolated, ACCOUNTS)
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["beta"] = "Network policy is required.\nmore detail"
    _, _, err = login(world, ["y", "1"])
    assert status(err, "beta") == "? Couldn't check Cortex Training: couldn't sign in: Network policy is required."


def test_the_check_is_asked_only_once_per_login(isolated):
    write_profiles(isolated, ACCOUNTS)
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["alpha"] = "denied"
    world.sign_in_error["alpha2"] = "denied"
    # check -> ACME-ALPHA couldn't check; pick alpha -> fails -> back to the list (no second question) -> beta
    code, summary, err = login(world, ["y", "1", "3"])
    assert code == 0
    assert err.count("Check these accounts now?") == 1
    assert summary["profile"] == "beta"


def test_no_check_is_offered_when_no_profile_signs_in_without_prompts(isolated):
    write_profiles(isolated, """\
        [desk]
        account = "ACME-DESK"
        user = "ME"
        authenticator = "externalbrowser"
        database = "CORTEX_TRAINING_DB"
    """)
    _, _, err = login(World(databases={"CORTEX_TRAINING_DB"}), ["1"])
    assert "Check these accounts now?" not in err


def test_the_check_tries_the_accounts_next_profile_when_one_cannot_sign_in(isolated):
    write_profiles(isolated, ACCOUNTS)
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error["alpha"] = "Password was not given but private key is encrypted"
    _, _, err = login(world, ["y", "2"])
    assert status(err, "alpha").startswith("? Couldn't check Cortex Training: couldn't sign in: Password was not given")
    assert status(err, "alpha2") == "✓ Cortex Training is enabled on ACME-ALPHA"


def test_when_no_profile_on_an_account_signs_in_the_others_say_so(isolated):
    write_profiles(isolated, ACCOUNTS + """\
    [alpha3]
    account = "ACME-ALPHA"
    user = "ME"
    authenticator = "externalbrowser"
    """)
    world = World(databases={"CORTEX_TRAINING_DB"})
    world.sign_in_error.update({"alpha": "denied", "alpha2": "denied"})
    _, _, err = login(world, ["y", "4"])
    assert status(err, "alpha3") == "? Couldn't check Cortex Training: no profile on this account signed in without asking for something"


def test_an_encrypted_key_without_a_passphrase_may_prompt(tmp_path):
    key = tmp_path / "key.p8"
    key.write_text("-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n")
    values = {"account": "A", "user": "U", "authenticator": "SNOWFLAKE_JWT", "private_key_file": str(key)}
    assert _login.describe_profile(values).verdict == "Sign-in unclear"
    assert "may ask for its passphrase" in _login.describe_profile(values).sign_in
    assert _login.describe_profile({**values, "private_key_file_pwd": "x"}).verdict == "Signs in without prompts"
    key.write_text("-----BEGIN PRIVATE KEY-----\nabc\n")
    assert _login.describe_profile(values).verdict == "Signs in without prompts"


# ─── Output ─────────────────────────────────────────────────────────────────


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_the_ending_says_what_to_do_next(isolated):
    write_profiles(isolated, PROFILES)
    code, _, err = login(World(databases={"TRAIN_DB"}), ["n", "3", "1"])
    assert code == 0
    assert "  ✓ Signed in to ACME-PROD as ME (role TRAINER)" in err
    assert "  ✓ Cortex Training is enabled on this account" in err
    assert "  ✓ Database TRAIN_DB is ready" in err
    assert "You're all set. Cortex Training will use profile train." in err
    assert "Try it:  cortex-training capacity" in err


def test_no_color_without_a_terminal(isolated):
    write_profiles(isolated, PROFILES)
    _, _, err = login(World(databases={"TRAIN_DB"}), ["n", "3"])
    assert "\033[" not in err


def test_color_in_a_terminal_unless_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert _login._Style(_Terminal()).ok("done") == "\033[32m✓\033[0m done"
    monkeypatch.setenv("NO_COLOR", "1")
    assert _login._Style(_Terminal()).ok("done") == "✓ done"


def test_the_json_summary_is_printed_only_when_piped(isolated):
    write_profiles(isolated, PROFILES)
    stdout = _Terminal()
    code = _login.run_login(
        None,
        stdin=io.StringIO("n\n3\n1\n"),
        stdout=stdout,
        stderr=io.StringIO(),
        interactive=True,
        open_client=World(databases={"TRAIN_DB"}).open,
    )
    assert code == 0
    assert stdout.getvalue() == ""


def test_login_reports_an_overriding_variable_first_and_reminds_at_the_end(isolated, monkeypatch):
    write_profiles(isolated, PROFILES)
    monkeypatch.setenv("CORTEX_TRAINING_CONFIG", "/tmp/trial-config.json")
    monkeypatch.setenv("SHELL", "/bin/zsh")
    code, _, err = login(World(databases={"TRAIN_DB"}), ["n", "3", "1"])
    assert code == 0
    assert err.startswith("Checking your shell for settings that could override this setup...")
    assert "✗ CORTEX_TRAINING_CONFIG = /tmp/trial-config.json" in err
    ending = err.split("You're all set.")[1]
    assert "Heads-up: your shell still overrides this setup." in ending
    assert "CORTEX_TRAINING_CONFIG  clear it with:  unset CORTEX_TRAINING_CONFIG" in ending


def test_a_terminal_gets_the_logo_and_section_rules(isolated, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    write_profiles(isolated, PROFILES)
    stderr = _Terminal()
    code = _login.run_login(
        None,
        stdin=io.StringIO("n\n3\n1\n"),
        stdout=_Terminal(),
        stderr=stderr,
        interactive=True,
        open_client=World(databases={"TRAIN_DB"}).open,
    )
    out = stderr.getvalue()
    assert code == 0
    assert out.startswith(_login._LOGO + "\n")
    rule = "─" * 20
    assert f"{rule}" in out
    assert "\n\n\n" not in out  # never two blank lines in a row
    for title in ("Checking your shell for settings", "Let's connect Cortex Training", "Checking profile train..."):
        line = next(i for i, text in enumerate(out.splitlines()) if text.startswith(title))
        assert out.splitlines()[line - 1].startswith(rule)


def test_logs_and_scripts_get_no_logo_or_rules(isolated):
    write_profiles(isolated, PROFILES)
    _, _, err = login(World(databases={"TRAIN_DB"}), ["n", "3", "1"])
    assert "T R A I N I N G" not in err
    assert "─" * 20 not in err

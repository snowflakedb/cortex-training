"""`_job_access.check`: can the profile's role create training jobs? Answered from QA6-shaped rows."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from cortex_training import _database_setup
from cortex_training import _job_access


def grant(privilege, grantee, granted_to="ROLE", on="SCHEMA"):
    # The columns SHOW GRANTS ON returns (lower-cased by run_statement), as seen on a live account.
    return {"privilege": privilege, "granted_on": on, "name": "X", "granted_to": granted_to, "grantee_name": grantee}


class Snowflake:
    """Answers the statements the check runs; records them."""

    def __init__(self, schema=(), account=(), session=("TRAINER",), database_roles=(), secondary=None):
        self.schema, self.account = list(schema), list(account)
        self.session, self.database_roles = set(session), set(database_roles)
        self.secondary = secondary or {"roles": "", "value": ""}
        self.statements = []
        self.fail = {}

    def run(self, client, statement):
        self.statements.append(statement)
        for pattern, error in self.fail.items():
            if pattern in statement:
                raise RuntimeError(error)
        if statement.startswith("SELECT CURRENT_SECONDARY_ROLES()"):
            return [{"roles": json.dumps(self.secondary)}]
        if statement.startswith("SELECT CURRENT_ROLE()"):
            return [{"role": "TRAINER"}]
        if statement.startswith("USE SECONDARY ROLES"):
            return []
        if statement.startswith("SHOW GRANTS ON SCHEMA"):
            return self.schema
        if statement == "SHOW GRANTS ON ACCOUNT":
            return self.account
        if statement.startswith("SELECT IS_"):
            roles = re.findall(r"IS_ROLE_IN_SESSION\('([^']*)'\)", statement)
            database_roles = re.findall(r"IS_DATABASE_ROLE_IN_SESSION\('([^']*)'\)", statement)
            held = any(role in self.session for role in roles) or any(role in self.database_roles for role in database_roles)
            return [{"held": held}]
        raise AssertionError(f"unexpected statement: {statement}")


@pytest.fixture
def snowflake(monkeypatch):
    world = Snowflake()
    monkeypatch.setattr(_database_setup, "run_statement", world.run)
    return world


CLIENT = SimpleNamespace(database="TRAIN_DB", schema="RUNS")
ENABLED = [grant("USE CORTEX TRAINING", "ACCOUNTADMIN", on="ACCOUNT")]


def test_a_role_with_both_privileges_is_ready(snowflake):
    snowflake.schema = [grant("USAGE", "TRAINER"), grant("CREATE EXPERIMENT", "TRAINER")]
    snowflake.account = [*ENABLED, grant("USE CORTEX TRAINING", "TRAINER", on="ACCOUNT")]
    access = _job_access.check(CLIENT, "TRAINER")
    assert access.ready and access.role == "TRAINER"
    assert "SHOW GRANTS ON SCHEMA TRAIN_DB.RUNS" in snowflake.statements


def test_missing_privileges_come_with_the_grants_to_ask_for(snowflake):
    snowflake.schema = [grant("USAGE", "TRAINER"), grant("CREATE EXPERIMENT", "OTHER_ROLE")]
    snowflake.account = ENABLED
    access = _job_access.check(CLIENT, "TRAINER")
    assert not access.ready and access.unknown is None
    assert [(m.privilege, m.on, m.grant) for m in access.missing] == [
        ("CREATE EXPERIMENT", "schema TRAIN_DB.RUNS", "GRANT CREATE EXPERIMENT ON SCHEMA TRAIN_DB.RUNS TO ROLE TRAINER;"),
        ("USE CORTEX TRAINING", "the account", "GRANT USE CORTEX TRAINING ON ACCOUNT TO ROLE TRAINER;"),
    ]


def test_an_owner_has_every_privilege_on_its_schema(snowflake):
    snowflake.schema = [grant("OWNERSHIP", "TRAINER")]
    snowflake.account = [*ENABLED, grant("USE CORTEX TRAINING", "PUBLIC", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready


def test_a_privilege_held_through_an_inherited_role_counts(snowflake):
    # TRAINER inherits DATA_TEAM, which holds both; IS_ROLE_IN_SESSION says so.
    snowflake.session = {"TRAINER", "DATA_TEAM"}
    snowflake.schema = [grant("CREATE EXPERIMENT", "DATA_TEAM")]
    snowflake.account = [*ENABLED, grant("USE CORTEX TRAINING", "DATA_TEAM", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready


def test_a_grant_to_public_counts_without_asking(snowflake):
    snowflake.schema = [grant("CREATE EXPERIMENT", "PUBLIC")]
    snowflake.account = [*ENABLED, grant("USE CORTEX TRAINING", "PUBLIC", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready
    assert not any(s.startswith("SELECT IS_") for s in snowflake.statements)


def test_a_database_role_holder_is_checked_as_a_database_role(snowflake):
    snowflake.database_roles = {"TRAIN_DB.EXPERIMENTERS"}
    snowflake.schema = [grant("CREATE EXPERIMENT", "TRAIN_DB.EXPERIMENTERS", granted_to="DATABASE_ROLE")]
    snowflake.account = [*ENABLED, grant("USE CORTEX TRAINING", "PUBLIC", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready
    assert any("IS_DATABASE_ROLE_IN_SESSION('TRAIN_DB.EXPERIMENTERS')" in s for s in snowflake.statements)


def test_an_account_without_the_privilege_yet_doesnt_warn_about_it(snowflake):
    # Before the privilege reaches an account, no role (not even ACCOUNTADMIN) holds it.
    snowflake.schema = [grant("CREATE EXPERIMENT", "TRAINER")]
    snowflake.account = [grant("CREATE DATABASE", "SYSADMIN", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready


def test_secondary_roles_are_turned_off_for_the_check_and_back_on(snowflake):
    # Creating a job counts only the primary role; a secondary role mustn't make it look ready.
    snowflake.secondary = {"roles": "ANALYST", "value": "ALL"}
    snowflake.schema = [grant("CREATE EXPERIMENT", "TRAINER")]
    snowflake.account = [*ENABLED, grant("USE CORTEX TRAINING", "TRAINER", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready
    uses = [s for s in snowflake.statements if s.startswith("USE SECONDARY ROLES")]
    assert uses == ["USE SECONDARY ROLES NONE", "USE SECONDARY ROLES ALL"]
    assert snowflake.statements.index("USE SECONDARY ROLES NONE") < snowflake.statements.index("SHOW GRANTS ON SCHEMA TRAIN_DB.RUNS")


def test_a_pat_session_changes_nothing_about_roles(snowflake):
    snowflake.schema = [grant("CREATE EXPERIMENT", "TRAINER")]
    snowflake.account = ENABLED + [grant("USE CORTEX TRAINING", "TRAINER", on="ACCOUNT")]
    _job_access.check(CLIENT, "TRAINER")
    assert not any(s.startswith("USE ") for s in snowflake.statements)


@pytest.mark.parametrize("failing", ["USE SECONDARY ROLES NONE", "SHOW GRANTS ON SCHEMA", "SHOW GRANTS ON ACCOUNT", "SELECT IS_"])
def test_anything_unexpected_is_couldnt_check_never_a_false_warning(snowflake, failing):
    snowflake.secondary = {"roles": "ANALYST", "value": ""}
    snowflake.schema = [grant("CREATE EXPERIMENT", "OTHER")]
    snowflake.account = ENABLED
    snowflake.fail = {failing: "003107 (42501): SQL execution error: denied"}
    access = _job_access.check(CLIENT, "TRAINER")
    assert access.unknown == "003107 (42501): SQL execution error: denied"
    assert access.missing == () and not access.ready


def test_the_role_comes_from_the_session_when_not_known(snowflake):
    snowflake.schema = [grant("CREATE EXPERIMENT", "TRAINER")]
    snowflake.account = ENABLED + [grant("USE CORTEX TRAINING", "TRAINER", on="ACCOUNT")]
    assert _job_access.check(CLIENT, None).role == "TRAINER"


def test_a_role_name_that_needs_quotes_is_quoted_in_the_grant(snowflake):
    snowflake.session = {"Data Team"}
    snowflake.schema = []
    snowflake.account = ENABLED
    access = _job_access.check(CLIENT, "Data Team")
    assert access.missing[0].grant == 'GRANT CREATE EXPERIMENT ON SCHEMA TRAIN_DB.RUNS TO ROLE "Data Team";'
    # And a quote inside a holder's name can't break out of the IS_ROLE_IN_SESSION literal.
    snowflake.schema = [grant("CREATE EXPERIMENT", "it's")]
    _job_access.check(CLIENT, "TRAINER")
    assert any("IS_ROLE_IN_SESSION('it''s')" in s for s in snowflake.statements)


class SqlApiSnowflake(Snowflake):
    """The SQL API: functions need a warehouse, SHOW GRANTS OF ROLE works."""

    def __init__(self, granted_to=None, invisible=(), **kwargs):
        super().__init__(**kwargs)
        self.granted_to = granted_to or {}  # role -> roles it's granted to
        self.invisible = set(invisible)

    def run(self, client, statement):
        for pattern, error in self.fail.items():
            if pattern in statement:
                self.statements.append(statement)
                raise RuntimeError(error)
        if statement.startswith("SELECT IS_"):
            self.statements.append(statement)
            raise RuntimeError("422 (391920): ... set the warehouse field in the body of the request")
        if statement.startswith("SHOW GRANTS OF ROLE "):
            self.statements.append(statement)
            role = statement[len("SHOW GRANTS OF ROLE "):].strip('"')
            if role in self.invisible:
                raise RuntimeError("002003 (02000): SQL compilation error: Role does not exist or not authorized.")
            return [{"role": role, "granted_to": "ROLE", "grantee_name": parent} for parent in self.granted_to.get(role, ())]
        return super().run(client, statement)


@pytest.fixture
def sql_api(monkeypatch):
    def make(**kwargs):
        world = SqlApiSnowflake(**kwargs)
        monkeypatch.setattr(_database_setup, "run_statement", world.run)
        return world

    return make


def test_through_the_sql_api_it_walks_up_from_the_holders(sql_api):
    # DATA_TEAM holds both privileges, and DATA_TEAM is granted to TRAINER: ready, with no warehouse.
    world = sql_api(granted_to={"DATA_TEAM": ["TRAINER"]})
    world.schema = [grant("CREATE EXPERIMENT", "DATA_TEAM")]
    world.account = [*ENABLED, grant("USE CORTEX TRAINING", "DATA_TEAM", on="ACCOUNT")]
    assert _job_access.check(CLIENT, "TRAINER").ready
    assert "SHOW GRANTS OF ROLE DATA_TEAM" in world.statements


def test_through_the_sql_api_two_levels_up_and_missing_ones(sql_api):
    world = sql_api(granted_to={"EXPERIMENTERS": ["DATA_TEAM"], "DATA_TEAM": ["TRAINER"]}, invisible={"ACCOUNTADMIN"})
    world.schema = [grant("CREATE EXPERIMENT", "EXPERIMENTERS")]
    world.account = ENABLED  # Only ACCOUNTADMIN, which TRAINER can't see, so doesn't inherit.
    access = _job_access.check(CLIENT, "TRAINER")
    assert [m.privilege for m in access.missing] == ["USE CORTEX TRAINING"]


def test_through_the_sql_api_another_error_is_couldnt_check(sql_api):
    world = sql_api()
    world.schema = [grant("CREATE EXPERIMENT", "DATA_TEAM")]
    world.account = ENABLED
    world.fail = {"SHOW GRANTS OF ROLE": "390114: authentication token has expired"}
    access = _job_access.check(CLIENT, "TRAINER")
    assert access.unknown and "expired" in access.unknown and access.missing == ()

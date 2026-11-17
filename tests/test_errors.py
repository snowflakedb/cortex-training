"""A refused request reads the same in the Python client, the CLI and the TUI.

``cortex_training.client.errors`` reads the service's answer once and holds the
one table of explanations. The shared fixtures below are the service's own
refusals, with the status, code and message its REST layer sends (see
docs/reference/rest-api.md sections 3.6 and 3.7). Each is raised through the
real client and shown by every surface: the exception's text, the CLI's
``error:`` output and the TUI's one-line error.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock

import pytest
import requests

import cortex_training.cli.commands as cli
from cortex_training import CortexTrainingClient
from cortex_training import DatabaseAccessError
from cortex_training import _login
from cortex_training.client import AccountJobListUnavailable
from cortex_training.client import ChunkGroupConflictError
from cortex_training.client import errors
from cortex_training.tui import views
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError

_PAT_HEADERS = {"X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN"}
_REQUEST_ID = "sf-req-1"


@pytest.fixture(autouse=True)
def _no_connection_from_the_shell(monkeypatch):
    """The CLI output compared here must not gain a warning about a connection the developer's shell exports."""
    for name in ("CONFIG", "CONNECTION", "HOST", "PAT", "BASE_URL", "DATABASE", "SCHEMA", "LOGIN_FILE", "ENDPOINT"):
        monkeypatch.delenv(f"CORTEX_TRAINING_{name}", raising=False)
    for name in ("HOST", "PAT", "DATABASE", "SCHEMA"):
        monkeypatch.delenv(f"SNOWFLAKE_{name}", raising=False)


@dataclass(frozen=True)
class Refusal:
    """One answer the service refuses a request with, and what every surface must show for it."""

    status: int
    reason: str
    body: dict | bytes
    code: str | None = None
    message: str | None = None
    # The explanation's first line, or None when the client has none for the answer.
    explanation: str | None = None
    sent_pat: bool = False

    @property
    def content(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        # As the service sends it: code, error_code, message and request_id, all top level.
        return json.dumps({**self.body, "request_id": _REQUEST_ID, "error_code": self.body.get("code")}).encode()


_SCHEMA_MESSAGE = "Schema MY_DB.PUBLIC is not found or not authorized"
_CREATE_EXPERIMENT_MESSAGE = "CREATE EXPERIMENT privilege required on schema MY_DB.PUBLIC"
_STAGE_MESSAGE = "Stage MY_DB.PUBLIC.EXPORTS is not found or not authorized"
_EXPERIMENT_MESSAGE = "Linked experiment is not found or not authorized for the training job"

REFUSALS = {
    "schema-not-found": Refusal(
        400, "Bad Request", {"code": "517602", "message": _SCHEMA_MESSAGE}, "517602", _SCHEMA_MESSAGE,
        "Role <ROLE> (couldn't determine your role) can't find database MY_DB or its schema PUBLIC.",
    ),
    "create-experiment-missing": Refusal(
        400, "Bad Request", {"code": "517602", "message": _CREATE_EXPERIMENT_MESSAGE}, "517602",
        _CREATE_EXPERIMENT_MESSAGE,
    ),
    "stage-not-found": Refusal(
        400, "Bad Request", {"code": "517602", "message": _STAGE_MESSAGE}, "517602", _STAGE_MESSAGE,
    ),
    "rejected-pat": Refusal(
        401, "Unauthorized", {"code": "390303", "message": "Invalid OAuth access token."}, "390303",
        "Invalid OAuth access token.", "Snowflake rejected your credentials (HTTP 401). The most common causes:",
        sent_pat=True,
    ),
    "server-error-not-json": Refusal(500, "Internal Server Error", b"<html><body>Server Error</body></html>"),
    "experiment-access": Refusal(
        500, "Internal Server Error", {"code": "517603", "message": _EXPERIMENT_MESSAGE}, "517603",
        _EXPERIMENT_MESSAGE, "This role can't read the job's Experiment (saved logs, GPU metrics and checkpoints "
        "live there). Ask for access to the Experiment, or use the role that created the job.",
    ),
    "an-unknown-invalid-request": Refusal(
        400, "Bad Request", {"code": "517602", "message": "Unsupported model: acme/unknown"}, "517602",
        "Unsupported model: acme/unknown",
    ),
}


def _client(refusal: Refusal) -> tuple[CortexTrainingClient, requests.Response]:
    """A client whose every GET the service answers with ``refusal``."""
    client = CortexTrainingClient(
        base_url="https://test.snowflakecomputing.com", database="MY_DB", schema="PUBLIC", max_retries=0
    )
    response = requests.Response()
    response.status_code = refusal.status
    response.reason = refusal.reason
    response.url = f"{client._prefix}/job-1"
    response.headers["Content-Type"] = "text/html" if isinstance(refusal.body, bytes) else "application/json"
    response.headers["x-snowflake-request-id"] = _REQUEST_ID
    response._content = refusal.content
    response._content_consumed = True
    if refusal.sent_pat:
        response.request = requests.Request("GET", response.url, headers=_PAT_HEADERS).prepare()
    client._session = MagicMock()
    client._session.get.return_value = response
    return client, response


def _sdk_error(refusal: Refusal) -> tuple[requests.HTTPError, requests.Response]:
    client, response = _client(refusal)
    with pytest.raises(requests.HTTPError) as excinfo:
        client.get_job("job-1")
    return excinfo.value, response


def _cli_output(refusal: Refusal) -> str:
    """Everything ``cortex-training get job-1`` prints, stdout then stderr."""
    client, _response = _client(refusal)
    stdout, stderr = io.StringIO(), io.StringIO()
    assert cli.main(["get", "job-1"], client_factory=lambda _args: client, stdout=stdout, stderr=stderr) == 1
    return stdout.getvalue() + stderr.getvalue()


def _requests_text(response: requests.Response) -> str:
    """What ``raise_for_status`` alone says about ``response``."""
    with pytest.raises(requests.HTTPError) as plain:
        response.raise_for_status()
    return str(plain.value)


def _shown(refusal: Refusal) -> list[tuple[str, str]]:
    """What each surface shows for ``refusal``."""
    error, _response = _sdk_error(refusal)
    return [("sdk", str(error)), ("cli", _cli_output(refusal)), ("tui", views.describe_error(error))]


# ─── Every surface ───────────────────────────────────────────────────────


@pytest.mark.parametrize("refusal", REFUSALS.values(), ids=REFUSALS.keys())
def test_every_surface_shows_the_same_code_message_and_explanation(refusal):
    for surface, text in _shown(refusal):
        one_line = views.collapse(text)
        if refusal.code:
            assert f"({refusal.code})" in text, surface
        if refusal.message:
            assert text.count(refusal.message) == 1, surface
        if refusal.explanation:
            assert one_line.count(views.collapse(refusal.explanation)) == 1, surface
            # After the service's message, never instead of it.
            assert one_line.index(views.collapse(refusal.explanation)) > one_line.index(refusal.message), surface
        else:
            assert "\n" not in text.removesuffix("\n").replace("\nresponse body: ", " "), surface


@pytest.mark.parametrize("refusal", REFUSALS.values(), ids=REFUSALS.keys())
def test_the_python_client_ends_its_first_line_with_the_service_code_and_message(refusal):
    error, response = _sdk_error(refusal)

    first, *rest = str(error).splitlines()
    suffix = f" ({refusal.code}): {refusal.message}" if refusal.code else ""
    assert first == f"[snowflake request id: {_REQUEST_ID}] {_requests_text(response)}{suffix}"
    assert (error.server_code, error.server_message, error.snowflake_request_id) == (
        refusal.code, refusal.message, _REQUEST_ID
    )
    if refusal.explanation:
        assert error.explanation.strip().startswith(refusal.explanation)
        assert "\n".join(rest) == error.explanation
    else:
        assert error.explanation is None and rest == []
    assert error.response is response


@pytest.mark.parametrize("refusal", REFUSALS.values(), ids=REFUSALS.keys())
def test_the_cli_prints_the_python_client_text_and_a_body_only_when_nothing_was_read_from_it(refusal):
    error, _response = _sdk_error(refusal)

    output = _cli_output(refusal)

    if isinstance(refusal.body, bytes):
        assert output == f"error: {error}\nresponse body: {refusal.body.decode()}\n"
    else:
        assert output == f"error: {error}\n"


@pytest.mark.parametrize("refusal", REFUSALS.values(), ids=REFUSALS.keys())
def test_the_tui_shows_one_line_without_the_url(refusal):
    error, response = _sdk_error(refusal)

    line = views.describe_error(error)

    assert "\n" not in line
    assert response.url not in line and "for url" not in line
    head = f"HTTP {refusal.status} {refusal.reason}"
    assert line.startswith(head + (f" ({refusal.code}): {refusal.message}" if refusal.code else ""))
    if refusal.explanation:
        assert line.endswith(views.collapse(error.explanation))
    elif not refusal.code:
        assert line == head


def test_a_missing_schema_is_still_a_database_access_error():
    error, response = _sdk_error(REFUSALS["schema-not-found"])

    assert isinstance(error, DatabaseAccessError)
    assert isinstance(error, requests.HTTPError) and isinstance(error, RuntimeError)
    assert error.response is response
    assert (error.kind, error.name, error.role) == ("schema", "MY_DB.PUBLIC", None)


# ─── Reading an answer ───────────────────────────────────────────────────


def _answer(content: bytes | str, status: int = 400, *, reason: str = "Bad Request") -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.reason = reason
    response._content = content.encode() if isinstance(content, str) else content
    response._content_consumed = True
    return response


@pytest.mark.parametrize(
    ("body", "code", "message"),
    [
        pytest.param(
            {"code": "517602", "error_code": "517602", "message": "bad", "request_id": "r"}, "517602", "bad",
            id="the-service-body",
        ),
        pytest.param({"error": {"code": "517602", "message": "bad"}}, "517602", "bad", id="wrapped"),
        pytest.param(
            {"error": {"message": "The requested model is not available.", "type": "not_found_error", "code": "404"}},
            "404", "The requested model is not available.", id="chat-completions",
        ),
        pytest.param({"error": "gateway refused"}, None, "gateway refused", id="an-error-string"),
        pytest.param(
            {"error": "invalid_token", "error_description": "The access token expired"}, None,
            "invalid_token - The access token expired", id="an-oauth-error-and-its-description",
        ),
        pytest.param(
            {"error": "Bad Request", "data": {"code": "517602", "message": "Schema DB.SCH is not found or not authorized"}},
            "517602", "Schema DB.SCH is not found or not authorized", id="an-error-string-over-a-nested-code-and-message",
        ),
        pytest.param({"code": "517604", "error": "busy"}, "517604", "busy", id="a-code-with-an-error-string"),
        pytest.param({"error": "outer", "data": {"detail": "x"}}, None, "outer", id="an-error-string-over-nothing-else"),
        pytest.param({"code": "", "error_code": "", "message": "no code"}, None, "no code", id="an-empty-code"),
        pytest.param({"code": "517602", "message": "  "}, "517602", None, id="a-blank-message"),
        pytest.param({"detail": [{"loc": ["body"], "msg": "bad"}]}, None, None, id="a-validation-list"),
    ],
)
def test_the_code_and_message_are_read_from_each_body_shape(body, code, message):
    error = errors.read_server_error(_answer(json.dumps(body)))

    assert (error.code, error.message) == (code, message)


def test_a_body_that_is_not_json_has_no_code_or_message():
    response = _answer(b"<html>busy</html>", 503, reason="Service Unavailable")
    response.headers["x-snowflake-request-id"] = "sf-1"

    assert errors.read_server_error(response) == errors.ServerError(
        503, "Service Unavailable", None, None, "sf-1", body_text="<html>busy</html>"
    )
    assert errors.read_server_error(None) is None


def test_an_error_string_does_not_hide_a_missing_schema_nested_under_it():
    body = {"error": "Bad Request", "data": {"code": "517602", "message": _SCHEMA_MESSAGE}}
    client, _response = _client(Refusal(400, "Bad Request", body))

    with pytest.raises(DatabaseAccessError) as excinfo:
        client.get_job("job-1")

    assert (excinfo.value.server_code, excinfo.value.server_message) == ("517602", _SCHEMA_MESSAGE)
    assert excinfo.value.explanation.startswith(REFUSALS["schema-not-found"].explanation)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_EXPERIMENT_MESSAGE.encode(), id="plain-text"),
        pytest.param(json.dumps({"detail": f"Cortex training getCheckpoints failed: {_EXPERIMENT_MESSAGE}"}).encode(),
                     id="detail-only"),
    ],
)
def test_a_body_without_a_message_is_read_as_text_for_the_experiment_explanation(body):
    refusal = Refusal(500, "Internal Server Error", body)
    error, _response = _sdk_error(refusal)

    explanation = REFUSALS["experiment-access"].explanation
    assert error.server_message is None and error.explanation == explanation
    assert str(error).endswith("\n" + explanation)
    assert views.experiment_access_hint(error) == explanation
    assert views.describe_error(error).endswith(explanation)
    assert _cli_output(refusal).count(explanation) == 1


def test_a_parsed_message_is_what_the_explanations_read():
    # The body's text names the Experiment too, but the service's own message is about something else.
    body = {"code": "517603", "message": "Internal error", "trace": _EXPERIMENT_MESSAGE}
    error, _response = _sdk_error(Refusal(500, "Internal Server Error", body))

    assert error.explanation is None
    assert views.experiment_access_hint(error) is None


def test_an_oauth_description_is_shown_whole():
    body = {"error": "invalid_token", "error_description": "Token expired at 12:00 UTC; request a new one"}

    error = errors.read_server_error(_answer(json.dumps(body), 401, reason="Unauthorized"))

    # Joined so that "invalid_token: Token" can't read as a credential to redact.
    assert error.shown_message == "invalid_token - Token expired at 12:00 UTC; request a new one"


def test_a_message_that_restates_the_reason_phrase_still_shows_on_every_surface():
    refusal = Refusal(403, "Forbidden", {"error": "Forbidden"})
    error, response = _sdk_error(refusal)

    # requests' own "403 Client Error: Forbidden for url" holds the same word by chance.
    assert str(error) == f"[snowflake request id: {_REQUEST_ID}] {_requests_text(response)}: Forbidden"
    assert _cli_output(refusal) == f"error: {error}\n"
    assert views.describe_error(error) == "HTTP 403 Forbidden: Forbidden"


def test_the_shown_message_is_redacted_and_cut():
    error = errors.read_server_error(_answer(json.dumps({"message": "token=abc " + "x" * 2000})))

    assert error.message.startswith("token=abc ")  # matched whole
    assert error.shown_message.startswith("token=<redacted> ")
    assert len(error.shown_message) == errors.MAX_SERVER_MESSAGE_CHARS + len("...")


# ─── The table of explanations ───────────────────────────────────────────


def _error(status: int | None, code: str | None, message: str) -> errors.ServerError:
    return errors.ServerError(status, "", code, message, None)


@pytest.mark.parametrize(
    ("error", "context", "topic"),
    [
        (_error(400, "517602", _SCHEMA_MESSAGE), None, errors.SCHEMA_NOT_FOUND),
        (_error(400, "517602", 'Schema "my db".PUBLIC is not found or not authorized.'), None, errors.SCHEMA_NOT_FOUND),
        (_error(422, "517602", _SCHEMA_MESSAGE), None, None),
        (_error(400, "517604", _SCHEMA_MESSAGE), None, None),
        (_error(400, "517602", _STAGE_MESSAGE), None, None),
        (_error(400, "517602", _CREATE_EXPERIMENT_MESSAGE), None, None),
        # A named experiment on create is the caller's input, not the job's Experiment.
        (_error(400, "517602", "Experiment MY_DB.PUBLIC.EXP not found or not authorized"), None, None),
        (_error(500, "517603", _EXPERIMENT_MESSAGE), None, errors.EXPERIMENT_ACCESS),
        (_error(500, "517603", "Linked experiment is not found for job JOB"), None, errors.EXPERIMENT_ACCESS),
        # Relayed from another component, the message is wrapped.
        (_error(500, "517604", f"Cortex training getCheckpoints failed: {_EXPERIMENT_MESSAGE}"), None,
         errors.EXPERIMENT_ACCESS),
        (_error(401, "390303", "Invalid OAuth access token."), errors.RequestContext(sent_pat=True),
         errors.CREDENTIALS_REJECTED),
        # A session token comes from a login that already accepted the PAT.
        (_error(401, "390303", "Invalid OAuth access token."), None, None),
        # The service's own code: Snowflake let the PAT in, and the training backend refused.
        (_error(401, "517604", "Cortex training getJob failed: backend token expired"),
         errors.RequestContext(sent_pat=True), None),
        (_error(401, None, ""), errors.RequestContext(sent_pat=True), errors.CREDENTIALS_REJECTED),
        (_error(400, "517602", "Unsupported model: acme/unknown"), None, None),
    ],
)
def test_the_first_entry_that_applies_explains_the_answer(error, context, topic):
    found = errors.explain(error, context)

    assert (found.topic if found else None) == topic


def test_the_schema_explanation_names_what_the_client_knows():
    context = errors.RequestContext(role="TRAINER", profile_client=True, profile="prod", database="DB", schema="SCH")

    text = errors.explain(_error(400, "517602", "Schema DB.SCH is not found or not authorized"), context).text

    assert text.splitlines()[0] == "Role TRAINER can't find database DB or its schema SCH (from profile prod)."
    assert "GRANT USAGE ON SCHEMA DB.SCH TO ROLE TRAINER" in text
    # Knowing nothing about the client, it still says what to do.
    general = errors.explain(_error(400, "517602", "Schema DB.SCH is not found or not authorized")).text
    assert general.startswith("The database or schema in the request doesn't exist, or this role can't use it.")


@pytest.mark.parametrize(
    "text",
    [
        "003001 (42501): SQL access control error: Insufficient privileges to operate on experiment 'E'",
        "002003 (02000): SQL compilation error: Experiment 'DB.SCH.TRAINING_RUNS' does not exist or not authorized.",
        "002003 (02000): SQL compilation error: Experiment \"E\" does not exist or not authorized.",
        "snow://experiment/DB.SCH.E/versions/run/logs' does not exist or not authorized",
    ],
)
def test_a_refused_sql_statement_on_the_experiment_is_explained_from_its_text(text):
    found = errors.look_up(RuntimeError(text))

    assert found is not None and found.topic == errors.EXPERIMENT_ACCESS
    assert views.experiment_access_hint(RuntimeError(text)) == found.text


def test_a_refusal_about_another_object_is_not_an_experiment_problem():
    text = "003001 (42501): SQL access control error: Insufficient privileges to operate on warehouse 'W' for experiment E"

    assert errors.look_up(RuntimeError(text)) is None
    assert views.experiment_access_hint(RuntimeError(text)) is None


def test_the_attached_explanation_wins_even_when_there_is_none():
    response = _answer(json.dumps({"code": "517602", "message": _SCHEMA_MESSAGE}))
    error = requests.HTTPError("400", response=response)
    assert errors.explanation_of(error).startswith("The database or schema in the request")

    errors.attach(error, errors.read_server_error(response), None)

    assert errors.explanation_of(error) is None


def test_the_message_gains_the_code_message_and_explanation_once():
    error = _error(400, "517602", "job JOB is busy")

    once = errors.with_server_error("400 Client Error", error, "Next steps.", hide=lambda s: s.replace("JOB", "DEP"))

    assert once == "400 Client Error (517602): job DEP is busy\nNext steps."
    assert errors.with_server_error(once, error, "Next steps.", hide=lambda s: s.replace("JOB", "DEP")) == once


def test_a_message_of_several_lines_is_added_once_too():
    error = _error(500, "517604", "Cortex training getJob failed: rpc error\ndetails: line two")

    once = errors.with_server_error("500 Server Error", error, None)

    assert once == "500 Server Error (517604): Cortex training getJob failed: rpc error\ndetails: line two"
    assert errors.with_server_error(once, error, None) == once
    assert errors.with_server_error(once, error, "Next steps.") == once + "\nNext steps."


# ─── The TUI's access hints ──────────────────────────────────────────────


def test_the_tui_access_hint_keeps_the_service_message_before_the_explanation():
    error, _response = _sdk_error(REFUSALS["experiment-access"])

    hint = views.access_hint(error, "saved logs")

    assert hint == views.describe_error(error)
    assert hint.index(_EXPERIMENT_MESSAGE) < hint.index("This role can't read the job's Experiment")


def test_the_tui_access_hint_still_names_other_refused_reads():
    refusal = Refusal(403, "Forbidden", {"code": "003001", "message": "Insufficient privileges on the stage"})
    error, _response = _sdk_error(refusal)

    assert views.access_hint(error, "GPU metrics") == (
        f"this role isn't allowed to read this job's GPU metrics ({views.describe_error(error)})"
    )
    assert views.access_hint(_sdk_error(REFUSALS["stage-not-found"])[0], "GPU metrics") is None


def test_a_tui_line_with_no_room_leaves_the_explanation_out():
    error, _response = _sdk_error(REFUSALS["experiment-access"])

    assert views.describe_error(error, explain=False) == f"HTTP 500 Internal Server Error (517603): {_EXPERIMENT_MESSAGE}"
    assert views.with_explanation(views.describe_error(error, explain=False), error) == views.describe_error(error)


_NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


class _EveryListFails:
    """A client bound to MY_DB.PUBLIC whose account-level and per-schema job lists both raise ``error``."""

    database, schema = "MY_DB", "PUBLIC"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def list_jobs_page(self, **_filters) -> dict:
        raise self.error

    def list_jobs(self, status=None) -> list:
        raise self.error


def test_the_tui_job_list_names_both_routes_before_the_explanation():
    error, _response = _sdk_error(REFUSALS["rejected-pat"])

    with pytest.raises(JobListError) as failed:
        JobLister(_EveryListFails(error)).load(_NOW)

    line = str(failed.value)
    head = "HTTP 401 Unauthorized (390303): Invalid OAuth access token."
    assert line.startswith(f"the account-wide job list answered {head}; and list_jobs answered {head}")
    assert line.index("; and list_jobs") < line.index("Snowflake rejected your credentials (HTTP 401).")


# ─── Errors raised in place of an HTTP error ─────────────────────────────


def test_an_account_list_refusal_carries_the_answers_fields():
    client, _response = _client(Refusal(403, "Forbidden", {"code": "003001", "message": "Insufficient privileges"}))

    with pytest.raises(AccountJobListUnavailable) as excinfo:
        client.list_jobs_page()

    error = excinfo.value
    assert (error.server_code, error.server_message, error.snowflake_request_id, error.explanation) == (
        "003001", "Insufficient privileges", _REQUEST_ID, None
    )


def test_a_chunk_group_error_keeps_its_body_in_the_cli():
    message = "chunk group g1 was reused with a different total"
    _client_unused, response = _client(
        Refusal(409, "Conflict", {"code": "chunk_group_conflict", "message": message, "chunk_group_id": "g1"})
    )
    error = ChunkGroupConflictError(message, response=response, detail={"code": "chunk_group_conflict"})

    shown = cli._format_error(error)

    # Its own text leaves out the code and the group's ids, which only the body has.
    assert shown.endswith(f"\nresponse body: {response.text}")
    assert (error.server_code, error.server_message, error.explanation) == ("chunk_group_conflict", message, None)


# ─── Refused SQL on the job's Experiment ─────────────────────────────────

_SQL_REFUSAL = "003001 (42501): SQL access control error: Insufficient privileges to operate on experiment 'RUNS'"


def _refused_sql():
    from snowflake.connector.errors import ProgrammingError

    return ProgrammingError(msg=_SQL_REFUSAL, errno=3001, sqlstate="42501", sfqid="query-1")


def test_the_cli_explains_a_refused_sql_statement_as_the_tui_does():
    error = _refused_sql()
    explanation = errors.explanation_of(error)
    assert explanation.startswith("This role can't read the job's Experiment, where saved logs and GPU metrics live")

    shown = cli._format_connector_error(error)

    assert shown.count(explanation) == 1 and shown.index(explanation) > shown.index("Insufficient privileges")
    assert "snowflake query id: query-1" in shown
    assert views.describe_error(error).count(views.collapse(explanation)) == 1
    # Any other error the CLI prints gets it the same way, once.
    assert cli._format_error(RuntimeError(_SQL_REFUSAL)) == f"{_SQL_REFUSAL}\n{explanation}"


def test_the_get_warning_stays_one_line_with_the_explanation_under_it():
    error, _response = _sdk_error(REFUSALS["experiment-access"])
    client = MagicMock()
    client.get_experiment_run.side_effect = error
    stderr = io.StringIO()

    assert cli._job_experiment_name(client, "job-1", stderr) is None

    warning, *under = stderr.getvalue().splitlines()
    assert warning == (
        f"warning: could not read the job's experiment ({str(error).splitlines()[0]}); experiment_name is null, "
        "so a job submitted from this gets an experiment of its own"
    )
    assert "\n".join(under) == error.explanation


def test_the_get_warning_keeps_every_line_of_an_unexplained_error_on_its_one_line():
    client = MagicMock()
    client.get_experiment_run.side_effect = RuntimeError("the run failed:\n  step 3 timed out")
    stderr = io.StringIO()

    assert cli._job_experiment_name(client, "job-1", stderr) is None

    assert stderr.getvalue() == (
        "warning: could not read the job's experiment (the run failed: step 3 timed out); experiment_name is null, "
        "so a job submitted from this gets an experiment of its own\n"
    )


def test_an_explanation_reads_only_the_start_of_a_long_message():
    long_prefix = "x" * 5000

    assert errors.explain(_error(500, "517603", _EXPERIMENT_MESSAGE + long_prefix)) is not None
    assert errors.explain(_error(500, "517603", long_prefix + _EXPERIMENT_MESSAGE)) is None
    # A message of many experiment paths, which a whole-message search would scan quadratically.
    assert errors.explain(_error(500, "517603", "snow://experiment/x" * 200_000)) is None


# ─── A rejected PAT ──────────────────────────────────────────────────────


def test_a_rejected_pat_is_explained_even_when_the_answer_is_not_json():
    error, _response = _sdk_error(Refusal(401, "Unauthorized", b"<html>Unauthorized</html>", sent_pat=True))

    assert error.server_message is None
    assert error.explanation.strip().startswith("Snowflake rejected your credentials (HTTP 401).")
    assert str(error).endswith(error.explanation)


def test_a_401_the_training_backend_relays_is_not_a_rejected_pat():
    refusal = Refusal(401, "Unauthorized", {"code": "517604", "message": "Cortex training getJob failed: expired"},
                      sent_pat=True)

    error, _response = _sdk_error(refusal)

    assert error.explanation is None
    assert "Snowflake rejected your credentials" not in _cli_output(refusal) + views.describe_error(error)


# ─── Login ───────────────────────────────────────────────────────────────


def _login_flow() -> "_login._Login":
    flow = _login._Login(_login._Prompter(io.StringIO(), io.StringIO()), lambda *args: None)
    flow.offer_delete = lambda *args, **kwargs: None
    return flow


def test_login_names_a_refused_request_by_its_explanation_or_its_message():
    schema_error, _response = _sdk_error(REFUSALS["schema-not-found"])
    privilege_error, _response = _sdk_error(REFUSALS["create-experiment-missing"])

    assert _login._error_line(schema_error) == (
        f"[snowflake request id: {_REQUEST_ID}] {REFUSALS['schema-not-found'].explanation}"
    )
    assert _login._error_line(privilege_error) == (
        f"[snowflake request id: {_REQUEST_ID}] HTTP 400 (517602): {_CREATE_EXPERIMENT_MESSAGE}"
    )
    assert _login._error_line(RuntimeError("first line\nsecond line")) == "first line"


def test_the_login_check_names_the_role_that_cant_find_the_schema():
    client, _response = _client(REFUSALS["schema-not-found"])
    client._auth_provider = MagicMock()
    client._auth_provider.get_token.return_value = "session-token"
    client._auth_provider.connection_name = "prod"
    client._auth_provider.role = "TRAINER"

    with pytest.raises(_login.ProfileFailed) as failed:
        _login_flow().verify_service(_login.Check("prod", "MY_DB", "PUBLIC", client=client))

    assert str(failed.value) == (
        "Cortex Training can't find database MY_DB or schema PUBLIC with profile prod: "
        f"[snowflake request id: {_REQUEST_ID}] "
        "Role TRAINER can't find database MY_DB or its schema PUBLIC (from profile prod)."
    )


def test_converting_an_old_config_names_the_role_that_cant_find_the_schema():
    from pathlib import Path

    from cortex_training import _convert

    error, _response = _sdk_error(REFUSALS["schema-not-found"])
    old = _convert.OldConfig(path=Path("config.json"), raw={}, host="ACCOUNT.snowflakecomputing.com", pat="unused",
                             database="MY_DB", schema="PUBLIC", settings={})

    assert _convert.explain(error, old) == (
        f"The check failed: [snowflake request id: {_REQUEST_ID}] {REFUSALS['schema-not-found'].explanation}"
    )


def test_a_service_message_that_says_isnt_enabled_is_not_a_disabled_account():
    refusal = Refusal(500, "Internal Server Error", {"code": "517603", "message": "Cortex Training isn't enabled yet"})
    client, _response = _client(refusal)
    flow = _login_flow()
    flow.profiles = lambda: {"prod": {}}
    flow.open_client = lambda *args: client

    status, reason, signed_in = flow._probe_profile("prod")

    assert (status, signed_in) == (_login.CT_COULDNT_CHECK, True)
    assert reason.endswith("HTTP 500 (517603): Cortex Training isn't enabled yet")

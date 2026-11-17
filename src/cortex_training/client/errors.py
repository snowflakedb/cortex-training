"""What a refused Cortex Training request says, read once for every place that shows it.

:func:`read_server_error` reads an HTTP answer's status, the service's error
``code`` and ``message`` and the Snowflake request id. :func:`explain` looks the
answer up in :data:`EXPLANATIONS`, the one table of next steps the client adds
to refusals it recognizes. The client's exceptions, the CLI's ``error:`` line
and the TUI's one-line errors are all built from these two, so a refusal shows
the same code, message and explanation in a traceback, a terminal and the TUI.
An explanation always follows the service's message; it never replaces it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from cortex_training.client.auth import credentials_rejected_hint

logger = logging.getLogger(__name__)

# Longest server error message kept in an error the client raises.
MAX_SERVER_MESSAGE_CHARS = 1000
SNOWFLAKE_REQUEST_ID_HEADER = "x-snowflake-request-id"
_TOKEN_TYPE_HEADER = "X-Snowflake-Authorization-Token-Type"
_PAT_TOKEN_TYPE = "PROGRAMMATIC_ACCESS_TOKEN"

# The attributes the client sets on the HTTP errors it raises (see attach()).
SERVER_ERROR_ATTRIBUTES = ("server_code", "server_message", "snowflake_request_id", "explanation")

_SECRET_IN_ERROR_RE = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*"
    r"(?:bearer\s+|snowflake\s+token\s*=\s*)?|"
    r"bearer\s+|snowflake\s+token\s*=\s*|"
    r"(?:password|secret|credential|pat|api[_-]?key|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|token)"
    r"[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|\S+)"
)


def redact_secrets(text: str) -> str:
    """Return ``text`` with the obvious secrets ``_SECRET_IN_ERROR_RE`` finds redacted."""
    return _SECRET_IN_ERROR_RE.sub(r"\1<redacted>", text)


def bounded_server_message(message: str) -> str:
    """Return ``message`` cut to ``MAX_SERVER_MESSAGE_CHARS``, ending in ``...`` when cut."""
    if len(message) > MAX_SERVER_MESSAGE_CHARS:
        return message[:MAX_SERVER_MESSAGE_CHARS] + "..."
    return message


def response_snowflake_request_id(response: Any) -> str | None:
    """The ``x-snowflake-request-id`` header of ``response``, or None."""
    headers = getattr(response, "headers", None) or {}
    request_id = headers.get(SNOWFLAKE_REQUEST_ID_HEADER)
    return request_id if isinstance(request_id, str) and request_id else None


def sent_pat(response: Any, fallback_headers: Any = None) -> bool:
    """Whether the request ``response`` answers sent a PAT directly.

    Connection-profile requests carry a session token from a login that
    already accepted the PAT, so a 401 there is not a PAT problem.
    ``fallback_headers`` are read when the response has no request, as a
    mocked one may not.
    """
    headers = getattr(getattr(response, "request", None), "headers", None) or fallback_headers or {}
    return headers.get(_TOKEN_TYPE_HEADER) == _PAT_TOKEN_TYPE


def iter_error_dicts(value: Any, *, depth: int = 0) -> Iterator[dict]:
    """Yield dictionaries from direct or service-wrapped JSON error bodies."""
    if depth > 8:
        return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_error_dicts(child, depth=depth + 1)
        return
    if isinstance(value, list):
        for child in value:
            yield from iter_error_dicts(child, depth=depth + 1)
        return
    if not isinstance(value, str):
        return

    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            decoded, _ = decoder.raw_decode(value[index:])
        except RecursionError:
            # Nested deeper than the decoder recurses. Each later bracket would
            # decode nearly the same text again, so nothing more is read from it.
            return
        except (TypeError, ValueError):
            continue
        yield from iter_error_dicts(decoded, depth=depth + 1)


@dataclass(frozen=True)
class ServerError:
    """What one HTTP answer says about why it refused a request.

    ``message`` is the service's own text, whole and unredacted (the
    explanations match its first ``_MATCHED_CHARS`` characters);
    :attr:`shown_message` is how errors show it. ``body_text`` is the start of the
    answer's body as text, which the explanations read instead when no message
    was found in it (a plain-text body, or one with only a ``detail``); it is
    never shown. ``status`` is None when the answer has no integer status (as a
    mocked one may not).
    """

    status: int | None
    reason: str
    code: str | None
    message: str | None
    snowflake_request_id: str | None
    body_text: str | None = None

    @property
    def shown_message(self) -> str | None:
        """``message`` as errors show it: common credentials redacted, cut at 1,000 characters."""
        return bounded_server_message(redact_secrets(self.message)) if self.message else None

    @property
    def suffix(self) -> str:
        """`` (<code>): <message>``, or the half the answer has; ``""`` with neither."""
        text = f" ({self.code})" if self.code else ""
        if self.shown_message:
            text += f": {self.shown_message}"
        return text


def _error_text(candidate: dict) -> str | None:
    """A dict's string ``error``, followed by its ``error_description`` when it has one; else None.

    As some gateways send it (OAuth's ``{"error": "invalid_token",
    "error_description": "..."}``). The two are joined with `` - ``, which the
    credential redaction does not read as ``token: <value>``.
    """
    error = candidate.get("error")
    if not isinstance(error, str) or not error.strip():
        return None
    description = candidate.get("error_description")
    if isinstance(description, str) and description.strip():
        return f"{error.strip()} - {description.strip()}"
    return error.strip()


def _code_and_message(body: Any) -> tuple[str | None, str | None]:
    """The first ``code`` and ``message`` in a JSON error body, plain or wrapped.

    The service sends ``{"code", "error_code", "message", "request_id"}``, its
    chat completions ``{"error": {"message", "type", "code"}}``. The first dict
    with a ``code`` or a ``message`` gives them, so a code and message nested
    under another field are found even when the outer dict has a string
    ``error``; that dict's own ``error`` string (see :func:`_error_text`) stands
    in for a ``message`` it lacks. Only when no dict has a code or a message
    does the first string ``error`` count as the message on its own. An empty
    code (the service sends ``""`` for an error without one) is no code.
    """
    fallback = None
    for candidate in iter_error_dicts(body):
        raw_code = candidate.get("code") or candidate.get("error_code")
        code = str(raw_code).strip() if raw_code is not None else ""
        message = candidate.get("message")
        if code or isinstance(message, str):
            if not (isinstance(message, str) and message.strip()):
                message = _error_text(candidate)
            return code or None, message.strip() if isinstance(message, str) and message.strip() else None
        if fallback is None:
            fallback = _error_text(candidate)
    return None, fallback


def read_server_error(response: Any) -> ServerError | None:
    """Read what ``response`` says about a refusal; None without a response.

    Never raises: a body that is not JSON, or can't be read (such as a string
    nested deeper than the JSON decoder recurses), has no code or message.
    """
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    code = message = body_text = None
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - not JSON, or no body
        body = None
    try:
        code, message = _code_and_message(body)
    except Exception:  # noqa: BLE001 - such as a RecursionError on a deeply nested string
        logger.debug("could not read the service's error from the response body", exc_info=True)
    try:
        raw = response.text
        body_text = raw[:_MATCHED_CHARS] if isinstance(raw, str) and raw.strip() else None
    except Exception:  # noqa: BLE001 - a body that can't be read as text
        logger.debug("could not read the response body as text", exc_info=True)
    return ServerError(
        status=status if isinstance(status, int) and not isinstance(status, bool) else None,
        reason=str(getattr(response, "reason", "") or ""),
        code=code,
        message=message,
        snowflake_request_id=response_snowflake_request_id(response),
        body_text=body_text,
    )


# ─── Explanations ───────────────────────────────────────────────────────

SCHEMA_NOT_FOUND = "schema_not_found"
EXPERIMENT_ACCESS = "experiment_access"
CREDENTIALS_REJECTED = "credentials_rejected"

# The API's invalid-request code; every 400 refusal shares it, so the message
# tells them apart (see docs/reference/rest-api.md section 3.7).
INVALID_REQUEST_CODE = "517602"


@dataclass(frozen=True)
class RequestContext:
    """What the client knows about a refused request, for explanations that name it.

    ``profile_client`` is whether it signs in with a Snowflake connection
    profile, ``profile`` that profile's name (None for the default profile).
    Fields left unknown give a more general explanation.
    """

    sent_pat: bool = False
    role: str | None = None
    profile_client: bool = False
    profile: str | None = None
    database: str | None = None
    schema: str | None = None


@dataclass(frozen=True)
class Explanation:
    """The next steps for a refusal; ``topic`` names which kind it is."""

    topic: str
    text: str


@dataclass(frozen=True)
class _Entry:
    """One row of :data:`EXPLANATIONS`: when it applies, and what it says.

    ``pattern`` is searched in the service's message; in the answer's body as
    text when no message was found in it (a plain-text body, or one with only a
    ``detail``); and, for an error with no HTTP answer, such as a refused SQL
    statement, in the error's text. ``status``, ``code`` and ``needs_pat``
    further limit it to an HTTP answer with that status, that code, or to a
    request that sent a PAT; ``unless_code`` leaves out answers whose code it
    matches.
    """

    topic: str
    pattern: re.Pattern[str]
    text: Callable[[RequestContext], str]
    status: int | None = None
    code: str | None = None
    needs_pat: bool = False
    unless_code: re.Pattern[str] | None = None

    def applies(self, error: ServerError, context: RequestContext) -> bool:
        if self.status is not None and error.status != self.status:
            return False
        if self.code is not None and error.code != self.code:
            return False
        if self.unless_code is not None and error.code and self.unless_code.fullmatch(error.code):
            return False
        if self.needs_pat and not context.sent_pat:
            return False
        # Only the start, so that a huge message costs little on every failed attempt.
        said = error.message if error.message is not None else (error.body_text or "")
        return bool(self.pattern.search(said[:_MATCHED_CHARS]))


# How much of a message the explanations look at. Every refusal they explain
# says so in its first sentence; a message can be far longer (a relayed log).
_MATCHED_CHARS = 4000


def _schema_not_found(context: RequestContext) -> str:
    """Explain the API's 400 saying the request path's DATABASE.SCHEMA can't be resolved.

    The service says the same whether the database or the schema is missing,
    or the role can't use either, so the explanation covers all three.
    """
    database, schema, role = context.database, context.schema, context.role
    if not (database and schema):
        return (
            "The database or schema in the request doesn't exist, or this role can't use it. "
            "Create it, ask an admin for USAGE on both, or use another database."
        )
    if role:
        who = f"Role {role}"
    elif context.sent_pat:
        who = "Your PAT's role (its role restriction, or your user's DEFAULT_ROLE if it has none)"
    else:
        who = "Role <ROLE> (couldn't determine your role)"
    grantee = role or "<ROLE>"
    source = ""
    if context.profile_client:
        source = f" (from profile {context.profile})" if context.profile else " (from your default Snowflake profile)"
    create_database, create_schema = schema_statements(database, schema)
    lines = [
        f"{who} can't find database {database} or its schema {schema}{source}.",
        f"- If they don't exist: run `{create_database}` and `{create_schema}` (or ask an admin to).",
        f"- If they exist: ask an admin to run `GRANT USAGE ON DATABASE {database} TO ROLE {grantee}` "
        f"and `GRANT USAGE ON SCHEMA {database}.{schema} TO ROLE {grantee}`.",
    ]
    if not role and context.sent_pat:
        lines.insert(
            1,
            "- If that's the wrong role: restrict the PAT to a role that has access, or run "
            "`ALTER USER <USER> SET DEFAULT_ROLE = <ROLE>`. Snowsight shows the token's role under "
            "Settings > Authentication.",
        )
    lines.append("- Or use another database: --database NAME / CORTEX_TRAINING_DATABASE.")
    return "\n".join(lines)


def schema_statements(database: str, schema: str) -> tuple[str, str]:
    """The CREATE DATABASE and CREATE SCHEMA statements that set up ``database.schema``."""
    return f"CREATE DATABASE IF NOT EXISTS {database}", f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}"


_ASK = "Ask for access to the Experiment, or use the role that created the job."

# How Snowflake refuses SQL on the job's Experiment for this role (the LIST and
# GET that read saved logs and GPU metrics): "003001 (42501): SQL access control
# error: Insufficient privileges to operate on experiment 'NAME'", "Experiment
# 'NAME' does not exist or not authorized", or the same said of a
# snow://experiment/ path. The refusal itself must be about the Experiment, so a
# denial about some other object (a warehouse, a schema) keeps its own text even
# when the message names the Experiment elsewhere.
_EXPERIMENT_SQL_DENIED = re.compile(
    r"insufficient privileges to operate on experiment\b"
    r"|\bexperiment(?:\s+run)?\s+(?:'[^']*'|\"[^\"]*\"|[^\s'\"]+)\s+does not exist or not authorized"
    r"|snow://experiment/[^\s'\"]*['\"]?:?\s+does not exist or not authorized",
    re.IGNORECASE,
)


def _text(value: str) -> Callable[[RequestContext], str]:
    return lambda _context: value


# The first entry that applies explains the refusal.
EXPLANATIONS: tuple[_Entry, ...] = (
    _Entry(
        SCHEMA_NOT_FOUND,
        re.compile(r"^Schema .+ is not found or not authorized"),
        _schema_not_found,
        status=400,
        code=INVALID_REQUEST_CODE,
    ),
    # The service's 500 (code 517603) when it can't resolve the job's
    # Experiment for the caller: the checkpoint routes say "... or not
    # authorized for the training job", and the experiment-run route that saved
    # logs and GPU metrics start with says "... for job <id>", which a deleted
    # Experiment gets too. Searched anywhere in the message's first
    # _MATCHED_CHARS characters, since a failure relayed from another component
    # reads "Cortex training <op> failed: ...".
    _Entry(
        EXPERIMENT_ACCESS,
        re.compile(r"Linked experiment is not found or not authorized"),
        _text(
            "This role can't read the job's Experiment (saved logs, GPU metrics and checkpoints live there). "
            + _ASK
        ),
    ),
    _Entry(
        EXPERIMENT_ACCESS,
        re.compile(r"Linked experiment is not found"),
        _text(
            "This role can't read the job's Experiment, or it no longer exists (saved logs, GPU metrics and "
            "checkpoints live there). " + _ASK
        ),
    ),
    # A MONITOR-only role: the service still lists the checkpoints.
    _Entry(
        EXPERIMENT_ACCESS,
        _EXPERIMENT_SQL_DENIED,
        _text(
            "This role can't read the job's Experiment, where saved logs and GPU metrics live "
            "(listing checkpoints works). " + _ASK
        ),
    ),
    # A 401 that carries one of the service's own codes (5176xx, such as a
    # failure relayed from the training backend) answered a request that
    # Snowflake had already let in, so the PAT is not what was refused.
    _Entry(
        CREDENTIALS_REJECTED,
        re.compile(""),
        lambda _context: credentials_rejected_hint("Snowflake rejected your credentials (HTTP 401)."),
        status=401,
        needs_pat=True,
        unless_code=re.compile(r"5176\d\d"),
    ),
)


def explain(error: ServerError | None, context: RequestContext | None = None) -> Explanation | None:
    """The first entry of :data:`EXPLANATIONS` that applies to ``error``, or None."""
    if error is None:
        return None
    context = context or RequestContext()
    for entry in EXPLANATIONS:
        if entry.applies(error, context):
            return Explanation(entry.topic, entry.text(context))
    return None


def look_up(exc: BaseException) -> Explanation | None:
    """Look ``exc`` up in :data:`EXPLANATIONS` now, knowing only what it carries.

    An HTTP error is matched on its answer; any other error, such as a refused
    SQL statement, on its text. Prefer :func:`explanation_of`, which returns
    what the client attached, with what it knew about the request.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        return explain(read_server_error(response), RequestContext(sent_pat=sent_pat(response)))
    return explain(ServerError(status=None, reason="", code=None, message=str(exc), snowflake_request_id=None))


def explanation_of(exc: BaseException) -> str | None:
    """The explanation for ``exc``: the one the client attached, else :func:`look_up`'s."""
    if hasattr(exc, "explanation"):
        attached = getattr(exc, "explanation")
        return attached if isinstance(attached, str) and attached else None
    found = look_up(exc)
    return found.text if found else None


def _unchanged(text: str) -> str:
    return text


def _ends_a_line(text: str, part: str) -> bool:
    """Whether ``part`` is in ``text`` with the end of ``text``, or a line break, right after it."""
    start = text.find(part)
    while start >= 0:
        end = start + len(part)
        if end == len(text) or text[end] == "\n":
            return True
        start = text.find(part, start + 1)
    return False


def with_server_error(
    message: str,
    error: ServerError | None,
    explanation: str | None,
    *,
    hide: Callable[[str], str] = _unchanged,
) -> str:
    """``message`` with the service's code and message at the end of its first line, then ``explanation`` last.

    ``hide`` rewrites what is added, such as to hide a backing job ID. What is
    already there is not added again, so the same error can pass through more
    than once. The code and message count as there only where the client puts
    them, ending a line: requests' own text can hold the same words by chance,
    as ``403 Client Error: Forbidden for url`` holds a body's ``Forbidden``.
    """
    suffix = hide(error.suffix) if error and error.suffix else ""
    if suffix and not _ends_a_line(message, suffix):
        first, newline, rest = message.partition("\n")
        message = first + suffix + newline + rest
    if explanation:
        explanation = hide(explanation)
        if explanation not in message:
            message += "\n" + explanation
    return message


def attach(
    exc: BaseException,
    error: ServerError | None,
    explanation: str | None,
    *,
    message: str | None = None,
) -> None:
    """Set the :data:`SERVER_ERROR_ATTRIBUTES` on ``exc``.

    ``server_message`` is the service's message as the error shows it
    (``message`` when the caller rewrote it, such as to hide a backing job ID).
    """
    exc.server_code = error.code if error else None  # type: ignore[attr-defined]
    exc.server_message = message if message is not None else (error.shown_message if error else None)  # type: ignore[attr-defined]
    exc.snowflake_request_id = error.snowflake_request_id if error else None  # type: ignore[attr-defined]
    exc.explanation = explanation or None  # type: ignore[attr-defined]


def carry(exc: BaseException, source: BaseException | None = None) -> None:
    """Give ``exc``, an error raised in place of an HTTP error, the :data:`SERVER_ERROR_ATTRIBUTES`.

    They are ``source``'s, for an error whose message quotes ``source``'s;
    otherwise they are read from ``exc``'s own answer, with no explanation,
    since such an error's message is its own. An error that has them keeps
    them.
    """
    if all(hasattr(exc, name) for name in SERVER_ERROR_ATTRIBUTES):
        return
    if source is not None and all(hasattr(source, name) for name in SERVER_ERROR_ATTRIBUTES):
        for name in SERVER_ERROR_ATTRIBUTES:
            setattr(exc, name, getattr(source, name))
        return
    attach(exc, read_server_error(getattr(exc, "response", None)), None)


def decorate(
    exc: BaseException,
    error: ServerError | None,
    explanation: str | None,
    *,
    hide: Callable[[str], str] = _unchanged,
) -> None:
    """End ``exc``'s message as :func:`with_server_error` does, and :func:`attach` the same fields."""
    if exc.args and isinstance(exc.args[0], str):
        exc.args = (with_server_error(exc.args[0], error, explanation, hide=hide), *exc.args[1:])
    shown = error.shown_message if error else None
    attach(exc, error, hide(explanation) if explanation else None, message=hide(shown) if shown else None)

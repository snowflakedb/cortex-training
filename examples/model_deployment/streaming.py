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

"""Call a model deployment's OpenAI-compatible chat completions endpoint.

A deployment serves ``POST {base_url}/chat/completions``. With ``stream=True``
the response is server-sent events: one ``data: {chunk}`` line per delta, and
``data: [DONE]`` at the end. The endpoint takes ``max_completion_tokens`` and
rejects ``max_tokens`` (HTTP 400 ``unsupported_parameter``). Only ``requests``
is needed, so the example runs without the ``openai`` package.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests

SSE_DATA_PREFIX = "data:"
SSE_DONE = "[DONE]"
# Response bodies are cut to this many characters in error messages.
ERROR_BODY_LIMIT = 500


def deployment_headers(pat: str) -> dict[str, str]:
    """Return the headers that authenticate a deployment call with a PAT."""
    return {
        "Authorization": f"Bearer {pat}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "Content-Type": "application/json",
    }


# Either a PAT, or a function returning the auth headers, such as
# ``CortexTrainingClient.model_deployment_headers``. After a session-expired 401
# it is called with ``rejected=<the headers that were sent>`` to retry once.
Auth = str | Callable[..., dict[str, str]]

# Snowflake error codes for an expired or invalidated session token.
SESSION_EXPIRED_CODES = frozenset({"390111", "390112", "390114"})


def _headers(auth: Auth, rejected: dict[str, str] | None = None) -> dict[str, str]:
    if isinstance(auth, str):
        return deployment_headers(auth)
    headers = auth() if rejected is None else auth(rejected=rejected)
    return {**headers, "Content-Type": "application/json"}


def _session_expired(sent: dict[str, str], response: requests.Response) -> bool:
    """Whether ``response`` rejected the session token in ``sent`` as expired.

    A PAT is never refreshed, and a 401 with any other Snowflake error code is
    not an expiry, so neither is retried (that would only repeat a login).
    """
    if response.status_code != 401 or not sent.get("Authorization", "").startswith("Snowflake Token="):
        return False
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    codes = [
        str(code)
        for code in (
            body.get("code") if isinstance(body, dict) else None,
            error.get("code") if isinstance(error, dict) else None,
        )
        if code is not None
    ]
    return not codes or any(code in SESSION_EXPIRED_CODES for code in codes)


def _post(http: requests.Session, url: str, auth: Auth, extra: dict[str, str], **kwargs: Any) -> requests.Response:
    """POST with ``auth``; a session-expired 401 is retried once with a new session."""
    sent = _headers(auth)
    response = http.post(url, headers={**sent, **extra}, **kwargs)
    if not isinstance(auth, str) and _session_expired(sent, response):
        response.close()
        response = http.post(url, headers={**_headers(auth, rejected=sent), **extra}, **kwargs)
    return response


def chat_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    """Return an OpenAI ``messages`` list for one user prompt."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages


class DeploymentCallError(RuntimeError):
    """A chat completions call failed, before or after its HTTP 200."""


def _raise_for_error_body(body: Any) -> None:
    if isinstance(body, dict) and body.get("error"):
        error = body["error"]
        message = error.get("message") if isinstance(error, dict) else error
        raise DeploymentCallError(f"chat completions returned an error: {message}")


def iter_sse_chunks(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    """Yield each JSON chunk of a server-sent-event stream until ``[DONE]``.

    Blank lines, comments (``:``), and non-``data`` fields are skipped. The
    status is already 200 when the stream starts, so a failure after that is
    raised here as :class:`DeploymentCallError`: an ``error`` chunk, a chunk
    that is not JSON, or a stream that ends without ``[DONE]``.
    """
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith(SSE_DATA_PREFIX):
            continue
        data = line[len(SSE_DATA_PREFIX) :].strip()
        if data == SSE_DONE:
            return
        if not data:
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise DeploymentCallError(
                f"chat completions stream sent a chunk that is not JSON: {data[:ERROR_BODY_LIMIT]}"
            ) from exc
        _raise_for_error_body(chunk)
        yield chunk
    raise DeploymentCallError(f"chat completions stream ended without {SSE_DONE}")


def chunk_text(chunk: dict[str, Any]) -> str:
    """Return the content delta of the first choice in a streamed chunk."""
    choices = chunk.get("choices") or ()
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


@dataclass
class ChatResult:
    """A completed chat call: its text and how long it took."""

    text: str = ""
    # Seconds from sending the request to the first content delta (streaming
    # only), and to the end of the response.
    time_to_first_token: float | None = None
    total_seconds: float = 0.0
    chunks: int = 0
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    body = response.text[:ERROR_BODY_LIMIT]
    if response.status_code == 404:
        raise DeploymentCallError(
            "chat completions returned HTTP 404: inference may not be enabled on this "
            f"account yet, or the deployment is not READY: {body}"
        )
    raise DeploymentCallError(f"chat completions returned HTTP {response.status_code}: {body}")


def stream_chat(
    base_url: str,
    auth: Auth,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float,
    on_delta: Any = None,
    verify_ssl: bool = True,
    timeout: float = 300.0,
    session: requests.Session | None = None,
) -> ChatResult:
    """Stream one chat completion, calling ``on_delta(text)`` for each delta."""
    http = session or requests.Session()
    result = ChatResult()
    deltas: list[str] = []
    started = time.monotonic()
    with _post(
        http,
        f"{base_url.rstrip('/')}/chat/completions",
        auth,
        {"Accept": "text/event-stream"},
        json={
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "stream": True,
            # Adds a final chunk with no choices that carries the token counts.
            "stream_options": {"include_usage": True},
        },
        stream=True,
        verify=verify_ssl,
        timeout=timeout,
    ) as response:
        _raise_for_status(response)
        # Byte lines, decoded as UTF-8 by iter_sse_chunks: requests decodes an
        # event-stream with no charset as Latin-1, and str.splitlines() also
        # splits on U+0085 and U+2028, which can sit unescaped inside a chunk.
        for chunk in iter_sse_chunks(response.iter_lines()):
            result.chunks += 1
            if chunk.get("usage"):
                result.usage = chunk["usage"]
            choices = chunk.get("choices") or ()
            if choices and choices[0].get("finish_reason"):
                result.finish_reason = choices[0]["finish_reason"]
            text = chunk_text(chunk)
            if not text:
                continue
            if result.time_to_first_token is None:
                result.time_to_first_token = time.monotonic() - started
            deltas.append(text)
            if on_delta is not None:
                on_delta(text)
    result.text = "".join(deltas)
    result.total_seconds = time.monotonic() - started
    return result


def complete_chat(
    base_url: str,
    auth: Auth,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float,
    verify_ssl: bool = True,
    timeout: float = 300.0,
    session: requests.Session | None = None,
) -> ChatResult:
    """Run one non-streaming chat completion."""
    http = session or requests.Session()
    started = time.monotonic()
    response = _post(
        http,
        f"{base_url.rstrip('/')}/chat/completions",
        auth,
        {},
        json={
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        },
        verify=verify_ssl,
        timeout=timeout,
    )
    _raise_for_status(response)
    body = response.json()
    _raise_for_error_body(body)
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    return ChatResult(
        text=text,
        total_seconds=time.monotonic() - started,
        finish_reason=choice.get("finish_reason"),
        usage=body.get("usage"),
    )

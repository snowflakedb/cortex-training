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

"""Gate for client APIs that ship before they are supported.

Unfinished client code can merge to main and go out in a release ahead of the
server work it depends on, as long as it is off by default. Customers calling
a gated API get an ``ExperimentalFeatureError``; internal users opt in through
an env var::

    CORTEX_TRAINING_EXPERIMENTAL=tail_logs,custom_code   # named features
    CORTEX_TRAINING_EXPERIMENTAL=all                     # everything

Gate a whole method with the decorator, or one argument / body field with
``require_experimental``::

    @experimental("tail_logs")
    def tail_logs(self, job_id: str) -> Iterator[dict]: ...

    if body.get("custom_code"):
        require_experimental("custom_code", "the create-job `custom_code` field")

This only hides the feature. Anyone can read the source or call the REST API
directly, so a feature that must not be used in production also needs a
server-side gate.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Callable
from typing import Any
from typing import TypeVar

EXPERIMENTAL_ENV = "CORTEX_TRAINING_EXPERIMENTAL"

# Values that enable every experimental feature at once.
_ENABLE_ALL = frozenset({"all", "*"})

# Feature names go in an env var and in error messages, so keep them to one
# unambiguous spelling.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

F = TypeVar("F", bound=Callable[..., Any])


class ExperimentalFeatureError(RuntimeError):
    """Raised when an experimental API is used without opting in."""

    def __init__(self, name: str, what: str):
        self.feature = name
        super().__init__(
            f"{what} is experimental and not supported. To use it anyway "
            f"(internal use only), set {EXPERIMENTAL_ENV}={name}"
        )


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name) or name in _ENABLE_ALL:
        raise ValueError(
            f"invalid experimental feature name {name!r}: use lower_snake_case"
        )


def experimental_enabled(name: str) -> bool:
    """True when ``name`` is opted in via ``CORTEX_TRAINING_EXPERIMENTAL``.

    Read on every call rather than at import, so the env var can be set after
    the package is imported (and tests can toggle it).
    """
    _check_name(name)
    raw = os.environ.get(EXPERIMENTAL_ENV, "")
    enabled = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return name in enabled or not enabled.isdisjoint(_ENABLE_ALL)


def require_experimental(name: str, what: str) -> None:
    """Raise ``ExperimentalFeatureError`` unless ``name`` is opted in.

    ``what`` names the gated thing in the error, e.g. "the `foo` argument".
    """
    if not experimental_enabled(name):
        raise ExperimentalFeatureError(name, what)


def experimental(name: str) -> Callable[[F], F]:
    """Refuse calls to the decorated function unless ``name`` is opted in.

    The check runs when the function is called, before its body, so for a
    generator function it fails at the call rather than on first ``next()``.
    The wrapper carries ``__cortex_training_experimental__ = name`` so tooling
    can find every gated API.
    """
    _check_name(name)

    def decorate(fn: F) -> F:
        what = f"{fn.__qualname__}()"

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            require_experimental(name, what)
            return fn(*args, **kwargs)

        wrapped.__cortex_training_experimental__ = name  # type: ignore[attr-defined]
        wrapped.__doc__ = (
            f"[Experimental: {name}] Not supported; set "
            f"{EXPERIMENTAL_ENV}={name} to enable.\n\n{fn.__doc__ or ''}"
        ).rstrip()
        return wrapped  # type: ignore[return-value]

    return decorate

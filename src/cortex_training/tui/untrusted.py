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

"""The service's job records, made safe to draw where the TUI receives them.

Job records and job lists hold text that users submitted
(a model name, a config value) and text the service builds (ids, statuses,
hardware), and the TUI draws all of it on a terminal. A string that carries an
escape sequence would act there: retitle the window, write the clipboard
(OSC 52), or hide the rest of the screen. So the TUI reads them through
:func:`printable_client`, which returns every string in them, dict keys too,
through :func:`cortex_training.tui.views.printable`: control, bidi and
zero-width characters become U+FFFD before any screen sees them.

Log lines don't go through these reads; the Logs tab draws them itself. Like
``views.py`` this never imports ``textual``.
"""

from __future__ import annotations

import functools
from typing import Any

from cortex_training.tui.views import printable_data

# The client methods whose answers are job records.
_RECORD_READS = frozenset({"get_job", "list_jobs", "list_jobs_page"})


class PrintableClient:
    """A client whose job reads come back printable (see the module docstring).

    Every other attribute and method is the wrapped client's own, read and set
    through to it, and a method the wrapped client lacks is missing here too
    (the job list checks for ``list_jobs_page``). ``for_job`` returns its
    per-location client wrapped the same way. :attr:`wrapped` is the client.
    """

    def __init__(self, client: Any) -> None:
        object.__setattr__(self, "wrapped", client)

    def __getattr__(self, name: str) -> Any:
        if name == "wrapped":  # not set yet (a copy being made): don't recurse
            raise AttributeError(name)
        value = getattr(self.wrapped, name)
        if not callable(value):
            return value
        if name in _RECORD_READS:
            return _printable_answer(value)
        if name == "iter_jobs":
            return _printable_rows(value)
        if name == "for_job":
            return _printable_child(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.wrapped, name, value)

    def __repr__(self) -> str:
        return f"PrintableClient({self.wrapped!r})"


def printable_client(client: Any) -> Any:
    """``client`` wrapped in a :class:`PrintableClient`, once (None stays None)."""
    if client is None or isinstance(client, PrintableClient):
        return client
    return PrintableClient(client)


def unwrapped(client: Any) -> Any:
    """The client a :class:`PrintableClient` wraps; any other client as it is."""
    return client.wrapped if isinstance(client, PrintableClient) else client


# functools.wraps keeps each method's signature for inspect.signature: the job
# list passes retry bounds to list_jobs only when the client's method takes them.
def _printable_answer(method):
    @functools.wraps(method)
    def read(*args, **kwargs):
        return printable_data(method(*args, **kwargs))

    return read


def _printable_rows(method):
    @functools.wraps(method)
    def rows(*args, **kwargs):
        for row in method(*args, **kwargs):
            yield printable_data(row)

    return rows


def _printable_child(method):
    @functools.wraps(method)
    def child(*args, **kwargs):
        return printable_client(method(*args, **kwargs))

    return child

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

"""Export a connection as the ``ARCTIC_CORTEX_*`` variables other tools read.

Tools such as the Arctic Platform launcher (used by the OpenHands, SkyRL and
Harbor recipes) don't use this client: they read a host, database, schema and
programmatic access token (PAT) from the environment. This prints them as shell
``export`` lines for the profile Cortex Training uses, so the shell can set them::

    eval "$(python -m cortex_training.env)"            # bash, zsh
    python -m cortex_training.env | source             # fish
    eval "$(python -m cortex_training.env connection=NAME)"

A program can't set its parent shell's variables, so it prints them for the
shell to run. The output holds the PAT, so it is never written to a terminal:
run on its own, it says how to use it instead. It reads only local files.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from typing import TextIO

from cortex_training.connection import pat_credentials

USAGE = (
    'Prints your PAT, so it is not shown here. Use it as:\n'
    '  eval "$(python -m cortex_training.env)"      # bash, zsh\n'
    '  python -m cortex_training.env | source       # fish\n'
    'Add connection=NAME to use another profile.\n'
)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Write ``export`` lines for the resolved connection; return the exit code."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = list(sys.argv[1:] if argv is None else argv)
    connection = None
    for arg in args:
        if arg.startswith("connection=") and arg.split("=", 1)[1]:
            connection = arg.split("=", 1)[1]
        else:
            stderr.write(f"error: unknown argument {arg!r}\n{USAGE}")
            return 2
    if getattr(stdout, "isatty", lambda: False)():
        stderr.write(USAGE)
        return 2
    try:
        credentials = pat_credentials(connection)
    except Exception as exc:
        stderr.write(f"error: {exc}\n")
        return 1
    # Notes for an old connection form go to stderr, never into the exported lines.
    for note in credentials.notes:
        stderr.write(f"warning: {note}\n")
    values = {
        "ARCTIC_CORTEX_HOST": credentials.host,
        "ARCTIC_CORTEX_DATABASE": credentials.database,
        "ARCTIC_CORTEX_SCHEMA": credentials.schema,
        "ARCTIC_CORTEX_PAT": credentials.pat,
    }
    stdout.write("".join(f"export {name}={shlex.quote(value)}\n" for name, value in values.items()))
    stderr.write(f"Exported ARCTIC_CORTEX_HOST, _DATABASE, _SCHEMA and _PAT from {credentials.source}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

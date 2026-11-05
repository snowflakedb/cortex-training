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

"""Run a command with a connection in the ``ARCTIC_CORTEX_*`` variables other tools read.

Tools such as the Arctic Platform launcher (used by the OpenHands, SkyRL,
Harbor and prime-rl recipes) don't use this client: they read a host,
database, schema and programmatic access token (PAT) from the environment.
This sets them, for the profile Cortex Training uses, in the command it starts::

    python -m cortex_training.env -- bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
    python -m cortex_training.env connection=NAME -- <command and its arguments>

The PAT is only ever in that command's environment: it isn't printed, logged,
or left in your shell. On POSIX the command replaces this process, so it gets
signals such as Ctrl-C directly. It reads only local files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TextIO

from cortex_training.connection import pat_credentials

USAGE = (
    "usage: python -m cortex_training.env [connection=NAME] -- COMMAND [ARG ...]\n"
    "Runs COMMAND with ARCTIC_CORTEX_HOST, _DATABASE, _SCHEMA and _PAT set from the profile\n"
    "Cortex Training uses (or connection=NAME). The PAT is never printed.\n"
)


def _exec(command: Sequence[str], env: Mapping[str, str]) -> int:
    """Run ``command`` with ``env``: replace this process on POSIX, else wait for it."""
    if os.name == "posix":
        os.execvpe(command[0], list(command), dict(env))  # Doesn't return.
    return subprocess.call(list(command), env=dict(env))


def main(
    argv: Sequence[str] | None = None,
    *,
    stderr: TextIO | None = None,
    run: Callable[[Sequence[str], Mapping[str, str]], int] = _exec,
) -> int:
    """Run the command after ``--`` with the connection's variables; return its exit code."""
    stderr = stderr or sys.stderr
    args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in args or args.index("--") == len(args) - 1:
        stderr.write(USAGE)
        return 2
    split = args.index("--")
    options, command = args[:split], args[split + 1 :]
    connection = None
    for arg in options:
        if arg.startswith("connection=") and arg.split("=", 1)[1]:
            connection = arg.split("=", 1)[1]
        else:
            stderr.write(f"error: unknown argument {arg!r}\n{USAGE}")
            return 2
    try:
        credentials = pat_credentials(connection)
    except (ValueError, OSError) as exc:  # The picker's and profile file's errors are ValueErrors.
        stderr.write(f"error: {exc}\n")
        return 1
    # Notes for an old connection form go to stderr, like every command's.
    for note in credentials.notes:
        stderr.write(f"warning: {note}\n")
    env = {
        **os.environ,
        "ARCTIC_CORTEX_HOST": credentials.host,
        "ARCTIC_CORTEX_DATABASE": credentials.database,
        "ARCTIC_CORTEX_SCHEMA": credentials.schema,
        "ARCTIC_CORTEX_PAT": credentials.pat,
    }
    stderr.write(f"Running {command[0]} with ARCTIC_CORTEX_* from {credentials.source}.\n")
    stderr.flush()
    try:
        return run(command, env)
    except FileNotFoundError:
        stderr.write(f"error: command not found: {command[0]}\n")
        return 127


if __name__ == "__main__":
    raise SystemExit(main())

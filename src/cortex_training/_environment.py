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

"""Find shell environment variables that change which connection Cortex Training uses.

A variable left set in a shell (or a shell startup file) can quietly send
commands to another account, file, or database than the one ``cortex-training
login`` set up. This only reads the environment; it never changes it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Overrides the profile that login sets up.
OVERRIDES = "overrides"
# Changes a default, such as which profile or folder is used.
CHANGES = "changes"


@dataclass(frozen=True)
class _Variable:
    name: str
    severity: str
    effect: str
    secret: bool = False


_OLDER_SNOWFLAKE = (
    "An older setting that other Snowflake tools often set. Cortex Training commands use it too, "
    "instead of your profile."
)

_VARIABLES = (
    _Variable(
        "CORTEX_TRAINING_CONFIG",
        OVERRIDES,
        "Makes cortex-training commands use this JSON file instead of the profile you set up here.",
    ),
    _Variable(
        "CORTEX_TRAINING_CONNECTION",
        OVERRIDES,
        "Makes cortex-training commands use this profile instead of the one you set up here.",
    ),
    _Variable(
        "CORTEX_TRAINING_HOST",
        OVERRIDES,
        "Can make commands sign in to this account instead of your profile's (meant for CI).",
    ),
    _Variable(
        "CORTEX_TRAINING_PAT",
        OVERRIDES,
        "Can make commands sign in with this token instead of your profile (meant for CI).",
        secret=True,
    ),
    _Variable(
        "CORTEX_TRAINING_BASE_URL",
        OVERRIDES,
        "Sends cortex-training commands to this server instead of your Snowflake account.",
    ),
    _Variable("SNOWFLAKE_HOST", OVERRIDES, _OLDER_SNOWFLAKE),
    _Variable("SNOWFLAKE_PAT", OVERRIDES, _OLDER_SNOWFLAKE, secret=True),
    _Variable("SNOWFLAKE_DATABASE", OVERRIDES, _OLDER_SNOWFLAKE),
    _Variable("SNOWFLAKE_SCHEMA", OVERRIDES, _OLDER_SNOWFLAKE),
    _Variable(
        "CORTEX_TRAINING_DATABASE",
        CHANGES,
        "Makes Cortex Training use this database instead of your profile's.",
    ),
    _Variable(
        "CORTEX_TRAINING_SCHEMA",
        CHANGES,
        "Makes Cortex Training use this schema instead of your profile's.",
    ),
    _Variable(
        "SNOWFLAKE_DEFAULT_CONNECTION_NAME",
        CHANGES,
        "Changes which profile Snowflake treats as your default, for snow and other tools too.",
    ),
    _Variable(
        "SNOWFLAKE_HOME",
        CHANGES,
        "Changes the folder Snowflake connection profiles, and the choice you save here "
        "(cortex-training/config.json), are read from and saved to. Terminals without it "
        "won't see what you set up here.",
    ),
    _Variable(
        "XDG_CONFIG_HOME",
        CHANGES,
        "Changes where the older login.json (from `cortex-training login PATH`) is looked up.",
    ),
)


@dataclass(frozen=True)
class Finding:
    """One environment variable that changes which connection Cortex Training uses."""

    name: str
    # The value to show; secrets are never shown.
    shown: str
    severity: str
    effect: str


def _shown(value: str, secret: bool) -> str:
    if secret:
        return "(set)"
    home = str(Path.home())
    return "~" + value[len(home):] if value.startswith(home + os.sep) else value


def find(environ: Mapping[str, str] | None = None) -> list[Finding]:
    """Return the variables in ``environ`` that change which connection is used."""
    environ = os.environ if environ is None else environ
    findings = []
    for variable in _VARIABLES:
        value = environ.get(variable.name)
        if value is None or not value.strip():
            continue
        findings.append(Finding(variable.name, _shown(value.strip(), variable.secret), variable.severity, variable.effect))
    return findings


def shell_name(environ: Mapping[str, str] | None = None) -> str:
    """The user's shell (fish, zsh, bash, ...) from ``$SHELL``."""
    environ = os.environ if environ is None else environ
    return Path(environ.get("SHELL", "")).name or "sh"


def clear_command(name: str, shell: str) -> str:
    """The command that clears ``name`` in ``shell``."""
    if shell == "fish":
        # `set -e` alone leaves a universal variable (`set -U`) in place.
        return f"set -e {name}; set -eU {name}"
    return f"unset {name}"


def startup_hint(shell: str) -> str:
    """Where a variable may be set again in every new shell."""
    files = {
        "fish": "~/.config/fish/config.fish",
        "zsh": "~/.zshrc or ~/.zprofile",
        "bash": "~/.bashrc or ~/.bash_profile",
    }
    return files.get(shell, "your shell startup file")

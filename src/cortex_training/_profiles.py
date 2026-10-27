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

"""Read and write Snowflake connection profiles the way the Connector reads them.

The Connector keeps profiles in one folder: ``$SNOWFLAKE_HOME`` (default
``~/.snowflake``) when that folder exists, otherwise the platform's config
folder (for example ``~/.config/snowflake`` on Linux). In that folder:

- If ``connections.toml`` exists, profiles are its top-level ``[name]`` tables,
  and profiles in ``config.toml`` are ignored.
- Otherwise profiles are the ``[connections.name]`` tables in ``config.toml``.

Writing goes to the file the Connector reads, so adding a profile never hides
the ones other Snowflake tools use. Every other profile, setting, and comment
is kept, the file is replaced atomically with mode 600, and the default
profile is never changed.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from collections.abc import Iterator
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import filelock

CONNECTIONS_FILE = "connections.toml"
CONFIG_FILE = "config.toml"
DEFAULT_PROFILE_ENV = "SNOWFLAKE_DEFAULT_CONNECTION_NAME"

# Names this module creates. Existing profiles may use any name the Connector
# accepts (for example ["prod.us"]); those can still be replaced or removed.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ProfileFileError(ValueError):
    """A profile file cannot be read or written."""


def snowflake_dir() -> Path:
    """The Connector's config folder, by the Connector's rule, at call time."""
    home = Path(os.environ.get("SNOWFLAKE_HOME", "~/.snowflake/")).expanduser()
    if home.exists():
        return home
    from platformdirs import PlatformDirs

    return Path(PlatformDirs(appname="snowflake", appauthor=False).user_config_path)


def profile_file() -> Path:
    """Return the file the Connector reads profiles from (and where to write them).

    ``connections.toml`` when it exists; otherwise ``config.toml`` when it already
    holds a ``[connections]`` table; otherwise a new ``connections.toml``.
    """
    folder = snowflake_dir()
    connections = folder / CONNECTIONS_FILE
    if connections.is_file():
        return connections
    config = folder / CONFIG_FILE
    if config.is_file() and isinstance(_parse(config).get("connections"), Mapping):
        return config
    return connections


def _profile_tables(path: Path, document: Any) -> Any:
    """Return the part of ``document`` that holds the profiles, by which file ``path`` is.

    The two files lay profiles out differently, so this is decided explicitly
    from the file name rather than assumed.
    """
    if path.name == CONNECTIONS_FILE:
        # connections.toml: every top-level [name] table is a profile.
        return document
    if path.name == CONFIG_FILE:
        # config.toml: profiles are the [connections.name] tables; other keys
        # (default_connection_name, [cli], ...) are settings, not profiles.
        return document["connections"]
    raise ProfileFileError(f"not a Snowflake profile file: {path}")


def read_profiles() -> dict[str, dict[str, Any]]:
    """Return every profile the Connector can use, by name."""
    path = profile_file()
    if not path.is_file():
        return {}
    tables = _profile_tables(path, _parse(path))
    return {str(name): dict(table) for name, table in tables.items() if isinstance(table, Mapping)}


def default_profile_name() -> str:
    """The profile the Connector uses when none is named.

    ``SNOWFLAKE_DEFAULT_CONNECTION_NAME``, else ``default_connection_name`` in
    ``config.toml``, else ``default``.
    """
    name = (os.environ.get(DEFAULT_PROFILE_ENV) or "").strip()
    if name:
        return name
    config = snowflake_dir() / CONFIG_FILE
    if config.is_file():
        value = _parse(config).get("default_connection_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "default"


def write_profile(name: str, values: Mapping[str, Any]) -> Path:
    """Add or replace the profile ``name`` and return the file written.

    An existing profile is replaced as a whole: keys not in ``values``, and
    comments inside that profile's table, are dropped. Every other profile,
    setting, and comment in the file is kept. ``None`` values are left out, and
    path values are written as strings. A new profile's name must use letters,
    digits, ``_`` or ``-``; an existing profile can have any name.
    """
    if not values:
        raise ProfileFileError(f"profile {name!r} needs at least one setting")
    import tomlkit

    path = profile_file()
    with _locked(path):
        document = _parse(path, keep_format=True) if path.is_file() else tomlkit.document()
        tables = _profile_tables(path, document)
        if name not in tables:
            _check_name(name)
        table = tomlkit.table()
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, os.PathLike):
                value = os.fspath(value)
            try:
                table[key] = value
            except Exception:
                # Never echo the value: it may be a token or password.
                raise ProfileFileError(
                    f"profile {name!r}: can't write {key}: a {type(value).__name__} value can't be stored in TOML"
                ) from None
        tables[name] = table
        _replace(path, tomlkit.dumps(document))
    return path


def remove_profile(name: str) -> bool:
    """Remove the profile ``name``, whatever its name; return whether it existed."""
    import tomlkit

    path = profile_file()
    if not path.is_file():
        return False
    with _locked(path):
        document = _parse(path, keep_format=True)
        tables = _profile_tables(path, document)
        if name not in tables or not isinstance(tables[name], Mapping):
            return False
        del tables[name]
        _replace(path, tomlkit.dumps(document))
    return True


def _check_name(name: str) -> None:
    if not _PROFILE_NAME.match(name or ""):
        raise ProfileFileError(
            f"invalid profile name {name!r}: use letters, digits, '_' or '-'"
        )


def _parse(path: Path, *, keep_format: bool = False) -> Any:
    import tomlkit

    try:
        document = tomlkit.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProfileFileError(f"could not read {path}: {exc}") from exc
    return document if keep_format else document.unwrap()


def _real(path: Path) -> Path:
    """The file ``path`` names, following symlinks, so a linked dotfile stays linked."""
    return Path(os.path.realpath(path))


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock while the file is read, changed, and written.

    Uses a ``.<name>.lock`` file next to the real file, so two writers (two
    ``login`` runs, say) can't lose each other's profiles. ``filelock`` works on
    macOS, Linux, and Windows.
    """
    target = _real(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with filelock.FileLock(target.with_name(f".{target.name}.lock"), mode=0o600):
        yield


def _replace(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically and durably, readable only by the user.

    A symlinked ``path`` keeps its link: the file it points to is replaced. The
    temporary file has a unique name, so concurrent writers never share one.
    """
    target = _real(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            # The file holds every Snowflake credential: make sure it's on disk
            # before it replaces the old one.
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.chmod(target, 0o600)
    _fsync_folder(target.parent)


def _fsync_folder(folder: Path) -> None:
    """Flush the folder entry, so the replace itself survives a crash (POSIX only)."""
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)

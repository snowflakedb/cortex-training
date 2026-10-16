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

"""The saved consoles the TUI keeps for finished jobs: the least recently used go first.

No ``textual`` needed; each test prunes its own cache root.
"""

import os

from cortex_training.tui.log_cache import CONSOLE_MARKER
from cortex_training.tui.log_cache import job_cache_dir
from cortex_training.tui.log_cache import prune_consoles
from cortex_training.tui.log_cache import touch_console


def _console(root, job_id: str, *, size: int, used=None):
    """A kept console of ``size`` bytes, last used at ``used``; without ``used``, still being written (no marker)."""
    console = job_cache_dir(job_id, root=root) / "console"
    console.mkdir(parents=True)
    (console / "stdout.log").write_bytes(b"x" * size)
    if used is not None:
        (console / CONSOLE_MARKER).touch()
        os.utime(console / CONSOLE_MARKER, (used, used))
    return console


def test_the_least_recently_used_go_first_until_the_rest_fit(tmp_path):
    old = _console(tmp_path, "job-old", size=100, used=1_000)
    mid = _console(tmp_path, "job-mid", size=100, used=2_000)
    new = _console(tmp_path, "job-new", size=100, used=3_000)
    assert prune_consoles(root=tmp_path, max_bytes=200) == [old]
    assert not old.exists() and mid.exists() and new.exists()
    assert prune_consoles(root=tmp_path, max_bytes=200) == []


def test_never_the_console_just_kept_nor_one_still_being_written(tmp_path):
    kept = _console(tmp_path, "job-kept", size=100, used=1_000)  # the least recently used
    writing = _console(tmp_path, "job-writing", size=500)
    other = _console(tmp_path, "job-other", size=100, used=2_000)
    assert prune_consoles(root=tmp_path, keep=kept, max_bytes=0) == [other]
    assert kept.exists() and writing.exists()


def test_a_console_used_again_is_kept_longer(tmp_path):
    first = _console(tmp_path, "job-first", size=100, used=1_000)
    second = _console(tmp_path, "job-second", size=100, used=2_000)
    touch_console(first)
    assert prune_consoles(root=tmp_path, max_bytes=100) == [second]
    assert first.exists()


def test_a_cache_without_consoles_has_nothing_to_prune(tmp_path):
    assert prune_consoles(root=tmp_path / "never-written", max_bytes=0) == []

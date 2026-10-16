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

"""``cortex-training tui --debug``'s demo client keeps the real client's call shapes (no ``textual`` needed)."""

import inspect

import pytest

from cortex_training import CortexTrainingClient
from cortex_training.tui.demo import DemoClient

# The client methods the TUI calls.
TUI_CALLS = ("get_job", "list_jobs", "list_jobs_page", "for_job", "tail_logs", "download_metrics",
             "download_stdout_logs", "fetch_execution_logs")
# Parameters of the real client the demo leaves out: the TUI never passes them, or
# checks the client's signature for them first (sub_job_ids, on_chunk, stale_after_seconds).
_ENGINE = {"max_workers", "progress", "on_chunk", "sub_job_ids", "strict", "stale_after_seconds",
           "transfer_timeout_seconds"}
NOT_IN_DEMO = {
    "tail_logs": {"sub_job_type"},
    "download_metrics": _ENGINE,
    "download_stdout_logs": _ENGINE | {"priority", "on_commit"},
}


def _parameters(method, leave_out=frozenset()) -> list:
    """``(name, kind, default)`` of each parameter, in order: annotations may differ."""
    return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()
            if p.name not in leave_out]


@pytest.mark.parametrize("name", TUI_CALLS)
def test_the_demo_client_takes_what_the_real_client_takes(name):
    real = getattr(CortexTrainingClient, name)
    left_out = NOT_IN_DEMO.get(name, set())
    assert left_out <= set(inspect.signature(real).parameters)  # each one left out is still the client's
    assert _parameters(getattr(DemoClient, name)) == _parameters(real, left_out)

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

"""download_stdout_logs / download_metrics without sub_job_ids or priority:
every sub-job, sorted by id."""

import gzip

from tests import test_client as _client_tests

_make_client = _client_tests._make_client
# The stage-SQL fake of the execution-log tests (bound to a name pytest won't collect).
_FakeArtifactConnection = _client_tests.TestExecutionLogDownload.FakeArtifactConnection

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"


def _stage(monkeypatch, artifact_name, chunk_name):
    c = _make_client()
    root = f"_{artifact_name}"
    paths = {
        "job-1:sampling:0": [f"{root}/job-1:sampling:0/{chunk_name}.20260904-12000{i}.s{i}.gz" for i in range(3)],
        "job-1:training:0": [f"{root}/job-1:training:0/{chunk_name}.20260904-12000{i}.t{i}.gz" for i in range(2)],
    }
    listed = [path for chunk_paths in paths.values() for path in chunk_paths]
    connection = _FakeArtifactConnection(
        {RUN_URI + root + "/": [(f"/versions/RUN_ABC/{path}",) for path in listed],
         RUN_URI + "checkpoints/" + root + "/": []},
        {RUN_URI + path: gzip.compress(f"{path}\n".encode()) for path in listed},
    )
    monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (RUN_URI, "RUN_ABC"))
    monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: connection)
    return c, paths, connection


def test_without_the_new_arguments_every_sub_job_comes_sorted(tmp_path, monkeypatch):
    c, _, _ = _stage(monkeypatch, "stdout", "console")
    result = c.download_stdout_logs("job-1", tmp_path)
    assert [r["sub_job_id"] for r in result] == ["job-1:sampling:0", "job-1:training:0"]

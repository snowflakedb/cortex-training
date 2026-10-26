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

"""Dropping a reconstructed artifact's rows before a window (_artifact_download)."""

import json
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from cortex_training import _artifact_download as ad
from cortex_training.client import CortexTrainingClient

START = datetime(2026, 10, 20, 0, 0, 0, tzinfo=timezone.utc)
INTERVAL = 5.0


def _row(index, *, stamp=None, sub="a-sub-job"):
    at = START + timedelta(seconds=index * INTERVAL)
    return json.dumps({
        "record_version": 2,
        "sub_job_id": sub,
        "sample_ts": stamp if stamp is not None else at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_util_pct": float(index % 100),
    }, separators=(",", ":")) + "\n"


def _rows(count, *, first=0):
    """``count`` rows whose stamps continue the sequence from index ``first``."""
    return "".join(_row(index) for index in range(first, first + count)).encode()


def _write(tmp_path, data, *, names=("c0.gz", "c1.gz"), nbytes=None, skipped=None):
    """A reconstructed file and its manifest, as a download leaves them."""
    path = tmp_path / "gpu.jsonl"
    path.write_bytes(data)
    manifest = path.with_name("." + path.name + ".manifest.json")
    ad.write_manifest(manifest, list(names), len(data) if nbytes is None else nbytes, skipped)
    return path, manifest


def test_rows_before_the_window_go_and_the_manifest_count_moves_with_them(tmp_path):
    data = _rows(240)  # 20 minutes at 5 s
    path, manifest = _write(tmp_path, data)

    freed = ad.compact_window(path, manifest, keep_seconds=300)  # keep the last 5 minutes

    assert freed > 0
    kept = path.read_bytes()
    assert data.endswith(kept), "the kept part is the file's own tail, byte for byte"
    assert freed == len(data) - len(kept)
    state = ad.read_manifest(manifest)
    assert state.nbytes == len(kept), "the count must match what the file now holds"
    assert state.names == ["c0.gz", "c1.gz"], "a dropped chunk would be fetched again"
    # Every kept row is inside the window, and the row before the cut was outside it.
    times = [ad.row_time(line, "sample_ts") for line in kept.splitlines()]
    newest = max(times)
    window = 300 + ad._COMPACT_CLOCK_TOLERANCE_S  # opened earlier by the clock tolerance
    assert min(times) >= newest - window
    assert len(times) == int(window / INTERVAL) + 1, times[:3]


def test_the_cut_is_always_a_row_boundary(tmp_path):
    path, manifest = _write(tmp_path, _rows(100))

    ad.compact_window(path, manifest, keep_seconds=100)

    for line in path.read_bytes().splitlines():
        assert json.loads(line)["record_version"] == 2, "a row was cut in half"


def test_a_file_already_inside_the_window_is_left_alone(tmp_path):
    data = _rows(10)
    path, manifest = _write(tmp_path, data)

    assert ad.compact_window(path, manifest, keep_seconds=3600) == 0
    assert path.read_bytes() == data
    assert ad.read_manifest(manifest).nbytes == len(data)


def test_a_file_without_a_manifest_is_left_alone(tmp_path):
    data = _rows(100)
    path = tmp_path / "gpu.jsonl"
    path.write_bytes(data)
    missing = path.with_name("." + path.name + ".manifest.json")

    assert ad.compact_window(path, missing, keep_seconds=10) == 0
    assert path.read_bytes() == data, "a file with nothing to stay consistent with is not touched"


def test_a_file_shorter_than_its_manifest_is_left_alone(tmp_path):
    data = _rows(100)
    path, manifest = _write(tmp_path, data, nbytes=len(data) + 500)

    assert ad.compact_window(path, manifest, keep_seconds=10) == 0
    assert path.read_bytes() == data, "a file already due a rebuild is left for the rebuild"


def test_rows_whose_time_cannot_be_read_do_not_place_a_window(tmp_path):
    data = b"".join(json.dumps({"record_version": 2, "no": "stamp"}).encode() + b"\n" for _ in range(50))
    path, manifest = _write(tmp_path, data)

    assert ad.compact_window(path, manifest, keep_seconds=1) == 0
    assert path.read_bytes() == data


def test_a_nanosecond_fraction_and_a_z_zone_both_read(tmp_path):
    # Both are in the row schema and neither reads with fromisoformat alone on 3.10.
    assert ad.row_time(_row(0, stamp="2026-10-20T12:34:56.123456789Z").encode(), "sample_ts") is not None
    assert ad.row_time(_row(0, stamp="2026-10-20T12:34:56Z").encode(), "sample_ts") is not None
    assert ad.row_time(_row(0, stamp="2026-10-20T12:34:56.5+00:00").encode(), "sample_ts") is not None
    assert ad.row_time(_row(0, stamp="not a time").encode(), "sample_ts") is None
    assert ad.row_time(b"{not json", "sample_ts") is None
    nanos = ad.row_time(_row(0, stamp="2026-10-20T12:34:56.123456789Z").encode(), "sample_ts")
    micros = ad.row_time(_row(0, stamp="2026-10-20T12:34:56.123456Z").encode(), "sample_ts")
    assert nanos == micros, "digits past microseconds are dropped, not misread"


def test_an_uncommitted_tail_goes_and_the_file_is_exactly_its_committed_rows(tmp_path):
    # A killed download can leave the file longer than the manifest says. That tail is not
    # committed (resume truncates it), and kept, it could make a half-done compaction look
    # whole to resume: see the interruption test below.
    committed = _rows(100)
    data = committed + _rows(4, first=100)  # the tail continues the sequence, so a cut happens
    path, manifest = _write(tmp_path, data, nbytes=len(committed))

    freed = ad.compact_window(path, manifest, keep_seconds=100)

    assert freed > 0, "this case must exercise a real cut, or it proves nothing"
    state = ad.read_manifest(manifest)
    assert state.nbytes == len(committed) - freed == path.stat().st_size
    assert path.read_bytes() == committed[freed:]


def test_a_torn_last_line_is_not_cut_into(tmp_path):
    data = _rows(50) + b'{"record_version": 2, "sample_ts": "2026-'
    path, manifest = _write(tmp_path, data, nbytes=len(data))

    ad.compact_window(path, manifest, keep_seconds=50)

    assert path.read_bytes().endswith(b'{"record_version": 2, "sample_ts": "2026-')


def test_resume_appends_after_a_compaction_and_refetches_no_dropped_chunk(tmp_path):
    chunks = ["c0.gz", "c1.gz", "c2.gz"]
    path, manifest = _write(tmp_path, _rows(240), names=chunks)

    ad.compact_window(path, manifest, keep_seconds=300)

    state = CortexTrainingClient._resume_state(path, manifest, chunks)
    assert state is not None, "a compacted file must still be resumable, not rebuilt"
    assert state.names == chunks, "the dropped rows' chunks would be downloaded again"
    assert state.nbytes == path.stat().st_size, "resume would append in the wrong place"


def test_an_interruption_before_the_manifest_is_rewritten_reads_as_a_rebuild(tmp_path):
    # compact_window replaces the data file first. Between that and the manifest write the
    # file is shorter than the manifest says, which must read as "rebuild", never as resume.
    data = _rows(240)
    path, manifest = _write(tmp_path, data)
    stale = manifest.read_bytes()

    ad.compact_window(path, manifest, keep_seconds=300)
    manifest.write_bytes(stale)  # as if the process died before rewriting it

    assert CortexTrainingClient._resume_state(path, manifest, ["c0.gz", "c1.gz"]) is None


def test_the_loader_treats_a_compacted_file_as_present_and_a_half_one_as_absent(tmp_path):
    from cortex_training.tui import resource

    sub = tmp_path / "a-sub-job"
    sub.mkdir()
    data = _rows(240)
    path = sub / "gpu.jsonl"
    path.write_bytes(data)
    manifest = path.with_name("." + path.name + ".manifest.json")
    ad.write_manifest(manifest, ["c0.gz"], len(data))

    ad.compact_window(path, manifest, keep_seconds=300)
    assert resource.cached_metric_sources(tmp_path) == ["a-sub-job"], "the tab lost the sub-job"

    ad.write_manifest(manifest, ["c0.gz"], len(data))  # as an interruption would leave it
    assert resource.cached_metric_sources(tmp_path) == [], "a short file must not be read"


# ─── edge cases: each below once lost rows, broke a resume, or left a file ──

import calendar
import fnmatch
import os
import stat

import pytest


def _epoch(year, month, day, hour, minute, second, *, fraction=0.0, offset_minutes=0):
    """The epoch a stamp means, worked out without datetime.fromisoformat."""
    return calendar.timegm((year, month, day, hour, minute, second)) + fraction - offset_minutes * 60


@pytest.mark.parametrize("stamp, expected", [
    # The documented form (docs/reference/rest-api.md) has no fraction at all.
    ("2026-10-20T12:34:56Z", _epoch(2026, 10, 20, 12, 34, 56)),
    ("2026-10-20T12:34:56.5Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.5)),
    ("2026-10-20T12:34:56.12Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.12)),
    ("2026-10-20T12:34:56.123Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.123)),
    ("2026-10-20T12:34:56.1234Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.1234)),
    ("2026-10-20T12:34:56.12345Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.12345)),
    ("2026-10-20T12:34:56.123456Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.123456)),
    ("2026-10-20T12:34:56.123456789Z", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.123456)),
    ("2026-10-20T12:34:56+05:30", _epoch(2026, 10, 20, 12, 34, 56, offset_minutes=330)),
    ("2026-10-20T12:34:56.5-08:00", _epoch(2026, 10, 20, 12, 34, 56, fraction=0.5, offset_minutes=-480)),
])
def test_every_stamp_form_the_schema_allows_reads_as_the_time_it_means(stamp, expected):
    # Python 3.10, the oldest supported, reads a fraction of exactly 3 or 6 digits and no
    # other; a reader leaning on it returns None for the documented form and compaction
    # then silently never runs there.
    assert ad.row_time(_row(0, stamp=stamp).encode(), "sample_ts") == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("stamp", ["1969-12-31T23:59:59Z", "not a time", "2026-10-20 12:34:56Z", "2026-10-20T12:34:56"])
def test_a_stamp_the_resources_tab_would_not_read_places_no_window(stamp):
    assert ad.row_time(_row(0, stamp=stamp).encode(), "sample_ts") is None


def test_a_row_stamped_ahead_of_this_clock_does_not_move_the_window(tmp_path):
    # One worker with a skewed clock: taking its row as the newest put every real row
    # before the window, and with the chunk names kept they were never fetched again.
    real = _rows(240)
    skewed = _row(0, stamp=(START + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")).encode()
    path, manifest = _write(tmp_path, real + skewed)
    newest_real = START + timedelta(seconds=239 * INTERVAL)

    ad.compact_window(path, manifest, keep_seconds=300, now=newest_real.timestamp())

    kept = path.read_bytes()
    times = [ad.row_time(line, "sample_ts") for line in kept.splitlines()]
    in_window = [t for t in (ad.row_time(l, "sample_ts") for l in real.splitlines())
                 if t >= newest_real.timestamp() - 300]
    assert len(in_window) == 61
    assert all(t in times for t in in_window), "rows inside the window were dropped"


def test_a_committed_length_that_ends_inside_a_row_is_never_cut_past(tmp_path):
    # The cut once counted the row the committed bytes end inside, went past the manifest's
    # count, and wrote a negative one: every chunk fetched again, the sub-job gone meanwhile.
    data = _rows(240)
    lines = data.splitlines(keepends=True)
    committed = sum(len(line) for line in lines[:100]) + 10  # ten bytes into row 100
    path, manifest = _write(tmp_path, data, nbytes=committed)

    freed = ad.compact_window(path, manifest, keep_seconds=100)

    state = ad.read_manifest(manifest)
    assert state is not None, "the manifest no longer reads"
    assert 0 < state.nbytes == committed - freed
    assert freed < committed
    assert path.stat().st_size >= state.nbytes
    assert CortexTrainingClient._resume_state(path, manifest, ["c0.gz", "c1.gz"]) is not None


def test_rows_past_the_committed_bytes_do_not_place_the_window(tmp_path):
    # A later chunk landed but was never checkpointed: taking its rows as the newest put every
    # committed row before the window, and resume then truncated what was left.
    committed = _rows(240)
    later = _rows(5, first=10_000)
    path, manifest = _write(tmp_path, committed + later, nbytes=len(committed))

    ad.compact_window(path, manifest, keep_seconds=300)

    state = ad.read_manifest(manifest)
    kept_committed = path.read_bytes()[: state.nbytes]
    expected = int((300 + ad._COMPACT_CLOCK_TOLERANCE_S) / INTERVAL) + 1
    assert len(kept_committed.splitlines()) == expected, "the window was placed by uncommitted rows"


def test_a_compaction_cut_short_leaves_a_file_the_next_download_sweeps(tmp_path, monkeypatch):
    path, manifest = _write(tmp_path, _rows(240))
    seen = []
    real_replace = ad.os.replace

    def replace(src, dst):
        seen.extend(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp"))
        return real_replace(src, dst)

    monkeypatch.setattr(ad.os, "replace", replace)
    ad.compact_window(path, manifest, keep_seconds=300)

    temps = [name for name in seen if name.startswith(".gpu.jsonl.compact-")]
    assert temps, seen
    leftover = tmp_path / temps[0]
    leftover.write_bytes(b"what a killed compaction left")
    old = 0  # long before the sweep's age limit
    os.utime(leftover, (old, old))
    CortexTrainingClient._remove_orphan_temp_files(path, ".gpu-metrics-")
    assert not leftover.exists(), "a killed compaction's copy would stay in the directory forever"
    assert all(fnmatch.fnmatch(name, f".{path.name}.compact-*.tmp") for name in temps)


def test_compaction_keeps_the_file_s_mode(tmp_path):
    path, manifest = _write(tmp_path, _rows(240))
    os.chmod(path, 0o644)

    assert ad.compact_window(path, manifest, keep_seconds=300) > 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_a_directory_that_takes_no_new_file_leaves_everything_as_it_was(tmp_path, monkeypatch):
    data = _rows(240)
    path, manifest = _write(tmp_path, data)
    before = manifest.read_bytes()

    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ad.tempfile, "mkstemp", no_space)
    assert ad.compact_window(path, manifest, keep_seconds=300) == 0
    assert path.read_bytes() == data and manifest.read_bytes() == before


def test_an_oversized_row_stops_the_cut_instead_of_being_read_whole(tmp_path):
    head = _rows(50)
    huge = b'{"record_version":2,"pad":"' + b"x" * (2 << 20) + b'"}\n'
    tail = _rows(190, first=50)
    path, manifest = _write(tmp_path, head + huge + tail)

    freed = ad.compact_window(path, manifest, keep_seconds=60)

    assert freed <= len(head), "the cut went past a row it never read"


def test_an_interruption_before_the_manifest_is_rewritten_reads_as_a_rebuild_even_with_a_tail(tmp_path):
    # With a killed download's tail kept, the replaced file could still be as long as the
    # old count: resume then accepted the pair and truncated rows that had moved by the cut.
    committed = _rows(240)
    tail = _rows(400, first=240)  # longer than the cut below
    path, manifest = _write(tmp_path, committed + tail, nbytes=len(committed))
    stale = manifest.read_bytes()

    freed = ad.compact_window(path, manifest, keep_seconds=300)
    assert 0 < freed < len(tail), "this case needs a tail longer than the cut"
    manifest.write_bytes(stale)  # as if the process died before the manifest was rewritten

    assert CortexTrainingClient._resume_state(path, manifest, ["c0.gz", "c1.gz"]) is None


def test_the_data_is_replaced_and_made_durable_before_the_manifest_moves(tmp_path, monkeypatch):
    path, manifest = _write(tmp_path, _rows(240))
    order = []
    real_replace, real_fsync_dir, real_write = ad.os.replace, ad._fsync_directory, ad.write_manifest
    monkeypatch.setattr(ad.os, "replace", lambda src, dst: (order.append(("replace", Path(dst).name)),
                                                            real_replace(src, dst))[1])
    monkeypatch.setattr(ad, "_fsync_directory", lambda d: (order.append(("fsync", d)), real_fsync_dir(d))[1])
    monkeypatch.setattr(ad, "write_manifest", lambda *a, **k: (order.append(("manifest",)), real_write(*a, **k))[1])

    assert ad.compact_window(path, manifest, keep_seconds=300) > 0
    assert [step[0] for step in order[:3]] == ["replace", "fsync", "manifest"], order
    assert order[0][1] == path.name and order[1][1] == path.parent


def test_a_worker_whose_clock_ran_ahead_on_a_finished_job_costs_no_row_in_the_window(tmp_path):
    # The job ended days ago, one worker six hours fast: against this computer's clock that
    # worker's rows looked real, and the window moved six hours past every other row.
    real = _rows(240)
    end = START + timedelta(seconds=239 * INTERVAL)
    fast = "".join(_row(0, stamp=(START + timedelta(hours=6, seconds=k * INTERVAL)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        sub="a-fast-worker") for k in range(10)).encode()
    path, manifest = _write(tmp_path, real + fast)

    ad.compact_window(path, manifest, keep_seconds=300, now=end.timestamp())

    kept = {ad.row_time(line, "sample_ts") for line in path.read_bytes().splitlines()}
    in_window = [t for t in (ad.row_time(l, "sample_ts") for l in real.splitlines()) if t >= end.timestamp() - 300]
    assert in_window and all(t in kept for t in in_window), "rows inside the window were dropped"


def test_a_worker_a_little_ahead_on_a_running_job_costs_no_row_in_the_window(tmp_path):
    real = _rows(240)
    newest_real = START + timedelta(seconds=239 * INTERVAL)
    ahead = _row(0, stamp=(newest_real + timedelta(seconds=50)).strftime("%Y-%m-%dT%H:%M:%SZ")).encode()
    path, manifest = _write(tmp_path, real + ahead)

    ad.compact_window(path, manifest, keep_seconds=300, now=newest_real.timestamp())

    kept = {ad.row_time(line, "sample_ts") for line in path.read_bytes().splitlines()}
    in_window = [t for t in (ad.row_time(l, "sample_ts") for l in real.splitlines())
                 if t >= newest_real.timestamp() - 300]
    assert all(t in kept for t in in_window), "a row 50 s ahead cost rows inside the window"


def test_a_far_future_row_near_the_start_does_not_stop_compaction_for_good(tmp_path):
    head = _row(0, stamp="2999-01-01T00:00:00Z").encode()
    path, manifest = _write(tmp_path, head + _rows(240))

    freed = ad.compact_window(path, manifest, keep_seconds=300)

    assert freed > len(head), "one far-future row kept everything after it"


@pytest.mark.parametrize("keep", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_a_window_that_is_not_a_finite_length_is_refused(tmp_path, keep):
    path, manifest = _write(tmp_path, _rows(240))
    with pytest.raises(ValueError):
        ad.compact_window(path, manifest, keep_seconds=keep)


def test_a_file_its_owner_made_read_only_is_left_alone(tmp_path):
    data = _rows(240)
    path, manifest = _write(tmp_path, data)
    os.chmod(path, 0o444)
    try:
        assert ad.compact_window(path, manifest, keep_seconds=300) == 0
        assert path.read_bytes() == data
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")], "a copy was left behind"
    finally:
        os.chmod(path, 0o644)

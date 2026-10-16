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

"""Unit tests for the TUI's pure formatting helpers (no ``textual`` needed)."""

import pytest

from cortex_training.tui.format import entry_at_level
from cortex_training.tui.format import entry_matches
from cortex_training.tui.format import format_created_at
from cortex_training.tui.format import format_event
from cortex_training.tui.format import format_job_config
from cortex_training.tui.format import format_job_row
from cortex_training.tui.format import format_job_summary
from cortex_training.tui.format import format_log_entry
from cortex_training.tui.format import job_matches
from cortex_training.tui.format import job_status_color
from cortex_training.tui.format import sort_jobs
from cortex_training.tui.format import source_label
from cortex_training.tui.format import wrap_log_line


def test_format_structured_log_entry():
    line = format_log_entry(
        {"ts": "2026-06-12T00:00:01Z", "level": "INFO", "logger": "cortex_training.zones", "msg": "hello"}
    )
    assert "2026-06-12T00:00:01Z" in line
    assert "INFO" in line
    assert "cortex_training.zones:" in line
    assert line.endswith("hello")


def test_format_raw_log_entry():
    assert format_log_entry({"_raw": "plain text"}) == "plain text"


def test_format_truncated_entry_marks_suffix():
    out = format_log_entry({"_raw": "big line", "_truncated": True})
    assert out.endswith("[truncated]")


def test_format_entry_missing_msg_falls_back_to_json():
    out = format_log_entry({"foo": "bar"})
    assert "foo" in out and "bar" in out  # nothing silently dropped


def test_format_non_dict_entry():
    assert format_log_entry("already a string") == "already a string"


def test_format_event_full():
    out = format_event(
        {
            "ts": "2026-06-12T00:01:00Z",
            "kind": "zone_transition",
            "zone": "training-1",
            "from": "creating",
            "to": "ready",
            "sub_job_id": "s:training:0",
            "detail": "zone_manager=zm-1",
        }
    )
    assert "zone_transition" in out
    assert "zone=training-1" in out
    assert "creating→ready" in out
    assert "sub_job=s:training:0" in out
    assert "zone_manager=zm-1" in out


def test_format_event_minimal():
    assert format_event({"kind": "route_added"}) == "route_added"


def test_source_label_names_the_component():
    # Sources are sub-jobs: "<type> #<index>", type from job_type or the id.
    assert source_label({"sub_job_id": "u:training:0", "job_type": "training"}) == "training #0"
    assert source_label({"sub_job_id": "u:sampling:1", "job_type": "sampling"}) == "sampling #1"
    assert source_label({"sub_job_id": "u:training:2"}) == "training #2"  # type from id
    assert source_label("u:training:0") == "training #0"  # bare string id


def test_source_label_falls_back_to_server_label_then_id():
    assert source_label({"source_id": "weird", "label": "Custom"}) == "Custom"
    assert source_label({"source_id": "weird"}) == "weird"


def test_format_job_row():
    row = format_job_row(
        {
            "job_id": "abc-123",
            "status": "RUNNING",
            "created_at": "2026-06-13T18:49:54Z",
            "sub_jobs": [{"job_type": "training"}, {"job_type": "sampling"}],
        }
    )
    assert "RUNNING" in row and "abc-123" in row and "training" in row and "sampling" in row
    assert "2026-06-1" in row  # created date shown (tz may shift the day by ±1)
    assert "│" in row  # column separators present
    # JOB_STATE_ prefix stripped; type shown; missing created → placeholder column
    no_ts = format_job_row({"id": "j", "status": "JOB_STATE_FAILED", "job_type": "training"})
    assert no_ts.startswith("FAILED") and "training" in no_ts and "│" in no_ts and "—" in no_ts
    assert format_job_row({"job_id": "j"}).startswith("?")  # missing status


def test_job_status_color():
    # Subtle: only live jobs get a hue (green); everything else stays default.
    assert job_status_color("RUNNING") == "green"
    assert job_status_color("JOB_STATE_PENDING") == "green"
    assert job_status_color("FAILED") == ""
    assert job_status_color("CANCELLED") == ""
    assert job_status_color("TERMINATED") == ""
    assert job_status_color(None) == ""
    assert job_status_color("weird") == ""


def test_format_created_at():
    import re

    out = format_created_at("2026-06-13T18:49:54Z")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} \S+", out)  # local YYYY-MM-DD HH:MM TZ
    assert format_created_at(None) == ""
    assert format_created_at("") == ""
    assert format_created_at("not-a-date") == ""


def test_sort_jobs_newest_first():
    jobs = [
        {"job_id": "old", "created_at": "2026-06-10T00:00:00Z"},
        {"job_id": "new", "created_at": "2026-06-13T00:00:00Z"},
        {"job_id": "mid", "created_at": "2026-06-11T00:00:00Z"},
        {"job_id": "none"},  # missing created_at -> sorts last
    ]
    order = [j["job_id"] for j in sort_jobs(jobs)]
    assert order == ["new", "mid", "old", "none"]
    assert sort_jobs("not a list") == []


def test_sort_jobs_stable_on_equal_timestamps():
    jobs = [
        {"job_id": "first", "created_at": "2026-06-12T00:00:00Z"},
        {"job_id": "second", "created_at": "2026-06-12T00:00:00Z"},
    ]
    # equal created_at keeps original order (stable)
    assert [j["job_id"] for j in sort_jobs(jobs)] == ["first", "second"]


def test_format_job_summary():
    job = {
        "status": "RUNNING",
        "sub_jobs": [
            {
                "job_type": "TRAINING",
                "model_name": "Qwen/Qwen3-0.6B",
                "training_config": {
                    "n_gpus": 1.0,
                    "max_seq_len": 128.0,
                    "train_batch_size": 1.0,
                    "optimizer": {"name": "adamw", "lr": 1e-5},
                },
            }
        ],
    }
    s = format_job_summary(job)
    assert s.startswith("RUNNING")
    assert "training" in s and "Qwen/Qwen3-0.6B" in s
    assert "gpus=1" in s and "seq=128" in s and "bs=1" in s
    assert "opt=adamw/lr=1e-05" in s
    assert "gbs=" not in s and "clip=" not in s  # only what was passed
    # failure reason is appended
    failed = {"status": "FAILED", "reason": "oom killed", "sub_jobs": []}
    assert format_job_summary(failed).endswith("reason: oom killed")
    assert format_job_summary("not a dict") == ""


def test_format_job_config_is_config_only():
    # The reserved config line shows static params only — no status, no reason.
    job = {
        "status": "FAILED",
        "reason": "oom killed",
        "sub_jobs": [
            {
                "job_type": "TRAINING",
                "model_name": "Qwen/Qwen3-0.6B",
                "training_config": {
                    "n_gpus": 1.0,
                    "max_seq_len": 128.0,
                    "train_batch_size": 1.0,
                    "optimizer": {"name": "adamw", "lr": 1e-5},
                },
            }
        ],
    }
    s = format_job_config(job)
    assert "training" in s and "Qwen/Qwen3-0.6B" in s
    assert "gpus=1" in s and "seq=128" in s and "opt=adamw/lr=1e-05" in s
    assert "FAILED" not in s and "reason" not in s  # status/reason live on the subtitle
    assert format_job_config("not a dict") == ""


def test_job_matches():
    job = {"job_id": "abc-123", "status": "RUNNING", "sub_jobs": [{"job_type": "training", "model_name": "Qwen3"}]}
    assert job_matches(job, "") is True
    assert job_matches(job, "abc") and job_matches(job, "run")
    assert job_matches(job, "qwen") and job_matches(job, "training")
    assert not job_matches(job, "sampling")
    assert job_matches("not a dict", "q") is False


def test_entry_matches():
    assert entry_matches({"_raw": "loss=0.5 step=3"}, "loss")
    assert entry_matches({"msg": "hello", "level": "INFO"}, "hello")
    assert not entry_matches({"_raw": "x"}, "loss")
    assert entry_matches({"_raw": "x"}, "") is True


def test_entry_at_level():
    assert entry_at_level({"level": "ERROR"}, "WARNING")
    assert not entry_at_level({"level": "INFO"}, "WARNING")
    assert entry_at_level({"_raw": "x"}, "ERROR")  # unleveled lines always pass
    assert entry_at_level({"level": "INFO"}, None)  # no floor → all pass
    assert entry_at_level({"level": "INFO"}, "BOGUS")  # bad floor → all pass


def test_wrap_log_line_short_passes_through():
    assert wrap_log_line("hello world", 80) == ["hello world"]


def test_wrap_log_line_zero_or_negative_width_unwrapped():
    # Not laid out yet (width 0/neg): never wrap, never drop the line.
    assert wrap_log_line("a long line of text", 0) == ["a long line of text"]
    assert wrap_log_line("x" * 50, -1) == ["x" * 50]


def test_wrap_log_line_hard_breaks_long_token_without_loss():
    token = "abcdefghij" * 5  # 50 chars, no spaces (e.g. a URL or hash)
    pieces = wrap_log_line(token, 10)
    assert all(len(p) <= 10 for p in pieces)
    assert "".join(pieces) == token  # hard break loses nothing
    assert len(pieces) == 5


def test_wrap_log_line_word_wraps_within_width():
    pieces = wrap_log_line("the quick brown fox jumps over", 10)
    assert all(len(p) <= 10 for p in pieces)
    assert len(pieces) >= 3
    # every non-space character is preserved (only break points differ)
    assert "".join(pieces).replace(" ", "") == "thequickbrownfoxjumpsover"


def test_wrap_log_line_splits_embedded_newlines():
    assert wrap_log_line("a\nb\nc", 80) == ["a", "b", "c"]


def test_wrap_log_line_empty_is_one_blank_line():
    assert wrap_log_line("", 80) == [""]
    assert wrap_log_line("\n", 80) == ["", ""]  # blank lines preserved


# ─── Live tail lines: {"_raw": "<runtime time> <line>"} ──────────────────────

from cortex_training.tui.format import entry_time  # noqa: E402
from cortex_training.tui.format import line_level  # noqa: E402
from cortex_training.tui.format import split_runtime_time  # noqa: E402

_JSON_LINE = '{"ts": "2026-01-01T00:00:00.1+00:00", "level": "ERROR", "logger": "x", "msg": "boom"}'
_VLLM_LINE = "\x1b[36m(InferenceWorker pid=9, ip=10.0.0.1)\x1b[0m (EngineCore pid=7) WARNING 01-01 00:00:00 [a.py:1] slow"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-01-01T00:00:00.329214070Z hello", ("2026-01-01T00:00:00.329214070Z", "hello")),
        ("2026-01-01T00:00:00Z  indented", ("2026-01-01T00:00:00Z", " indented")),
        ("2026-01-01T00:00:00+02:00 x", ("2026-01-01T00:00:00+02:00", "x")),
        ("2026-01-01T00:00:00.1Z ", ("2026-01-01T00:00:00.1Z", "")),
        ("no time here", (None, "no time here")),
        ("2026-01-01 00:00:00 not runtime format", (None, "2026-01-01 00:00:00 not runtime format")),
    ],
)
def test_split_runtime_time(raw, expected):
    assert split_runtime_time(raw) == expected


def test_live_raw_line_renders_like_the_saved_console():
    for line in [_JSON_LINE, _VLLM_LINE, "plain", ""]:
        entry = {"_raw": f"2026-01-01T00:00:00.123456789Z {line}"}
        assert format_log_entry(entry) == line
        assert entry_time(entry) == "2026-01-01T00:00:00.123456789Z"
    assert format_log_entry({"_raw": "2026-01-01T00:00:00Z cut", "_truncated": True}) == "cut …[truncated]"
    # A line without a runtime time is shown unchanged.
    assert format_log_entry({"_raw": "x 2026-01-01T00:00:00Z y"}) == "x 2026-01-01T00:00:00Z y"


@pytest.mark.parametrize(
    "line, level",
    [
        (_JSON_LINE, "ERROR"),
        ('{"levelname": "WARNING", "message": "m"}', "WARNING"),
        ('{"msg": "no level"}', None),
        ('{"level": 3, "msg": "game level"}', None),
        ('{"level": "high", "msg": "priority"}', None),
        ('{"level": "warning", "msg": "lower-case"}', "WARNING"),
        ('{"not json', None),
        (_VLLM_LINE, "WARNING"),
        ("INFO 01-01 00:00:00 [a.py:1] ok", "INFO"),
        ("ERROR: something", "ERROR"),
        ("Information about INFO", None),
        ("warning lowercase is not a level", None),
        ("", None),
    ],
)
def test_line_level(line, level):
    assert line_level(line) == level


def test_level_filter_applies_to_live_raw_lines():
    stamp = "2026-01-01T00:00:00Z "
    error = {"_raw": stamp + _JSON_LINE}
    warning = {"_raw": stamp + _VLLM_LINE}
    info = {"_raw": stamp + '{"level": "INFO", "msg": "drain_metrics"}'}
    plain = {"_raw": stamp + "Traceback (most recent call last):"}
    entries = [error, warning, info, plain]
    assert [e for e in entries if entry_at_level(e, "ERROR")] == [error, plain]
    assert [e for e in entries if entry_at_level(e, "WARNING")] == [error, warning, plain]
    assert all(entry_at_level(e, None) for e in entries)
    # An explicit level field still wins.
    assert not entry_at_level({"_raw": stamp + _JSON_LINE, "level": "DEBUG"}, "INFO")


def test_filter_matches_the_line_not_the_runtime_time():
    entry = {"_raw": "2026-01-01T00:00:00Z payload"}
    assert entry_matches(entry, "payload")
    assert not entry_matches(entry, "2026-01-01T00")


def test_format_created_at_names_the_time_zone(monkeypatch):
    import time as time_module

    monkeypatch.setenv("TZ", "UTC")
    time_module.tzset()
    try:
        assert format_created_at("2026-06-13T18:49:54Z") == "2026-06-13 18:49 UTC"
    finally:
        monkeypatch.undo()
        time_module.tzset()


@pytest.mark.parametrize(
    "nbytes, text",
    [(None, "—"), (0, "0 B"), (1023, "1023 B"), (1024, "1.0 KiB"), (1153433, "1.1 MiB"),
     (5 * 1024**3, "5.0 GiB"), (3 * 1024**4, "3.0 TiB")],
)
def test_format_size(nbytes, text):
    from cortex_training.tui.format import format_size

    assert format_size(nbytes) == text


@pytest.mark.parametrize(
    "seconds, text",
    [(-3, "0s"), (0, "0s"), (59, "59s"), (60, "1m 0s"), (725, "12m 5s"), (3600, "1h 0m"),
     (7380, "2h 3m"), (184694, "2d 3h")],
)
def test_format_age(seconds, text):
    from cortex_training.tui.format import format_age

    assert format_age(seconds) == text


def test_local_clock_has_a_time_zone(monkeypatch):
    import time as time_module

    from cortex_training.tui.format import local_clock

    monkeypatch.setenv("TZ", "UTC")
    time_module.tzset()
    try:
        assert local_clock(0) == "00:00:00 UTC"
    finally:
        monkeypatch.undo()
        time_module.tzset()


def test_a_level_field_that_is_not_a_logging_level_never_hides_the_line():
    stamp = "2026-01-01T00:00:00Z "
    for line in ['{"level": 3, "msg": "m"}', '{"level": "high", "msg": "m"}', '{"level": null, "msg": "m"}']:
        assert entry_at_level({"_raw": stamp + line}, "ERROR")
    assert entry_at_level({"level": "high", "msg": "structured"}, "ERROR")
    assert entry_at_level({"level": 7, "msg": "structured"}, "ERROR")
    assert not entry_at_level({"level": "info", "msg": "structured"}, "ERROR")

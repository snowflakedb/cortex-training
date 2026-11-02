from __future__ import annotations

import io

import pytest

from cortex_training import _environment
from cortex_training import _login


def test_nothing_set_means_no_findings():
    assert _environment.find({"PATH": "/usr/bin", "CORTEX_TRAINING_CONFIG": "  "}) == []


def test_findings_name_each_variable_and_never_show_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    found = _environment.find(
        {
            "CORTEX_TRAINING_CONFIG": f"{tmp_path}/.config/cortex-training/trial-config.json",
            "CORTEX_TRAINING_PAT": "super-secret",
            "SNOWFLAKE_PAT": "also-secret",
            "SNOWFLAKE_DEFAULT_CONNECTION_NAME": "my_profile",
        }
    )
    assert [(f.name, f.shown, f.severity) for f in found] == [
        ("CORTEX_TRAINING_CONFIG", "~/.config/cortex-training/trial-config.json", _environment.OVERRIDES),
        ("CORTEX_TRAINING_PAT", "(set)", _environment.OVERRIDES),
        ("SNOWFLAKE_PAT", "(set)", _environment.OVERRIDES),
        ("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "my_profile", _environment.CHANGES),
    ]
    assert "secret" not in repr(found)


@pytest.mark.parametrize(
    ("name", "severity"),
    [
        ("CORTEX_TRAINING_CONNECTION", _environment.OVERRIDES),
        ("CORTEX_TRAINING_HOST", _environment.OVERRIDES),
        ("CORTEX_TRAINING_BASE_URL", _environment.OVERRIDES),
        ("SNOWFLAKE_HOST", _environment.OVERRIDES),
        ("SNOWFLAKE_DATABASE", _environment.OVERRIDES),
        ("SNOWFLAKE_SCHEMA", _environment.OVERRIDES),
        ("CORTEX_TRAINING_DATABASE", _environment.CHANGES),
        ("CORTEX_TRAINING_SCHEMA", _environment.CHANGES),
        ("SNOWFLAKE_HOME", _environment.CHANGES),
        ("XDG_CONFIG_HOME", _environment.CHANGES),
    ],
)
def test_each_variable_is_recognized(name, severity):
    (finding,) = _environment.find({name: "value"})
    assert (finding.name, finding.severity) == (name, severity)
    assert finding.effect


@pytest.mark.parametrize(
    ("shell", "command"),
    [
        ("fish", "set -e CORTEX_TRAINING_CONFIG; set -eU CORTEX_TRAINING_CONFIG"),
        ("zsh", "unset CORTEX_TRAINING_CONFIG"),
        ("bash", "unset CORTEX_TRAINING_CONFIG"),
    ],
)
def test_clear_command_matches_the_shell(shell, command):
    assert _environment.clear_command("CORTEX_TRAINING_CONFIG", shell) == command


def test_shell_name_and_startup_file():
    assert _environment.shell_name({"SHELL": "/opt/homebrew/bin/fish"}) == "fish"
    assert _environment.shell_name({}) == "sh"
    assert _environment.startup_hint("fish") == "~/.config/fish/config.fish"
    assert _environment.startup_hint("zsh") == "~/.zshrc or ~/.zprofile"


def _report(findings, shell="fish", remind=False):
    stderr = io.StringIO()
    prompter = _login._Prompter(io.StringIO(), stderr)
    (_login.remind_environment if remind else _login.report_environment)(prompter, findings, shell)
    return stderr.getvalue()


def test_the_report_says_nothing_overrides_when_clean():
    assert "✓ Nothing in your shell overrides it." in _report([])


def test_the_report_explains_an_override_and_how_to_clear_it():
    out = _report(_environment.find({"CORTEX_TRAINING_CONFIG": "/tmp/trial-config.json", "SNOWFLAKE_HOME": "/tmp/sf"}))
    assert "✗ CORTEX_TRAINING_CONFIG = /tmp/trial-config.json" in out
    assert "use this JSON file instead of the profile you set up here" in out
    assert "To clear it in this shell:  set -e CORTEX_TRAINING_CONFIG; set -eU CORTEX_TRAINING_CONFIG" in out
    assert "! SNOWFLAKE_HOME = /tmp/sf" in out
    assert "To clear it in this shell:  set -e SNOWFLAKE_HOME" not in out  # only overrides get a clear command
    assert "You can continue now and clear them afterwards." in out


def test_the_reminder_repeats_only_overrides():
    out = _report(
        _environment.find({"CORTEX_TRAINING_CONFIG": "/tmp/c.json", "XDG_CONFIG_HOME": "/tmp/x"}), shell="zsh", remind=True
    )
    assert "Heads-up: your shell still overrides this setup." in out
    assert "CORTEX_TRAINING_CONFIG  clear it with:  unset CORTEX_TRAINING_CONFIG" in out
    assert "XDG_CONFIG_HOME" not in out
    assert "remove it from ~/.zshrc or ~/.zprofile" in out
    assert _report(_environment.find({"XDG_CONFIG_HOME": "/tmp/x"}), remind=True) == ""

"""The agent skill that ships with the package: in step with the CLI, installable, and in the wheel."""

from __future__ import annotations

import io
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cortex_training import __version__
from cortex_training import _skills
from cortex_training.cli import commands as cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = REPO_ROOT / "src" / "cortex_training" / "skills"
SKILL_FILES = sorted(SKILLS.glob("*/SKILL.md"))
SKILL_TEXT = sorted(path for path in SKILLS.rglob("*.md"))


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path} has no frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields


def test_there_is_a_skill():
    assert [path.parent.name for path in SKILL_FILES] == ["cortex-training-setup"]


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_each_skill_has_frontmatter_and_stays_short(path):
    fields = _frontmatter(path)
    assert fields["name"] == path.parent.name
    assert len(fields["description"]) > 100
    assert len(path.read_text(encoding="utf-8").splitlines()) < 500


def _commands(text: str) -> list[str]:
    """Every `cortex-training ...` command, inline or on its own line in a code block."""
    inline = re.findall(r"`(cortex-training(?: [^`]*)?)`", text)
    blocks = [line.strip() for line in text.splitlines() if line.strip().startswith("cortex-training")]
    return sorted(set(inline + blocks))


@pytest.mark.parametrize("path", SKILL_TEXT, ids=lambda p: str(p.relative_to(SKILLS)))
def test_every_command_a_skill_shows_is_one_the_cli_accepts(path):
    commands = _commands(path.read_text(encoding="utf-8"))
    parser = cli.build_parser(include_tui=True)
    for command in commands:
        argv = shlex.split(command)[1:]
        if not argv:
            continue
        try:
            parser.parse_args(argv)
        except SystemExit as exc:  # --version and --help exit 0; anything else is a stale command.
            assert exc.code == 0, f"{path.name}: `{command}` isn't a valid cortex-training command"


@pytest.mark.parametrize("path", SKILL_TEXT, ids=lambda p: str(p.relative_to(SKILLS)))
def test_skills_dont_point_at_issues_or_pull_requests(path):
    # Users only have the CLI and these files: no "#123" history, which means nothing to them.
    assert not re.search(r"(?<![\w/])#\d+\b", path.read_text(encoding="utf-8"))



# ─── cortex-training skills install, and the out-of-date note ───────────────


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def _run(argv, monkeypatch, tmp_path, tty=False):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    stdout, stderr = io.StringIO(), (_Terminal() if tty else io.StringIO())
    code = cli.main(argv, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_install_copies_the_skill_and_records_the_version(monkeypatch, tmp_path):
    code, out, err = _run(["--compact", "skills", "install"], monkeypatch, tmp_path)
    target = tmp_path / "home" / ".claude" / "skills" / "cortex-training-setup"
    assert code == 0, err
    assert sorted(p.name for p in target.iterdir()) == [".installed-version", "SKILL.md", "troubleshooting.md"]
    assert (target / ".installed-version").read_text() == f"{__version__}\n"
    assert (target / "SKILL.md").read_bytes() == (SKILLS / "cortex-training-setup" / "SKILL.md").read_bytes()
    assert f"Installed the cortex-training-setup agent skill (cortex-training {__version__}) in {target}." in err
    assert '"replaced":false' in out


def test_install_again_replaces_its_own_copy_and_dir_picks_the_folder(monkeypatch, tmp_path):
    root = tmp_path / "agent-skills"
    _run(["skills", "install", "--dir", str(root)], monkeypatch, tmp_path)
    (root / "cortex-training-setup" / "stale.md").write_text("from an older version")
    code, out, err = _run(["--compact", "skills", "install", "--dir", str(root)], monkeypatch, tmp_path)
    assert code == 0 and "Updated the cortex-training-setup agent skill" in err and '"replaced":true' in out
    assert not (root / "cortex-training-setup" / "stale.md").exists()


def test_install_leaves_a_folder_that_isnt_this_skill_alone(monkeypatch, tmp_path):
    other = tmp_path / "agent-skills" / "cortex-training-setup"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("---\nname: something-else\n---\nmine\n")
    code, _, err = _run(["skills", "install", "--dir", str(other.parent)], monkeypatch, tmp_path)
    assert code == 1 and "isn't the cortex-training-setup skill, so it was left alone" in err
    assert (other / "SKILL.md").read_text().endswith("mine\n")


def test_the_note_appears_only_for_an_installed_out_of_date_skill(monkeypatch, tmp_path):
    root = tmp_path / "home" / ".claude" / "skills"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _skills.update_note() is None  # not installed: nothing to say
    _skills.install()
    assert _skills.update_note() is None  # current
    (root / "cortex-training-setup" / ".installed-version").write_text("0.0.9\n")
    assert _skills.update_note() is None  # the version alone doesn't make it out of date
    with (root / "cortex-training-setup" / "SKILL.md").open("a") as handle:
        handle.write("an older line\n")
    assert _skills.update_note() == (
        f"note: your cortex-training-setup agent skill is from cortex-training 0.0.9 (you have {__version__}). "
        "Run `cortex-training skills install` to update it."
    )


def test_status_shows_the_note_only_in_a_terminal(monkeypatch, tmp_path):
    from cortex_training import _picker

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _skills.install()
    with (tmp_path / "home" / ".claude" / "skills" / "cortex-training-setup" / "SKILL.md").open("a") as handle:
        handle.write("an older line\n")
    snowflake = Path(__import__("os").environ["SNOWFLAKE_HOME"])
    (snowflake / "connections.toml").write_text('[train]\naccount = "ACME"\nuser = "ME"\n')
    _picker.save_profile("train")
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "none.json"))
    _, _, err = _run(["status"], monkeypatch, tmp_path, tty=True)
    assert "agent skill is from" in err
    _, _, err = _run(["status"], monkeypatch, tmp_path)
    assert "agent skill is from" not in err


def test_the_wheel_ships_the_skill(tmp_path):
    # Built from a copy of the project, so no build output is left in the checkout.
    pytest.importorskip("setuptools")
    import shutil

    project = tmp_path / "project"
    shutil.copytree(REPO_ROOT / "src", project / "src", ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (REPO_ROOT / name).exists():
            shutil.copy(REPO_ROOT / name, project / name)
    out = tmp_path / "dist"
    script = "import sys; from setuptools import build_meta; build_meta.build_wheel(sys.argv[1])"
    subprocess.run([sys.executable, "-c", script, str(out)], cwd=project, check=True, capture_output=True, timeout=600)
    (wheel,) = out.glob("cortex_training-*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    for name in ("SKILL.md", "troubleshooting.md"):
        assert f"cortex_training/skills/cortex-training-setup/{name}" in names

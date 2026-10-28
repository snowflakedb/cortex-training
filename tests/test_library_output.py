from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "cortex_training"

# Entry points that own the terminal. Library code logs through ``logger`` so the
# CLI's JSON on stdout stays parseable.
_ALLOWED = {Path("cli/commands.py"), Path("tui/__main__.py")}


def _stdout_writes(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            lines.append(node.lineno)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "write"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "stdout"
        ):
            lines.append(node.lineno)
    return lines


def test_library_code_does_not_write_to_stdout():
    found = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found.extend(f"{relative}:{line}" for line in _stdout_writes(tree))

    assert not found, f"print() or sys.stdout.write() in library code: {found}"


def test_the_check_finds_a_print():
    assert _stdout_writes(ast.parse("print('x')\nimport sys\nsys.stdout.write('y')\n")) == [1, 3]

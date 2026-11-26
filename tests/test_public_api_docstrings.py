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

"""Merge gate for the public API docstrings.

The docstrings of every name in ``cortex_training.__all__`` are published verbatim on
docs.snowflake.com. Ruff (``ruff check``) enforces the structural rules; this test checks
what Ruff can't (``RULE_FIXES`` below describes each rule), and that Ruff's scope
(``[tool.ruff] include`` and each module's ``__all__``) still matches the exported API.
A violation is a string "<qualname>: <rule> <detail>"; the ones in
public_api_docstrings_baseline.txt are accepted for now.
"""

import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

import cortex_training

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; tomlkit is a runtime dependency
    import tomlkit as _tomlkit

    tomllib = None

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).with_name("public_api_docstrings_baseline.txt")
NOT_DOCUMENTED = {"wire", "__version__"}
GUIDE = "Background: the 'Public API docstrings' section of AGENTS.md."

# What each rule asks for, printed under the violations it explains.
RULE_FIXES = {
    "S1": "Start with a one-sentence summary ending in a period; a deprecated item's "
    "reads 'Deprecated: use <replacement>.'",
    "S2": "Add an Args: section that describes every parameter.",
    "S3": "Add a Returns: section (Yields: for an iterator).",
    "S4": "Keep private names out of public signatures: make the parameter private "
    "(_name), use a literal default, or type it with Callable[...] or an exported type.",
    "C1": "Don't use Latin abbreviations: write 'for example', 'that is', 'and so on', 'versus'.",
    "C2": "Don't mention repository paths or files; readers can't open them. Link a full https URL.",
    "C3": "Don't describe REST routes or wire framing; describe what the call does.",
    "C4": "Don't name private identifiers; name only what the reader can use.",
    "C5": "Write for the reader: no 'we', 'our', TODO, FIXME, or XXX.",
    "C6": "Don't name internal services (Neutrino, ZMD, the zone, control plane).",
    "C7": "Don't use em dashes; use a colon or two sentences.",
}


def _explain(violations: list[str]) -> str:
    rules = sorted({v.split(": ", 1)[1].split(" ", 1)[0] for v in violations})
    return "\n".join(f"  {rule}: {RULE_FIXES[rule]}" for rule in rules if rule in RULE_FIXES)

_DEPRECATED = re.compile(r"Deprecated: use \w")
_SECTION = re.compile(r"^(Args|Returns|Yields|Raises|Example|Examples|Note|Notes|Attributes):\s*$")
_URL = re.compile(r"https?://\S+")
_SPAN = re.compile(r"(`+)(?!`).+?(?<!`)\1", re.S)
_ROLE = re.compile(r":(?:meth|class|func|attr|mod|exc):`~?([^`]+)`")
_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")

# rule -> (pattern, applied to text with inline code spans removed)
_PROSE_RULES = {
    "C1": (re.compile(r"\be\.g\b\.?|\bi\.e\b\.?|\betc\b\.?|\bvs\."), True),
    "C2": (
        re.compile(
            r"\b(?:docs|src|tests|recipes)/|\b[\w-]+\.(?:md|py)\b|\bthis repo(?:sitory)?\b",
            re.I,
        ),
        True,
    ),
    "C3": (
        re.compile(
            r"\b(?:GET|POST|PUT|PATCH|DELETE) /|/api/v2|\?\w+=|DSSST1|application/octet-stream|proto3"
        ),
        False,
    ),
    "C5": (re.compile(r"\b(?:[Ww]e|[Oo]ur)\b|\b(?:TODO|FIXME|XXX)\b"), True),
    "C6": (
        re.compile(r"\b(?:Neutrino|ZMD|zone manager|the zone|control plane)\b", re.I),
        False,
    ),
    "C7": (re.compile("\u2014"), True),
}


# ── Docstring parsing ───────────────────────────────────────────────────────


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _sections(doc: str) -> dict[str, list[str]]:
    """Map a Google-style section name to its body lines."""
    found: dict[str, list[str]] = {}
    current = None
    for line in doc.splitlines():
        match = _SECTION.match(line) if _indent(line) == 0 else None
        if match:
            current = found.setdefault(match.group(1), [])
        elif current is not None:
            if line.strip() and _indent(line) == 0:
                current = None
            else:
                current.append(line)
    return found


def _prose(doc: str) -> str:
    """The docstring without code blocks and URLs."""
    kept = []
    fenced = False
    literal_base = None
    for line in doc.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if literal_base is not None:
            if not line.strip() or _indent(line) > literal_base:
                continue
            literal_base = None
        if line.lstrip().startswith(">>>"):
            continue
        if line.rstrip().endswith("::"):
            literal_base = _indent(line)
        kept.append(line)
    return _URL.sub(" ", "\n".join(kept))


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _has_private_component(dotted: str) -> bool:
    return any(_is_private(part) for part in dotted.split("."))


# ── Rule checks ─────────────────────────────────────────────────────────────


def _check_prose(qualname: str, doc: str) -> list[str]:
    violations = []
    prose = _prose(doc)
    outside_spans = _SPAN.sub(" ", prose)
    for rule, (pattern, skip_spans) in _PROSE_RULES.items():
        text = outside_spans if skip_spans else prose
        seen = []
        for match in pattern.finditer(text):
            token = match.group(0).strip().rstrip(".") if rule == "C1" else match.group(0).strip()
            if token not in seen:
                seen.append(token)
                violations.append(f'{qualname}: {rule} "{token}"')
    for match in _SPAN.finditer(prose):
        span = match.group(0).strip("`")
        if any(_has_private_component(token) for token in _DOTTED.findall(span)):
            violations.append(f"{qualname}: C4 backticked private name `{span.strip()}`")
    for match in _ROLE.finditer(prose):
        if _has_private_component(match.group(1)):
            violations.append(f"{qualname}: C4 role pointing at private name `{match.group(1)}`")
    return violations


def _private_names(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and _is_private(child.id):
            names.append(child.id)
        elif isinstance(child, ast.Attribute) and _is_private(child.attr):
            names.append(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                names.extend(_private_names(ast.parse(child.value, mode="eval")))
            except SyntaxError:
                pass
    return names


def _check_signature_privacy(qualname: str, func) -> list[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return []
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    violations = []
    for arg, default in params:
        if _is_private(arg.arg):
            continue
        if default is not None and _private_names(default):
            violations.append(f"{qualname}: S4 private default for {arg.arg}")
        if arg.annotation is not None and _private_names(arg.annotation):
            violations.append(f"{qualname}: S4 private annotation for {arg.arg}")
    for arg in (args.vararg, args.kwarg):
        if arg is not None and arg.annotation is not None and _private_names(arg.annotation):
            violations.append(f"{qualname}: S4 private annotation for {arg.arg}")
    if node.returns is not None and _private_names(node.returns):
        violations.append(f"{qualname}: S4 private annotation for return")
    return violations


def _check_summary(qualname: str, doc: str, *, required: bool) -> list[str]:
    if not doc.strip():
        # Ruff's D101, D102, D103, and D107 report this, except in a module whose name starts
        # with an underscore: Ruff takes everything in one as private, whatever its __all__.
        return [f"{qualname}: S1 missing docstring"] if required else []
    first = doc.splitlines()[0].strip()
    if not first.endswith(".") or re.search(r"[.!?] [A-Z]", first):
        return [f"{qualname}: S1 first line must be a one-sentence summary ending in a period"]
    if first.startswith("Deprecated") and not _DEPRECATED.match(first):
        return [f"{qualname}: S1 deprecated summary must read 'Deprecated: use <replacement>.'"]
    return []


def _in_private_module(obj) -> bool:
    return any(_is_private(part) for part in obj.__module__.split("."))


def check_callable(
    qualname: str, func, *, kind: str = "function", required: bool = False
) -> list[str]:
    """Check one public function, method or property; kind is "function" or "property"."""
    target = func.fget if isinstance(func, property) else func
    doc = inspect.cleandoc(target.__doc__ or "")
    violations = _check_summary(qualname, doc, required=required)
    violations += _check_signature_privacy(qualname, target)
    violations += _check_prose(qualname, doc)
    if kind == "property":
        return violations
    # Ruff's D417 checks every parameter against an Args section that exists, so only a
    # missing section is left to S2. Its DOC201 and DOC402 skip a one-line docstring and,
    # under the Google convention, a summary that starts with "Return", so S3 stays whole.
    signature = inspect.signature(target)
    has_parameters = any(
        name not in ("self", "cls") and not _is_private(name) for name in signature.parameters
    )
    sections = _sections(doc)
    if has_parameters and "Args" not in sections and not _DEPRECATED.match(doc):
        violations.append(f"{qualname}: S2 missing Args section")
    returns_none = signature.return_annotation in (None, "None")
    if (
        target.__name__ != "__init__"
        and not returns_none
        and not ({"Returns", "Yields"} & sections.keys())
    ):
        violations.append(f"{qualname}: S3 missing Returns or Yields section")
    return violations


def check_class(qualname: str, cls: type) -> list[str]:
    own_doc = cls.__dict__.get("__doc__") or ""
    if own_doc.startswith(f"{cls.__name__}("):  # dataclass-generated
        own_doc = ""
    required = _in_private_module(cls)
    violations = _check_summary(qualname, inspect.cleandoc(own_doc), required=required)
    violations += _check_prose(qualname, inspect.cleandoc(own_doc))
    is_exception = issubclass(cls, BaseException)
    for name, member in cls.__dict__.items():
        if name == "__init__":
            if inspect.isfunction(member) and member.__code__.co_filename != "<string>":
                if is_exception:
                    violations += _check_signature_privacy(f"{qualname}.__init__", member)
                else:
                    violations += check_callable(
                        f"{qualname}.__init__", member, required=required
                    )
            continue
        if _is_private(name) or name.startswith("__"):
            continue
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        if isinstance(member, property):
            violations += check_callable(
                f"{qualname}.{name}", member, kind="property", required=required
            )
        elif inspect.isfunction(member):
            violations += check_callable(f"{qualname}.{name}", member, required=required)
    return violations


def collect_violations() -> list[str]:
    violations = []
    for name in cortex_training.__all__:
        if name in NOT_DOCUMENTED:
            continue
        obj = getattr(cortex_training, name)
        if inspect.isclass(obj):
            violations += check_class(name, obj)
        else:
            violations += check_callable(name, obj, required=_in_private_module(obj))
    return violations


# ── Baseline ────────────────────────────────────────────────────────────────


def load_baseline() -> dict[str, str]:
    """Return {violation: reason} from the baseline file."""
    entries = {}
    for number, line in enumerate(BASELINE_PATH.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, reason = line.partition("  # ")
        assert reason.strip(), f"baseline line {number} needs a trailing '  # reason'"
        entries[key.strip()] = reason.strip()
    return entries


# ── Tests ───────────────────────────────────────────────────────────────────


def test_public_api_docstrings_have_no_new_violations():
    new = sorted(set(collect_violations()) - set(load_baseline()))
    assert not new, (
        f"{len(new)} public API docstring violation(s):\n  "
        + "\n  ".join(new)
        + f"\nHow to fix them:\n{_explain(new)}\n{GUIDE}"
    )


def test_public_api_docstring_baseline_has_no_stale_entries():
    stale = sorted(set(load_baseline()) - set(collect_violations()))
    assert not stale, (
        "These baseline entries no longer occur; remove them from "
        f"tests/public_api_docstrings_baseline.txt:\n  " + "\n  ".join(stale)
    )


def _ruff_include() -> set[str]:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    data = tomllib.loads(text) if tomllib else _tomlkit.parse(text)
    return set(data["tool"]["ruff"]["include"])


def test_ruff_scope_matches_the_exported_api():
    include = _ruff_include()
    problems = [f"{path} in [tool.ruff] include doesn't exist" for path in sorted(include)
                if not (REPO_ROOT / path).is_file()]
    for name in cortex_training.__all__:
        if name in NOT_DOCUMENTED:
            continue
        module = sys.modules[getattr(cortex_training, name).__module__]
        path = Path(module.__file__).resolve().relative_to(REPO_ROOT).as_posix()
        if path not in include:
            problems.append(f"{name}: add {path} to [tool.ruff] include in pyproject.toml")
        if name not in getattr(module, "__all__", ()):
            problems.append(f"{name}: add it to {module.__name__}.__all__, which Ruff reads")
    assert not problems, "Ruff's docstring scope has drifted from the public API:\n  " + "\n  ".join(
        problems
    )


def test_checker_flags_each_rule_family():
    def bad(self, a, retry_on=_private_hook, *args) -> int:
        """Does things, e.g. we call POST /x, see `_hidden` or src/a.py \u2014 TODO

        Args:
            a: The first.
        """

    def unlisted(a, b) -> int:
        """Return a number."""

    violations = " | ".join(check_callable("bad", bad) + check_callable("unlisted", unlisted))
    for expected in (
        "S1",
        "unlisted: S2 missing Args section",
        "unlisted: S3",
        "S4 private default for retry_on",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C7",
    ):
        assert expected in violations, (expected, violations)


def test_missing_docstring_is_only_ours_in_private_modules():
    def nodoc() -> None: ...

    assert check_callable("nodoc", nodoc) == []
    assert check_callable("nodoc", nodoc, required=True) == ["nodoc: S1 missing docstring"]


def test_checker_accepts_a_documented_function_and_a_deprecated_one():
    def good(self, name: str, *, wait: float = 1.0) -> str:
        """Return the name.

        Args:
            name: The name, such as `x_`.
            wait: How long to wait, in seconds. A long description
                can continue on an indented line.

        Returns:
            The name.
        """

    def old(*args, **kwargs) -> str:
        """Deprecated: use good.

        Returns:
            The result.
        """

    assert check_callable("good", good) == []
    assert check_callable("old", old) == []


def _private_hook(exc):
    return False

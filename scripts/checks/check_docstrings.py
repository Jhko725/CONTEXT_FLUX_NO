# Adapted from Jhko725/UniversalForceReconstruction scripts/checks/check_docstrings.py at b80a341.
"""Check what ruff's docstring rules leave unchecked: the form of the headers
`**Arguments:**`, `**Returns:**`, `**Raises:**` and `**Attributes:**`; a docstring on
every private function and class under ``src``; a bound on a module's length; and a
docstring that opens with a label from a plan.

`.claude/rules/python.md` fixes the form diffrax writes: each header
stands on its own line, followed by a blank line and then the bullets or the paragraph.
ruff's pydocstyle rules know Google and numpy sections, not these, so the check is a
regex: a header line with text after the closing stars, or without a blank line after
it, is flagged.

ruff's ``D103`` skips a name that starts with an underscore, so under ``src`` this
check requires a docstring on every ``_``-prefixed function, method and class too.
Dunder methods are exempt: ``__repr__`` says what it is.

A module under ``src`` is bounded at `MAX_MODULE_LINES`. A module that needs more
should be split by common functionalities
(`code-conventions` § Code goes where it belongs).

A docstring whose first line opens with a numbered label of a plan, such as
``Item 4:`` or ``Pass 2, unit 11:``, is flagged. The label names a handoff document
that the reader of the code does not have (`clear-prose` § Documents); the reviews of
the source repository's ``refactor-foundation`` found twelve such docstrings in its
tests.

The check applies to the clean scope of rule R7 only (the ``clean`` list of
``pyproject.toml``): a named path outside it is skipped, and the private-docstring and
length rules apply to the clean paths under ``src``.

Usage: `uv run python scripts/checks/check_docstrings.py path ...`, where `prek` passes
every tracked Python file. Exits 1 when anything is flagged.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLEAN: tuple[str, ...] = tuple(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "context_flux_no"
    ]["clean"]
)
"""The paths, relative to the repository root, where the strict checks apply."""
HEADER = re.compile(r"^\s*\*\*(Arguments|Returns|Raises|Attributes):\*\*(.*)$")
PRIVATE_SCOPE = tuple(entry for entry in CLEAN if entry.startswith("src/"))
MAX_MODULE_LINES = 600
PLAN_LABEL = re.compile(r"(?:Item|Unit|Pass|Round|Phase|Q) ?\d+\b", re.IGNORECASE)
"""A numbered label of a plan, where a docstring's first line opens with it."""


def in_scope(path: Path, scope: tuple[str, ...]) -> bool:
    """Whether `path` (absolute, or relative to the root) lies inside `scope`."""
    absolute = path if path.is_absolute() else ROOT / path
    try:
        relative = absolute.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return False
    return any(relative == entry or relative.startswith(entry + "/") for entry in scope)


def header_problems(path: Path, lines: list[str]) -> Iterator[str]:
    """The flagged headers of `path`, one ``file:line: reason`` per header."""
    for i, line in enumerate(lines, start=1):
        m = HEADER.match(line)
        if m is None:
            continue
        name, rest = m.groups()
        if rest.strip():
            yield f"{path}:{i}: text after `**{name}:**` on the same line"
        elif i == len(lines) or lines[i].strip():
            yield f"{path}:{i}: no blank line after `**{name}:**`"


def private_problems(path: Path, source: str) -> Iterator[str]:
    """The ``_``-prefixed functions, methods and classes of `path` without a
    docstring, dunders excepted."""
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        name = node.name
        if not name.startswith("_") or name.startswith("__"):
            continue
        if ast.get_docstring(node) is None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            yield f"{path}:{node.lineno}: private {kind} `{name}` has no docstring"


def label_problems(path: Path, source: str) -> Iterator[str]:
    """The docstrings of `path` whose first line opens with a plan label."""
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            continue
        label = PLAN_LABEL.match(ast.get_docstring(node) or "")
        if label is not None:
            line = node.body[0].lineno
            yield (
                f"{path}:{line}: the docstring opens with the plan label"
                f" `{label.group()}`; say what the code does"
            )


def length_problems(path: Path, lines: list[str]) -> Iterator[str]:
    """`path` when it is longer than `MAX_MODULE_LINES`."""
    count = len(lines) - (1 if lines and lines[-1] == "" else 0)
    if count > MAX_MODULE_LINES:
        yield (
            f"{path}:{count}: module has {count} lines, over {MAX_MODULE_LINES};"
            " split it by what serves what"
        )


def problems(path: Path) -> Iterator[str]:
    """Every flagged line of `path`, as ``file:line: reason``."""
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")
    yield from header_problems(path, lines)
    yield from label_problems(path, source)
    if in_scope(path, PRIVATE_SCOPE):
        yield from private_problems(path, source)
        yield from length_problems(path, lines)


def main(argv: list[str]) -> int:
    """Run the check over the files named in `argv`; return the exit code."""
    files = [Path(a) for a in argv if a.endswith(".py") and in_scope(Path(a), CLEAN)]
    found = [p for f in files for p in problems(f)]
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

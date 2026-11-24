# Adapted from Jhko725/UniversalForceReconstruction scripts/checks/check_containers.py at b80a341.
"""Check the container rule of the `code-conventions` skill: which class holds which
kind of value.

A value that a JAX transform sees is an ``eqx.Module``; a value that must stay out of
tracing is a frozen ``@dataclass`` holding NumPy or Python values. The rule cannot be
read off a type at runtime, so the check walks the syntax tree of every module in the
clean paths under ``src`` (rule R7, the ``clean`` list of ``pyproject.toml``) and flags
two patterns:

1. A class decorated ``@dataclass(frozen=True)``, in any spelling, whose field
   annotation mentions ``Array`` (``Float[Array, ...]``, a bare ``Array``,
   ``jax.Array``). A mutable dataclass is a Hydra schema or a stateful service, and a
   service holds what it needs.
2. A field whose default is ``eqx.field(static=True)`` and whose annotation mentions
   ``Array``: a static field decides the traced program's structure, and an array
   there is a new program per value.

In both tests, an ``Array`` inside a ``Callable[...]`` subscript is a function's
signature, not an array, and is not flagged.

The source's third rule, no ``NamedTuple`` under its ``system/`` subpackage, is left
out: this package has no subpackage whose values only JAX transforms see.

Usage: `uv run python scripts/checks/check_containers.py` (no arguments: it walks
the clean paths under `src` that exist). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import ast
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
# A clean path that a later unit adds is skipped until it exists.
ROOTS = [
    Path(entry)
    for entry in CLEAN
    if entry.startswith("src/") and (ROOT / entry).exists()
]


def mentions_array(annotation: ast.AST) -> bool:
    """Whether `annotation` names ``Array`` as a name or an attribute, also inside a
    string annotation, outside any ``Callable[...]`` subscript."""
    if (
        isinstance(annotation, ast.Subscript)
        and _name_of(annotation.value) == "Callable"
    ):
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "Array"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Array"
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            inner = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False  # a shape string such as " n dim", not an annotation
        return mentions_array(inner)
    return any(mentions_array(child) for child in ast.iter_child_nodes(annotation))


def _name_of(node: ast.expr) -> str:
    """The last component of a dotted name or call: ``dataclass`` for
    ``dataclasses.dataclass(frozen=True)``, ``""`` for anything else."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def is_frozen_dataclass(decorator: ast.expr) -> bool:
    """Whether `decorator` is ``dataclass(..., frozen=True, ...)`` in any spelling."""
    return (
        isinstance(decorator, ast.Call)
        and _name_of(decorator) == "dataclass"
        and any(
            k.arg == "frozen" and isinstance(k.value, ast.Constant) and k.value.value
            for k in decorator.keywords
        )
    )


def is_static_field(value: ast.expr | None) -> bool:
    """Whether `value` is a call ``eqx.field(..., static=True, ...)``."""
    if not isinstance(value, ast.Call) or _name_of(value.func) != "field":
        return False
    return any(
        k.arg == "static" and isinstance(k.value, ast.Constant) and k.value.value
        for k in value.keywords
    )


def fields(node: ast.ClassDef) -> list[tuple[str, ast.AnnAssign]]:
    """The annotated fields of the class body `node`, each with its name."""
    return [
        (stmt.target.id, stmt)
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]


def class_problems(path: Path, node: ast.ClassDef) -> Iterator[str]:
    """The flagged lines of the class `node` in `path`."""
    is_frozen = any(is_frozen_dataclass(d) for d in node.decorator_list)
    for field_name, f in fields(node):
        name = f"{node.name}.{field_name}"
        if not mentions_array(f.annotation):
            continue
        if is_frozen:
            yield (
                f"{path}:{f.lineno}: {name}: a frozen dataclass holds a jax.Array; "
                "an eqx.Module, or NumPy in the dataclass"
            )
        if is_static_field(f.value):
            yield (
                f"{path}:{f.lineno}: {name}: a static field holds a jax.Array; "
                "an array leaf instead"
            )


def problems(path: Path) -> Iterator[str]:
    """Every flagged line of `path` (relative to the root), as
    ``file:line: <class>.<field>: <rule>``."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield from class_problems(path, node)


def main() -> int:
    """Run the check over `ROOTS`; return the exit code."""
    files = sorted(
        f.relative_to(ROOT)
        for root in ROOTS
        for f in (
            [ROOT / root] if (ROOT / root).is_file() else (ROOT / root).rglob("*.py")
        )
    )
    found = [p for f in files for p in problems(f)]
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

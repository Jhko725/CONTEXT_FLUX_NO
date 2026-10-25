"""JSON for records (`runs.RunRecord`, `simulations.DatasetRecord`) and store attributes.

One encoder for the whole package: numpy scalars and arrays become lists/scalars, `Path`
becomes a string, sets become sorted lists; anything else falls back to `str`, so a record
never fails to serialize because a config value had an unexpected type. Decoding is plain
`json.loads` — records restore types from their dataclass fields, not from the JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _default(o: Any) -> Any:
    if hasattr(o, "tolist"):  # numpy arrays and scalars
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def dumps(obj: Any, *, indent: int | None = 2) -> str:
    """`json.dumps` with the package encoder and sorted keys."""
    return json.dumps(obj, indent=indent, default=_default, sort_keys=True)

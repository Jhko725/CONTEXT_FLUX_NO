"""Every module of the package imports on CPU, `legacy/` included.

The modules are found on the file system, so collecting the test imports nothing and a
module that fails to import fails its own test case.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest


def _module_names() -> list[str]:
    spec = importlib.util.find_spec("context_flux_no")
    assert spec is not None and spec.origin is not None
    package_dir = Path(spec.origin).parent
    names = []
    for path in sorted(package_dir.rglob("*.py")):
        parts = path.relative_to(package_dir.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


@pytest.mark.parametrize("module", _module_names())
def test_module_imports(module: str):
    """The module imports."""
    importlib.import_module(module)

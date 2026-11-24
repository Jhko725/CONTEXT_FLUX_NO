# Adapted from Jhko725/UniversalForceReconstruction tests/conftest.py at b80a341.
"""Test configuration: runtime shape checks and the Hypothesis profiles.

The import hook checks every jaxtyping annotation of the package's clean modules at
run time, with beartype. It covers the clean paths of rule R7 under `src/` only, read
from `clean` in `pyproject.toml`, where the scope is held once: the legacy modules'
annotations were never checked and are not held to them. The hook is installed before
any package import, since it only affects modules imported after it. It does not
enable x64: the package trains in float32, and `tests/verification/conftest.py`
enables x64 for its tier.

Hypothesis profiles (select with ``--hypothesis-profile=<name>``):
  dev      10 examples   quick local runs, the default of ``just test``
  default  50 examples   plain ``pytest``
  ci       100 examples  derandomized, for CI

The deadline is off and the ``too_slow`` health check is suppressed because the first
example of every test pays the XLA compile, which Hypothesis would otherwise report as
a flaky slow example. The ``ci`` profile is derandomized so that a CI failure is
reproducible; local runs stay random so that they keep exploring.
"""

import tomllib
from pathlib import Path

import jaxtyping
from hypothesis import HealthCheck, settings


_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_CLEAN = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"][
    "context_flux_no"
]["clean"]
jaxtyping.install_import_hook(
    [
        entry.removeprefix("src/").removesuffix(".py").replace("/", ".")
        for entry in _CLEAN
        if entry.startswith("src/")
    ],
    "beartype.beartype",
)

_common = settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow], print_blob=True
)
settings.register_profile("default", _common, max_examples=50)
settings.register_profile("dev", _common, max_examples=10)
settings.register_profile("ci", _common, max_examples=100, derandomize=True)
settings.load_profile("default")

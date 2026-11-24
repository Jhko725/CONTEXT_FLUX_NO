# Adapted from Jhko725/UniversalForceReconstruction tests/conftest.py at b80a341.
"""Test configuration: the Hypothesis profiles.

Hypothesis profiles (select with ``--hypothesis-profile=<name>``):
  dev      10 examples   quick local runs, the default of ``just test``
  default  50 examples   plain ``pytest``
  ci       100 examples  derandomized, for CI

The deadline is off and the ``too_slow`` health check is suppressed because the first
example of every test pays the XLA compile, which Hypothesis would otherwise report as
a flaky slow example. The ``ci`` profile is derandomized so that a CI failure is
reproducible; local runs stay random so that they keep exploring.
"""

from hypothesis import HealthCheck, settings


_common = settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow], print_blob=True
)
settings.register_profile("default", _common, max_examples=50)
settings.register_profile("dev", _common, max_examples=10)
settings.register_profile("ci", _common, max_examples=100, derandomize=True)
settings.load_profile("default")

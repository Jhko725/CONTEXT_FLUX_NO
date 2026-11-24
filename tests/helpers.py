"""Named solver configurations and tolerances shared by the test tiers; every call site uses
the name, and each entry carries its one-line reason."""

SHARPCLAW_SSP33 = {"time_integrator": "SSP33", "cfl_desired": 0.4, "cfl_max": 0.45}
"""SharpClaw with the third-order SSP Runge–Kutta (SSP coefficient 1; `pyclaw/sharpclaw/
solver.py:137`). Clawpack provides CFL defaults only for SSP104 (2.45 / 2.5 at coefficient 6,
`solver.py:146`), so SSP33 needs them spelled out: the same numbers per unit coefficient."""

PRESSURE_ROUND_TRIP_RTOL = 1.0e-3
"""Relative tolerance of `Euler2D.pressure` on a state from `to_conserved`, in float32: the
pressure is `E` minus the kinetic energy, so the rounding of `E` (about 6e-8 relative) is
amplified by `E / p`, at most about 1200 for the strategy of `tests/unit/test_admissible.py`."""

"""Cross-code agreement, verification tier (`-m slow`): V18a for 2-D Euler (Classic vs
SharpClaw, both PyClaw; measured 2026-10-25: ratios 8.3 / 3.5 over N = 16, 32, 64 at
T = 0.02) and V18, PyClaw vs Trixi on every family (skipped until `scripts/setup_julia.py`
has run on the machine; ratios to be measured on the laptop and recorded in
`docs/verification.md`). V19 (own WENO5) joins when it exists.

V18 pairs SharpClaw with the `SSP33` integrator against Trixi's `SSPRK33` — the same
third-order SSP Runge–Kutta on both sides, as in Shu's FD/FV-WENO vs DG comparisons [cited:
Shu 2001, ICASE 2001-11 §4] — so the distance measures the spatial schemes. Generator defaults
are untouched (`SSP104`). The ratio criterion is V18a's (ADR-0011 §4)."""

import numpy as np
import pytest
from context_flux_no.simulations import Grid, normalize_bcs
from context_flux_no.simulations.backends import trixi
from context_flux_no.simulations.initial_conditions import FromFunction
from context_flux_no.simulations.pde import CubicFlux2D


pytest.importorskip("clawpack")

from context_flux_no.simulations.backends.pyclaw import PyClawBackend  # noqa: E402

from tests.helpers import SHARPCLAW_SSP33  # noqa: E402
from tests.unit.make_goldens_simulations import CASES  # noqa: E402


pytestmark = pytest.mark.slow

MIN_RATIO = 1.5
HORIZON_1D = {
    "cubic_1d": 0.05,
    "sine_1d": 0.05,
    "burgers_1d": 0.05,
    "shallow_water_1d": 0.03,
}
needs_trixi = pytest.mark.skipif(
    not trixi.available(), reason="Julia/Trixi not set up (scripts/setup_julia.py)"
)


def test_euler_2d_classic_vs_sharpclaw():
    pde, ic, x_spans, _, _, _, kwargs = CASES["euler_2d"]
    d = []
    for N in (16, 32, 64):
        u_c, *_ = pde.solve(ic, x_spans, [N, N], (0, 0.02), 1, "periodic", **kwargs)
        u_s, *_ = pde.solve(
            ic,
            x_spans,
            [N, N],
            (0, 0.02),
            1,
            "periodic",
            backend=PyClawBackend("sharpclaw"),
            **kwargs,
        )
        d.append(float(np.abs(u_c[-1] - u_s[-1]).mean()))
    ratios = [d[i] / d[i + 1] for i in range(2)]
    assert all(r >= 1.5 for r in ratios), (d, ratios)


def _l1_sharpclaw_vs_trixi(pde, ic, x_spans, Nxs, T) -> float:
    u_s, *_ = pde.solve(
        ic,
        x_spans,
        Nxs,
        (0, T),
        1,
        "periodic",
        backend=PyClawBackend("sharpclaw", **SHARPCLAW_SSP33),
    )
    u_t, *_ = pde.solve(
        ic, x_spans, Nxs, (0, T), 1, "periodic", backend=trixi.TrixiBackend()
    )
    return float(np.abs(u_s[-1] - u_t[-1]).mean())


@needs_trixi
@pytest.mark.parametrize("name", sorted(HORIZON_1D))
def test_reference_backends_1d(name):
    """V18: SharpClaw (SSP33) and Trixi DGSEM converge to each other on every 1-D family."""
    pde, ic, x_spans, *_ = CASES[name]
    d = [
        _l1_sharpclaw_vs_trixi(pde, ic, x_spans, [N], HORIZON_1D[name])
        for N in (64, 128, 256)
    ]
    ratios = [d[i] / d[i + 1] for i in range(2)]
    assert all(r >= MIN_RATIO for r in ratios), (name, d, ratios)


@needs_trixi
def test_reference_backends_euler_2d():
    """V18 for 2-D Euler (compiled Riemann solver on the PyClaw side)."""
    pde, ic, x_spans, *_ = CASES["euler_2d"]
    d = [_l1_sharpclaw_vs_trixi(pde, ic, x_spans, [N, N], 0.02) for N in (16, 32, 64)]
    ratios = [d[i] / d[i + 1] for i in range(2)]
    assert all(r >= MIN_RATIO for r in ratios), (d, ratios)


@needs_trixi
def test_reference_backends_cubic_2d():
    """V18 for `CubicFlux2D`, whose only other solver is its own first-order Rusanov
    stepper (F8): the distance to Trixi shrinks at that stepper's rate (ratio → 2)."""
    pde = CubicFlux2D(1.0, 0.0, 0.0)
    ic = FromFunction(
        lambda c: (0.5 * np.sin(2 * np.pi * c[0]) * np.cos(2 * np.pi * c[1]))[None],
        n_eqn=1,
        n_spatial_dims=2,
    )
    x_spans, T = [(0, 1), (0, 1)], 0.02
    d = []
    for N in (16, 32, 64):
        u_own, *_ = pde.solve(ic, x_spans, [N, N], (0, T), 1, "periodic")
        grid = Grid(x_spans, [N, N])
        u_t = trixi.TrixiBackend().solve(
            pde, ic, grid, np.array([0.0, T]), normalize_bcs("periodic", 2)
        )
        d.append(float(np.abs(np.asarray(u_own[-1]) - u_t[-1]).mean()))
    ratios = [d[i] / d[i + 1] for i in range(2)]
    assert all(r >= MIN_RATIO for r in ratios), (d, ratios)

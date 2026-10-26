"""V18a — PyClaw Classic vs SharpClaw agree by mutual convergence (ADR-0004 §Consequences).

Two consistent schemes converge to the same entropy solution, so the L1 distance between them
on a shared grid must shrink under refinement; for a second-order and a fifth-order scheme the
distance is dominated by the second-order error and the ratio per halving approaches 4.
Measured 2026-10-25 on a smooth pre-shock horizon (recorded in `docs/verification.md`):
cubic 3.0/4.2, sine 4.1/4.3, Burgers 3.5/4.1, shallow water 4.7/5.0; the test requires ≥ 1.5.
The 2-D Euler case is in `test_crosscode.py` (`-m slow`); this file is fast and unmarked, so it
runs on every commit — a verification-tier test is placed by the claim it carries (V18a), not by
its speed (`docs/coding-standards.md` §Test contracts)."""

import numpy as np
import pytest


pytest.importorskip("clawpack")

from context_flux_no.simulations.pyclaw import PyClawBackend  # noqa: E402

from tests.unit.make_goldens_simulations import CASES  # noqa: E402


MIN_RATIO = 1.5
HORIZON = {
    "cubic_1d": 0.05,
    "sine_1d": 0.05,
    "burgers_1d": 0.05,
    "shallow_water_1d": 0.03,
}


def l1_classic_vs_sharpclaw(pde, ic, x_spans, N, T, **kwargs) -> float:
    u_classic, *_ = pde.solve(
        ic, x_spans, [N] * len(x_spans), (0, T), 1, "periodic", **kwargs
    )
    u_sharp, *_ = pde.solve(
        ic,
        x_spans,
        [N] * len(x_spans),
        (0, T),
        1,
        "periodic",
        backend=PyClawBackend("sharpclaw"),
        **kwargs,
    )
    return float(np.abs(u_classic[-1] - u_sharp[-1]).mean())


@pytest.mark.parametrize("name", sorted(HORIZON))
def test_classic_and_sharpclaw_converge_to_each_other(name):
    pde, ic, x_spans, *_ = CASES[name]
    d = [
        l1_classic_vs_sharpclaw(pde, ic, x_spans, N, HORIZON[name])
        for N in (64, 128, 256)
    ]
    ratios = [d[i] / d[i + 1] for i in range(2)]
    assert all(r >= MIN_RATIO for r in ratios), (name, d, ratios)

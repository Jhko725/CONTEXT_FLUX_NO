"""Cross-code agreement, verification tier (`-m slow`): V18a for 2-D Euler (Classic vs
SharpClaw, both PyClaw). V18 (PyClaw vs Trixi) and V19 (own WENO5) are added here when those
backends exist. Measured 2026-10-25: ratios 8.3 / 3.5 over N = 16, 32, 64 at T = 0.02."""

import numpy as np
import pytest


pytest.importorskip("clawpack")

from context_flux_no.simulations.pyclaw import PyClawBackend  # noqa: E402

from tests.unit.make_goldens_simulations import CASES  # noqa: E402


pytestmark = pytest.mark.slow


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

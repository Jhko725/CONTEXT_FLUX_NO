"""The cases behind `goldens_simulations/*.npz` and the script that (re)generates them.

The goldens pin `PyClawBackend("classic")` to the pre-2a per-class `solve()` output. They were
produced at commit `1ff31bb` (the last commit before the backend refactor) by running these
exact cases through the old methods; regenerate **only** with a change-document entry saying
which commit changed PyClaw's output and why:

    uv run python tests/unit/make_goldens_simulations.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from context_flux_no.simulations.pde import (
    Burgers1D,
    CubicFlux1D,
    Euler2D,
    ParametrizedShallowWater1D,
    SineFlux1D,
)


GOLDENS = Path(__file__).parent / "goldens_simulations"


def ic_scalar(xs):
    x = xs[0]
    return (0.5 + 0.4 * np.sin(2 * np.pi * x) + 0.1 * np.cos(6 * np.pi * x))[None]


def ic_sw(xs):
    x = xs[0]
    h = 1.0 + 0.2 * np.sin(2 * np.pi * x)
    return np.stack([h, 0.1 * h * np.cos(2 * np.pi * x)])


def ic_euler(xs):
    x, y = np.meshgrid(*xs, indexing="ij")
    rho = 1.0 + 0.2 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    u = 0.1 * np.ones_like(rho)
    v = -0.05 * np.ones_like(rho)
    p = 1.0 + 0.1 * np.cos(2 * np.pi * x)
    return rho, u, v, p


# name: (pde, ic_factory, x_spans, Nxs, t_span, Nt, solve kwargs)
CASES = {
    "cubic_1d": (
        CubicFlux1D(1.0, -0.5, 0.25),
        ic_scalar,
        [(0, 1)],
        [64],
        (0, 0.2),
        4,
        {},
    ),
    "sine_1d": (SineFlux1D(1.0, 2.0), ic_scalar, [(0, 1)], [64], (0, 0.2), 4, {}),
    "burgers_1d": (Burgers1D(), ic_scalar, [(0, 1)], [64], (0, 0.2), 4, {}),
    "shallow_water_1d": (
        ParametrizedShallowWater1D(1.0, 1.0, 1.0),
        ic_sw,
        [(0, 1)],
        [64],
        (0, 0.1),
        4,
        {},
    ),
    "euler_2d": (
        Euler2D(1.4),
        ic_euler,
        [(0, 1), (0, 1)],
        [16, 16],
        (0, 0.05),
        2,
        {"use_rho_v_p_ics": True},
    ),
}


def main() -> None:
    GOLDENS.mkdir(exist_ok=True)
    for name, (pde, ic, x_spans, Nxs, t_span, Nt, kwargs) in CASES.items():
        u, t, _ = pde.solve(ic, x_spans, Nxs, t_span, Nt, "periodic", **kwargs)
        np.savez_compressed(GOLDENS / f"{name}.npz", u=np.asarray(u), t=np.asarray(t))
        print(name, u.shape)


if __name__ == "__main__":
    main()

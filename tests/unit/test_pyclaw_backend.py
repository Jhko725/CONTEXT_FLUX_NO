"""`PyClawBackend`: the Classic scheme is bit-identical to the pre-2a per-class `solve()`
methods (goldens under `goldens_simulations/`, made at commit `1ff31bb` by the script in
`make_goldens_simulations.py`), SharpClaw runs on every PyClaw family, and the plumbing
(`Grid`, `normalize_bcs`, `describe`) behaves."""

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("clawpack")

from context_flux_no.simulations.backend import normalize_bcs  # noqa: E402
from context_flux_no.simulations.grid import Grid  # noqa: E402
from context_flux_no.simulations.pyclaw import PyClawBackend  # noqa: E402

from tests.unit.make_goldens_simulations import CASES, GOLDENS  # noqa: E402


@pytest.mark.parametrize("name", sorted(CASES))
def test_classic_is_bit_identical_to_pre_2a_solve(name):
    pde, ic, x_spans, Nxs, t_span, Nt, kwargs = CASES[name]
    golden = np.load(GOLDENS / f"{name}.npz")
    u, t, xs = pde.solve(ic, x_spans, Nxs, t_span, Nt, "periodic", **kwargs)
    assert u.shape == golden["u"].shape
    np.testing.assert_array_equal(u, golden["u"])
    np.testing.assert_array_equal(t, golden["t"])
    assert len(xs) == len(Nxs) and all(len(x) == n for x, n in zip(xs, Nxs))


@pytest.mark.parametrize("name", sorted(CASES))
def test_sharpclaw_runs_on_every_family(name):
    pde, ic, x_spans, Nxs, t_span, Nt, kwargs = CASES[name]
    backend = PyClawBackend("sharpclaw")
    u, t, _ = pde.solve(
        ic, x_spans, Nxs, t_span, Nt, "periodic", backend=backend, **kwargs
    )
    assert u.shape[0] == Nt + 1 and np.isfinite(u).all()
    # SharpClaw and Classic start from the same state and stay close on a short horizon
    u_c, *_ = pde.solve(ic, x_spans, Nxs, t_span, Nt, "periodic", **kwargs)
    np.testing.assert_array_equal(u[0], u_c[0])
    assert np.abs(u[-1] - u_c[-1]).max() < 0.25 * (np.abs(u_c[-1]).max() + 1e-12)


def test_grid_matches_pyclaw_centres():
    from clawpack import pyclaw

    g = Grid(((0.0, 1.0), (-1.0, 2.0)), (7, 5))
    for (lo, hi), n, c in zip(g.x_spans, g.Nxs, g.centers):
        np.testing.assert_array_equal(c, pyclaw.Dimension(lo, hi, n).centers)
    assert g.dxs == (1 / 7, 3 / 5) and g.axis_names == ("x", "y")
    with pytest.raises(ValueError):
        Grid(((0.0, 1.0),), (0,))


def test_normalize_bcs():
    assert normalize_bcs("periodic", 2) == (("periodic", "periodic"),) * 2
    assert normalize_bcs([("extrap", "wall")], 1) == (("extrap", "wall"),)
    with pytest.raises(ValueError):
        normalize_bcs([("periodic", "wall")], 1)
    with pytest.raises(ValueError):
        normalize_bcs("free", 1)
    with pytest.raises(ValueError):
        normalize_bcs("periodic", 1) and normalize_bcs(["periodic"], 2)


def test_backend_input_checks_and_describe():
    pde, ic, x_spans, Nxs, t_span, Nt, _ = CASES["cubic_1d"]
    backend = PyClawBackend("classic", cfl_desired=0.3)
    grid = Grid(x_spans, Nxs)
    with pytest.raises(ValueError):
        backend.solve(
            pde,
            np.zeros((2, *grid.shape)),
            grid,
            np.linspace(0, 1, 3),
            normalize_bcs("periodic", 1),
        )
    with pytest.raises(ValueError):
        backend.solve(
            pde,
            np.zeros((1, *grid.shape)),
            grid,
            np.array([0.0]),
            normalize_bcs("periodic", 1),
        )
    d = backend.describe()
    assert (
        d["name"] == "pyclaw"
        and d["scheme"] == "classic"
        and d["solver_overrides"] == {"cfl_desired": 0.3}
    )
    with pytest.raises(ValueError):
        PyClawBackend("weno")  # type: ignore[arg-type]


def test_non_periodic_bcs_run(tmp_path: Path):
    pde, ic, x_spans, Nxs, t_span, Nt, _ = CASES["burgers_1d"]
    u, *_ = pde.solve(ic, x_spans, Nxs, t_span, Nt, [("extrap", "extrap")])
    assert np.isfinite(u).all()


def test_blow_up_is_a_solver_failure():
    from context_flux_no.simulations import SolverFailure

    pde, ic, x_spans, Nxs, t_span, Nt, _ = CASES["cubic_1d"]
    # cfl_desired above cfl_max makes PyClaw refuse the run with a bare Exception
    backend = PyClawBackend("classic", cfl_desired=2.0, cfl_max=1.0)
    with pytest.raises(SolverFailure) as info:
        pde.solve(ic, x_spans, Nxs, t_span, Nt, "periodic", backend=backend)
    assert info.value.backend == "pyclaw/classic"

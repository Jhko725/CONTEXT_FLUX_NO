"""`PyClawBackend`: Clawpack's PyClaw as a `SolverBackend` (ADR-0004, ADR-0011).

Two schemes, same interface:

- `"classic"` — wave-propagation (Lax–Wendroff + Riemann solver + TVD limiters), second
  order, `num_ghost = 2`. What every ICLR dataset was generated with; kept bit-identical to
  the old per-class `solve()` methods (`tests/unit/test_pyclaw_backend.py` pins it).
- `"sharpclaw"` — method of lines with WENO5 reconstruction (`lim_type=2`) and an SSP
  Runge–Kutta integrator (`SSP104` by default), `num_ghost = 3`. The higher-order reference
  for the mutual-convergence test (V18a) and for the own WENO5 later (V19).

A PDE class tells the backend how PyClaw sees it through a `PyClawSpec` (`pde.pyclaw_spec`):
the Riemann solver (a Python callable with Clawpack's `rp(q_l, q_r, aux_l, aux_r,
problem_data)` signature, or a compiled `clawpack.riemann` module), the number of waves, the
kernel language, and per-scheme solver options (limiter, CFL numbers, transverse waves).
PyClaw runs only Fortran kernels in more than one dimension [code: clawpack 5.14.0,
`pyclaw/classic/solver.py:215` "Only Fortran kernels are supported in multi-D.",
`pyclaw/sharpclaw/solver.py:890` (2-D) and `:975` (3-D)], so a Python-kernel family is 1-D
on PyClaw and a 2-D family needs a compiled Riemann solver.

Boundary conditions: the three `BCName`s map onto `pyclaw.BC.{periodic, extrap, wall}` per
axis and side; `wall` relies on the Riemann solver's convention for which component is the
normal momentum (Clawpack: component `1 + axis`), which is what the shipped Euler/SW solvers
assume."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from context_flux_no.simulations.backend import BCs, SolverBackend, SolverFailure
from context_flux_no.simulations.grid import Grid


Scheme = Literal["classic", "sharpclaw"]


@dataclass(frozen=True)
class PyClawSpec:
    """How PyClaw solves one PDE class; see the module docstring."""

    riemann_solver: Any
    num_waves: int
    kernel_language: Literal["Python", "Fortran"] = "Python"
    fwave: bool = False
    problem_data: Mapping[str, Any] = field(default_factory=dict)
    classic_options: Mapping[str, Any] = field(default_factory=dict)
    sharpclaw_options: Mapping[str, Any] = field(default_factory=dict)


_LIMITERS = {"minmod": 1, "superbee": 2, "vanleer": 3, "MC": 4, "none": 0}


def _bc_enum(name: str) -> int:
    from clawpack import pyclaw

    return {
        "periodic": pyclaw.BC.periodic,
        "extrap": pyclaw.BC.extrap,
        "wall": pyclaw.BC.wall,
    }[name]


class PyClawBackend(SolverBackend):
    name = "pyclaw"

    def __init__(
        self,
        scheme: Scheme = "classic",
        *,
        verbose: bool = False,
        **solver_overrides: Any,
    ) -> None:
        """`solver_overrides` are set on the PyClaw solver after the spec's per-scheme
        options (e.g. `cfl_desired=0.5`, `weno_order=5`, `time_integrator="SSP33"`,
        `limiters="MC"`)."""
        if scheme not in ("classic", "sharpclaw"):
            raise ValueError(f"scheme must be 'classic' or 'sharpclaw', got {scheme!r}")
        self.scheme: Scheme = scheme
        self.verbose = verbose
        self.solver_overrides = dict(solver_overrides)

    # ------------------------------------------------------------------ setup ----
    def build_solver(self, pde: Any, n_spatial_dims: int) -> Any:
        from clawpack import pyclaw

        spec: PyClawSpec = pde.pyclaw_spec
        if n_spatial_dims not in (1, 2, 3):
            raise ValueError("PyClaw supports 1-, 2- and 3-D grids")
        if n_spatial_dims > 1 and spec.kernel_language != "Fortran":
            raise ValueError(
                f"{type(pde).__name__}: PyClaw runs only Fortran kernels in "
                f"{n_spatial_dims}-D (clawpack 5.14.0, pyclaw/classic/solver.py:215, "
                "pyclaw/sharpclaw/solver.py:890)"
            )
        if self.scheme == "classic":
            cls = {
                1: pyclaw.ClawSolver1D,
                2: pyclaw.ClawSolver2D,
                3: pyclaw.ClawSolver3D,
            }[n_spatial_dims]
            options = dict(spec.classic_options)
        else:
            if n_spatial_dims == 3:
                raise ValueError("SharpClaw has no 3-D solver in PyClaw")
            cls = {1: pyclaw.SharpClawSolver1D, 2: pyclaw.SharpClawSolver2D}[
                n_spatial_dims
            ]
            options = {"lim_type": 2, "weno_order": 5, **spec.sharpclaw_options}
        solver = cls(spec.riemann_solver)
        solver.num_eqn = pde.n_eqns
        solver.num_waves = spec.num_waves
        solver.kernel_language = spec.kernel_language
        solver.fwave = spec.fwave
        for key, value in (options | self.solver_overrides).items():
            if key == "limiters" and isinstance(value, str):
                value = _LIMITERS[value]
            setattr(solver, key, value)
        return solver

    # ------------------------------------------------------------------ solve ----
    def solve(
        self,
        pde: Any,
        u0: np.ndarray,
        grid: Grid,
        t_out: np.ndarray,
        bcs: BCs,
    ) -> np.ndarray:
        from clawpack import pyclaw

        t_out = self.check_inputs(pde, u0, grid, t_out)
        if not self.verbose:  # https://www.clawpack.org/pyclaw/output.html#logging
            logging.getLogger("pyclaw").setLevel(logging.CRITICAL)

        solver = self.build_solver(pde, grid.n_spatial_dims)
        for axis, (lo, hi) in enumerate(bcs):
            solver.bc_lower[axis] = _bc_enum(lo)
            solver.bc_upper[axis] = _bc_enum(hi)

        domain = pyclaw.Domain(
            [
                pyclaw.Dimension(lo, hi, n, name=name)
                for (lo, hi), n, name in zip(grid.x_spans, grid.Nxs, grid.axis_names)
            ]
        )
        state = pyclaw.State(domain, pde.n_eqns)
        state.problem_data.update(pde.pyclaw_spec.problem_data)
        state.q = np.asfortranarray(np.asarray(u0, dtype=float))
        state.t = float(t_out[0])

        controller = pyclaw.Controller()
        controller.solution = pyclaw.Solution(state, domain)
        controller.solver = solver
        controller.tfinal = float(t_out[-1])
        controller.output_style = 2  # explicit output times
        controller.out_times = [float(t) for t in t_out]
        controller.keep_copy = True
        controller.output_format = None
        controller.verbosity = 0
        try:
            controller.run()
        except Exception as e:  # noqa: BLE001 — PyClaw raises bare Exception for CFL/step limits
            # [code: clawpack 5.14.0 pyclaw/solver.py:651,664; pyclaw/controller.py:228-234]
            raise SolverFailure(f"pyclaw/{self.scheme}", str(e)) from e
        if not np.all(np.isfinite(controller.frames[-1].state.q)):
            raise SolverFailure(f"pyclaw/{self.scheme}", "non-finite state")

        u = np.stack([frame.state.q for frame in controller.frames], axis=0)
        if u.shape[0] != len(t_out):
            raise RuntimeError(
                f"PyClaw returned {u.shape[0]} frames for {len(t_out)} output times"
            )
        return u

    def describe(self) -> dict[str, Any]:
        import clawpack

        return {
            "name": self.name,
            "scheme": self.scheme,
            "clawpack_version": getattr(clawpack, "__version__", "unknown"),
            "solver_overrides": {
                k: _jsonable(v) for k, v in self.solver_overrides.items()
            },
        }


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

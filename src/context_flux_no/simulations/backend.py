"""`SolverBackend`: the interface every reference solver implements (ADR-0004, ADR-0011).

A backend turns an initial state on a `Grid` into a trajectory at requested output times. It
knows nothing about datasets, files or sampling: `simulations.generate` owns those. Boundary
conditions are *names* per (axis, side) until `fv/boundary.py` (Phase 2b) provides objects;
the names are the ones every backend can honour and they are recorded in `DatasetRecord`.

Concrete backends: `simulations.pyclaw.PyClawBackend` (Classic and SharpClaw; this branch),
`simulations.trixi.TrixiBackend` (Julia subprocess; pending), `simulations.jax.JaxBackend`
(own `fv/` stepper; Phase 4.7)."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, ClassVar, get_args, Literal

import numpy as np

from context_flux_no.simulations.grid import Grid


BCName = Literal["periodic", "extrap", "wall"]
BC_NAMES: tuple[str, ...] = get_args(BCName)
# per axis: (lower, upper)
BCs = tuple[tuple[BCName, BCName], ...]


def normalize_bcs(bc: str | Sequence[Any], n_spatial_dims: int) -> BCs:
    """`"periodic"` → periodic on every axis and side; `[("extrap", "wall"), ...]` as given.
    Periodic must be paired (a periodic lower face needs a periodic upper face)."""
    if isinstance(bc, str):
        pairs = [(bc, bc)] * n_spatial_dims
    else:
        pairs = [tuple(p) if not isinstance(p, str) else (p, p) for p in bc]
    if len(pairs) != n_spatial_dims:
        raise ValueError(
            f"expected {n_spatial_dims} boundary-condition pairs, got {len(pairs)}"
        )
    out = []
    for lo, hi in pairs:
        for name in (lo, hi):
            if name not in BC_NAMES:
                raise ValueError(
                    f"unknown boundary condition {name!r}; one of {BC_NAMES}"
                )
        if (lo == "periodic") != (hi == "periodic"):
            raise ValueError(
                "periodic boundaries must be periodic on both sides of an axis"
            )
        out.append((lo, hi))
    return tuple(out)  # ty: ignore[invalid-return-type]


class SolverFailure(RuntimeError):
    """A backend could not produce a trajectory for an otherwise valid request (CFL blow-up,
    step limit, non-finite state). `generate_dataset` counts these per trajectory; anything
    else that escapes a backend is a bug and propagates."""

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(f"{backend}: {reason}")
        self.backend = backend
        self.reason = reason


class SolverBackend(abc.ABC):
    """`solve(pde, u0, grid, t_out, bcs)`; subclasses set `name` and implement `solve`.
    `u0` is `[n_eqn, *grid.shape]` in conserved variables, `t_out` the strictly increasing
    output times whose first entry is the time of `u0`; the result is `[len(t_out), n_eqn,
    *grid.shape]` with `result[0] == u0` up to the backend's own representation."""

    name: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def solve(
        self,
        pde: Any,
        u0: np.ndarray,
        grid: Grid,
        t_out: np.ndarray,
        bcs: BCs,
    ) -> np.ndarray: ...

    def solve_batch(
        self,
        pde: Any,
        u0s: np.ndarray,
        grid: Grid,
        t_out: np.ndarray,
        bcs: BCs,
    ) -> np.ndarray:
        """`[batch, n_eqn, *N]` → `[batch, time, n_eqn, *N]`; a Python loop by default."""
        return np.stack([self.solve(pde, u0, grid, t_out, bcs) for u0 in u0s], axis=0)

    def describe(self) -> dict[str, Any]:
        """What `DatasetRecord.backend` stores: enough to rebuild this backend."""
        return {"name": self.name}

    @staticmethod
    def check_inputs(
        pde: Any, u0: np.ndarray, grid: Grid, t_out: np.ndarray
    ) -> np.ndarray:
        u0 = np.asarray(u0)
        if u0.shape != (pde.n_eqns, *grid.shape):
            raise ValueError(
                f"u0 has shape {u0.shape}, expected {(pde.n_eqns, *grid.shape)} for "
                f"{type(pde).__name__} on {grid}"
            )
        t_out = np.asarray(t_out, dtype=float)
        if t_out.ndim != 1 or len(t_out) < 2 or np.any(np.diff(t_out) <= 0):
            raise ValueError(
                "t_out must be a strictly increasing 1-D array of length >= 2"
            )
        return t_out

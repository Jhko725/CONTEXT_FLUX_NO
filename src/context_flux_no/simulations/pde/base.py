"""`AbstractPDEFamily`: the PDE-family contract of ADR-0001, growing tier by tier.

Today it holds the data-facing part of tier 0 — spatial dimension, the Well field layout
(`field_rank_names`), the physical `parameters`, primitive/conserved conversion where a family
has one — and how a reference backend solves the family (`pyclaw_spec`, a default `solve()`).
The numerics part of tier 0 (`flux(u, params, axis, x)`, `max_wave_speed`, `reflect`) is added
as Phase 2b–2e's `fv/` consumes it; Phase 4.1 is then the conformance check (`requires ⊆
capabilities` for every flux × family pairing), not a rewrite (ADR-0001, amended 2026-10-25)."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import numpy as np

from context_flux_no.simulations.backend import normalize_bcs, SolverBackend
from context_flux_no.simulations.grid import Grid
from context_flux_no.simulations.initial_conditions.base import InitialCondition


class AbstractPDEFamily(eqx.Module):
    """Base class of the PDE families under `simulations/pde/`.

    Subclasses provide `n_spatial_dims`, `field_rank_names` (Well layout: `(rank, name)` in
    state order; a rank-`r` field has `n_spatial_dims**r` components) and `parameters`. A
    family solved by PyClaw provides `pyclaw_spec`; the default `solve()` then runs the given
    backend (Classic PyClaw by default). A family with its own stepper overrides `solve()`."""

    n_spatial_dims: eqx.AbstractVar[int]
    field_rank_names: eqx.AbstractVar[tuple[tuple[int, str], ...]]

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, float]:
        """The physical parameters that vary across a dataset (the Well `scalars`)."""

    @property
    def coeffs(self) -> dict[str, float]:  # legacy name
        return self.parameters

    @property
    def n_eqns(self) -> int:
        return sum(self.n_spatial_dims**r for r, _ in self.field_rank_names)

    @property
    def pyclaw_spec(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} is not solved by PyClaw")

    def default_backend(self, *, verbose: bool = False) -> SolverBackend:
        from context_flux_no.simulations.pyclaw import PyClawBackend

        return PyClawBackend("classic", verbose=verbose)

    def solve(
        self,
        ic: InitialCondition,
        x_spans: Sequence[Sequence[float]],
        Nxs: Sequence[int],
        t_span: tuple[float, float],
        Nt: int,
        bc: str | Sequence[Any] = "periodic",
        *,
        backend: SolverBackend | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """`(u[Nt+1, n_eqn, *N], t[Nt+1], centers per axis)`: `ic` evaluated at the cell
        centers and evolved by `backend` over `linspace(*t_span, Nt + 1)`. (The PyClaw
        backend takes point values at centers as the cell state — the ICLR convention; see
        F12 for the averages-vs-points question.)"""
        if kwargs:
            raise TypeError(f"unexpected solve() arguments: {sorted(kwargs)}")
        grid = Grid(x_spans, Nxs)
        t_out = np.linspace(t_span[0], t_span[1], Nt + 1, endpoint=True)
        if ic.n_eqn != self.n_eqns or ic.n_spatial_dims != self.n_spatial_dims:
            raise ValueError(
                f"initial condition is {ic.n_eqn} × {ic.n_spatial_dims}-D, "
                f"{type(self).__name__} is {self.n_eqns} × {self.n_spatial_dims}-D"
            )
        u0 = ic.at_points(grid.meshgrid())
        backend = backend or self.default_backend(verbose=verbose)
        bcs = normalize_bcs(bc, grid.n_spatial_dims)
        u = backend.solve(self, u0, grid, t_out, bcs)
        return u, t_out, grid.centers

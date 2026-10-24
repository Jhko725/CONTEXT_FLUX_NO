"""The minimal PDE interface the simulation backends and the dataset writer need.

This is *not* Phase 4.1's `PDEFamily` (ADR-0001: `flux`, `max_wave_speed`, tiers); it is the
smallest common shape of the existing PDE classes — number of equations, the Well field
layout, the physical parameters, and how a reference backend solves it — so that one
`generate_dataset` and one `PyClawBackend` serve every family. `PDEFamily` replaces it in
Phase 4.1."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import equinox as eqx
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float

from context_flux_no.simulations.backend import normalize_bcs, SolverBackend
from context_flux_no.simulations.grid import Grid


class WellSchema(TypedDict):
    scalars: dict[str, np.ndarray]
    t0_fields: dict[str, np.ndarray]
    t1_fields: dict[str, np.ndarray]
    t2_fields: dict[str, np.ndarray]


ICFactory = Callable[
    [Sequence[np.ndarray]], np.ndarray
]  # cell centres per axis → [n_eqn, *N]


class AbstractHyperbolicConservationLaw(eqx.Module):
    """Base class of the PDE classes under `simulations/pde/`.

    Subclasses provide `n_spatial_dims`, `field_rank_names` (Well layout: `(rank, name)` in
    state order; a rank-`r` field has `n_spatial_dims**r` components) and `parameters`. A
    class solved by PyClaw provides `pyclaw_spec`; the default `solve()` then runs the given
    backend (Classic PyClaw by default). A class with its own stepper overrides `solve()`."""

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
        ic_factory: ICFactory,
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
        """`(u[Nt+1, n_eqn, *N], t[Nt+1], centres per axis)`: the initial condition from
        `ic_factory(centres)` evolved by `backend` over `linspace(*t_span, Nt + 1)`."""
        if kwargs:
            raise TypeError(f"unexpected solve() arguments: {sorted(kwargs)}")
        grid = Grid.from_sequences(x_spans, Nxs)
        t_out = np.linspace(t_span[0], t_span[1], Nt + 1, endpoint=True)
        u0 = np.asarray(ic_factory(grid.centers), dtype=float)
        backend = backend or self.default_backend(verbose=verbose)
        bcs = normalize_bcs(bc, grid.n_spatial_dims)
        u = backend.solve(self, u0, grid, t_out, bcs)
        return u, t_out, grid.centers

    def solution_to_well_schema(
        self, u: Float[Array, "time dim *x_grids"]
    ) -> WellSchema:
        well_schema = WellSchema(scalars={}, t0_fields={}, t1_fields={}, t2_fields={})

        idx_start = 0
        for rank, f_name in self.field_rank_names:
            idx_end = idx_start + self.n_spatial_dims**rank
            field_data = rearrange(u[:, idx_start:idx_end], "T C ... -> T ... C")
            if rank == 0:
                field_data = np.squeeze(field_data, axis=-1)
            well_schema[f"t{rank}_fields"][f_name] = field_data
            idx_start = idx_end

        for param_name, val in self.parameters.items():
            well_schema["scalars"][param_name] = val
        return well_schema

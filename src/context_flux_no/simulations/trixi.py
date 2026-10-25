"""`TrixiBackend`: Trixi.jl's DGSEM as a `SolverBackend` (ADR-0004 amended, ADR-0011), the
high-order reference for the cross-code test V18 and for the own schemes later (V19).

**What it computes** (change document `2026-10-25-phase2a-trixi-backend`, D-TX). One DG
element per finite-volume cell of the request's `Grid` (a `StructuredMesh`, so any cell count
goes), polynomial degree `polydeg` on Lobatto–Legendre nodes, a Lax–Friedrichs surface flux,
the Hennemann–Gassner shock-capturing blend, SSPRK33 at a fixed CFL number through Trixi's
`StepsizeCallback`, stopping exactly at the output times. The initial condition is evaluated
*on the DG nodes* (no interpolation from cell values), and the output at each time is the
element mean by LGL quadrature on the same grid as the FV backends — so a Trixi trajectory and
a PyClaw trajectory of the same request are arrays of the same shape holding the same
quantity up to each scheme's error (see F12 for what the FV stores declare).

**Scope.** Periodic boundaries only (`ValueError` otherwise; V17 brings Trixi's boundary
conditions). The six families with a `trixi_spec`: `CubicFlux1D`, `CubicFlux2D`,
`SineFlux1D`, `ParametrizedShallowWater1D` (custom equations in `ContextFluxTrixi`),
`Burgers1D` and `Euler2D` (Trixi's own). `ViscousBurgers1D` waits for tier 3.

**Transport** (D-TR). Julia runs in-process through `juliacall`; this module sets the
process environment *before* the first `import juliacall` (signals handled by Julia, threads
`auto`, and juliapkg's *shared project* at `reference/trixi/` so Julia itself, the committed
`Manifest.toml` and the `dev`-ed `ContextFluxTrixi` package are what runs). Arrays cross the
boundary in both directions as plain numpy arrays; nothing calls back from Julia into Python,
so the GIL is held for the whole solve and multithreading is Julia's own. The first call in a
process pays Julia's start-up and precompilation; `scripts/setup_julia.py` does that once.
PySR imports juliacall before numpy to dodge shared-library conflicts; here the import is
lazy (first `solve`), which the laptop runs (ledger 5) confirm or refute **[deduced]**.

**Julia interface** (`reference/trixi/ContextFluxTrixi/`; written on the laptop, see the change
document's *Laptop steps*). Every argument arrives as what PythonCall makes of a Python
object — `PyArray` for numpy arrays, `PyDict` for dicts, `String`, `Int`, `Float64` — so the
Julia functions are typed on `AbstractArray`/`AbstractDict` and copy what they keep. Shapes
are stated in index order; memory order differs between the two languages and both sides may
copy.

- `node_coordinates(x_spans::AbstractMatrix [d, 2], Nxs::AbstractVector{<:Integer} [d],
  polydeg::Integer) -> NTuple{d, Array{Float64}}`: one coordinate array per axis, each of shape
  `(polydeg + 1,) * d + tuple(Nxs)` — node indices first, then element indices in the mesh's
  own linear order along each axis. Python evaluates the initial condition on these with
  `InitialCondition.at_points`.
- `solve(equations::AbstractString, parameters::AbstractDict{String, Float64},
  u0_nodes::AbstractArray [n_eqn, (polydeg + 1,) * d..., Nxs...], x_spans [d, 2],
  Nxs [d], t_out::AbstractVector{Float64}, polydeg::Integer, cfl::Float64,
  surface_flux::AbstractString, shock_capturing::Bool) -> Array{Float64} [n_t, n_eqn, Nxs...]`:
  element means at every `t_out` (the first equals the mean of `u0_nodes`). `equations` is a
  constructor name from `TrixiSpec.equations` and `parameters` its keyword arguments; the
  package owns the name → constructor table. A failed integration (non-finite state, step
  limit, Trixi error) throws `ContextFluxTrixi.SolveFailed(msg)`; Python turns it into
  `SolverFailure("trixi", msg)`. Anything else propagates.
- `versions() -> Dict{String, String}`: `"trixi"`, `"julia"`, `"threads"` for `describe()`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from context_flux_no.simulations.backend import BCs, SolverBackend, SolverFailure
from context_flux_no.simulations.grid import Grid
from context_flux_no.simulations.initial_conditions.base import InitialCondition


REFERENCE_TRIXI = Path(__file__).resolve().parents[3] / "reference" / "trixi"
"""The Julia project (`Project.toml` + `Manifest.toml`) juliacall runs in."""

SurfaceFlux = Literal["lax_friedrichs"]

_ENVIRONMENT = {
    # Julia handles signals so that its GC and thread machinery work in-process
    # [cited: JuliaCall docs, "Configuration"]
    "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
    "PYTHON_JULIACALL_THREADS": "auto",
    # juliapkg's shared-project mode: use this directory's Project/Manifest as they are
    # [code: juliapkg 0.1.28, state.py:63-69, deps.py:560-607]
    "PYTHON_JULIAPKG_PROJECT": str(REFERENCE_TRIXI),
}


@dataclass(frozen=True)
class TrixiSpec:
    """How `ContextFluxTrixi` sees one PDE class: the equations constructor's name (one of
    the package's table) and the keyword arguments it takes."""

    equations: str
    parameters: Mapping[str, float] = field(default_factory=dict)


class TrixiUnavailable(RuntimeError):
    """Julia, the `reference/trixi` project or `ContextFluxTrixi` is not set up on this
    machine (`scripts/setup_julia.py` does that)."""


def available() -> bool:
    """Whether `scripts/setup_julia.py` has set the Julia side up here (juliapkg's record of
    a resolved project; cheap, no Julia start-up). Tests that need Trixi skip on `False`."""
    return (REFERENCE_TRIXI / "pyjuliapkg" / "meta.json").is_file()


def configure_environment() -> None:
    """Set the juliacall/juliapkg environment (values already set win); must run before
    the first `import juliacall` or `import juliapkg` in the process."""
    for key, value in _ENVIRONMENT.items():
        os.environ.setdefault(key, value)


def _julia() -> Any:
    """The `ContextFluxTrixi` module, starting Julia on first use."""
    if not (REFERENCE_TRIXI / "Project.toml").is_file():
        raise TrixiUnavailable(f"no Julia project at {REFERENCE_TRIXI}")
    if not available():
        raise TrixiUnavailable(
            f"Julia is not set up for {REFERENCE_TRIXI}; run `uv run python "
            "scripts/setup_julia.py`"
        )
    configure_environment()
    try:
        from juliacall import Main as jl
    except ImportError as e:
        raise TrixiUnavailable("juliacall is not installed (dev group)") from e
    jl.seval("using ContextFluxTrixi")
    return jl.ContextFluxTrixi


class TrixiBackend(SolverBackend):
    name = "trixi"

    def __init__(
        self,
        polydeg: int = 3,
        cfl: float = 0.5,
        surface_flux: SurfaceFlux = "lax_friedrichs",
        shock_capturing: bool = True,
    ) -> None:
        if polydeg < 1:
            raise ValueError("polydeg must be >= 1")
        if surface_flux != "lax_friedrichs":
            raise ValueError(f"unknown surface flux {surface_flux!r}")
        self.polydeg = int(polydeg)
        self.cfl = float(cfl)
        self.surface_flux: SurfaceFlux = surface_flux
        self.shock_capturing = bool(shock_capturing)

    def solve(
        self,
        pde: Any,
        ic: InitialCondition,
        grid: Grid,
        t_out: np.ndarray,
        bcs: BCs,
    ) -> np.ndarray:
        t_out = self.check_inputs(pde, ic, grid, t_out)
        if any(name != "periodic" for pair in bcs for name in pair):
            raise ValueError("TrixiBackend supports periodic boundaries only (V17)")
        spec: TrixiSpec = pde.trixi_spec
        cft = _julia()

        x_spans = np.asarray(grid.x_spans, dtype=float)
        Nxs = np.asarray(grid.Nxs, dtype=np.int64)
        nodes = [
            np.asarray(c, dtype=float)
            for c in cft.node_coordinates(x_spans, Nxs, self.polydeg)
        ]
        u0_nodes = np.ascontiguousarray(ic.at_points(nodes))
        try:
            u = cft.solve(
                spec.equations,
                {k: float(v) for k, v in spec.parameters.items()},
                u0_nodes,
                x_spans,
                Nxs,
                t_out,
                self.polydeg,
                self.cfl,
                self.surface_flux,
                self.shock_capturing,
            )
        except Exception as e:  # noqa: BLE001 — the Julia error type is only known by name
            if "SolveFailed" in type(e).__name__ or "SolveFailed" in str(e):
                raise SolverFailure(self.name, str(e)) from e
            raise
        u = np.array(u, dtype=float, order="C")
        if u.shape != (len(t_out), pde.n_eqns, *grid.shape):
            raise RuntimeError(
                f"ContextFluxTrixi returned {u.shape}, expected "
                f"{(len(t_out), pde.n_eqns, *grid.shape)}"
            )
        if not np.all(np.isfinite(u)):
            raise SolverFailure(self.name, "non-finite state")
        return u

    def describe(self) -> dict[str, Any]:
        versions = {str(k): str(v) for k, v in dict(_julia().versions()).items()}
        return {
            "name": self.name,
            "polydeg": self.polydeg,
            "cfl": self.cfl,
            "surface_flux": self.surface_flux,
            "shock_capturing": self.shock_capturing,
            "trixi_version": versions["trixi"],
            "julia_version": versions["julia"],
            "julia_threads": versions["threads"],
        }

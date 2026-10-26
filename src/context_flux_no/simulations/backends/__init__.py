"""Reference solvers behind one `SolverBackend` (ADR-0011): `base` (the interface, boundary
names, `SolverFailure`), `pyclaw` (Classic, SharpClaw), `trixi` (DGSEM through juliacall);
`jax` (the own `fv/` stepper) arrives in Phase 4.7."""

from .base import (
    BC_NAMES as BC_NAMES,
    BCName as BCName,
    BCs as BCs,
    normalize_bcs as normalize_bcs,
    SolverBackend as SolverBackend,
    SolverFailure as SolverFailure,
)
from .pyclaw import PyClawBackend as PyClawBackend, PyClawSpec as PyClawSpec
from .trixi import TrixiBackend as TrixiBackend, TrixiSpec as TrixiSpec

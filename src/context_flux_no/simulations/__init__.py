"""Reference solvers and data generation (ADR-0004, ADR-0011): `SolverBackend`s, the Well
writer with its `DatasetRecord`, `generate_dataset`, and the PDE classes under `pde/`."""

from .backends import (
    BC_NAMES as BC_NAMES,
    BCName as BCName,
    BCs as BCs,
    normalize_bcs as normalize_bcs,
    PyClawBackend as PyClawBackend,
    PyClawSpec as PyClawSpec,
    SolverBackend as SolverBackend,
    SolverFailure as SolverFailure,
    TrixiBackend as TrixiBackend,
    TrixiSpec as TrixiSpec,
)
from .generate import (
    generate_dataset as generate_dataset,
    sample_coefficients as sample_coefficients,
)
from .grid import Grid as Grid
from .record import DatasetRecord as DatasetRecord
from .writer import WellWriter as WellWriter

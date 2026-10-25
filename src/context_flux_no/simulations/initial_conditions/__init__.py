"""Initial conditions for data generation and verification (ADR-0011 §Initial conditions):
an `InitialCondition` is a function of space evaluated by each backend at the points its
scheme needs (`at_points`); an `InitialConditionSampler` draws one from a key on a domain."""

from .base import (
    Conserved as Conserved,
    FromFunction as FromFunction,
    InitialCondition as InitialCondition,
    InitialConditionSampler as InitialConditionSampler,
    Multichannel as Multichannel,
    Stacked as Stacked,
)
from .fourier import TruncatedFourier1D as TruncatedFourier1D
from .grf import (
    GaussianRandomField as GaussianRandomField,
    GaussianRandomFieldCondition as GaussianRandomFieldCondition,
)
from .step import (
    PeriodicRandomStepFunction1D as PeriodicRandomStepFunction1D,
    PeriodicRandomStepFunction2D as PeriodicRandomStepFunction2D,
    PeriodicStep1D as PeriodicStep1D,
    PeriodicVoronoi2D as PeriodicVoronoi2D,
)

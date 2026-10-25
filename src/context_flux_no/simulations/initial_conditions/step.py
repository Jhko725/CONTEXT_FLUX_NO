"""Piecewise-constant random initial conditions on periodic domains.

1-D: `num_jumps` jump positions drawn uniformly in `[x0, x1)`, a value per interval, the
interval wrapping around the periodic boundary. 2-D: a Voronoi tessellation of the torus from
`num_regions` seeds drawn uniformly in the domain, one `channels`-vector per region, so
discontinuities have arbitrary orientation and every channel shares the geometry. Both are
functions of position (positions are drawn in physical coordinates), not of a grid.

Before 2026-10-25 the 1-D jumps were drawn as *grid indices* and the 2-D domain was inferred
from the grid; the drawn distribution now depends only on the domain (change document
`2026-10-25-phase2a-trixi-backend.md`, Design)."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PRNGKeyArray

from context_flux_no.simulations.initial_conditions.base import (
    Coords,
    InitialCondition,
    InitialConditionSampler,
)


class PeriodicStep1D(InitialCondition):
    """`u0(x) = values[region(x)]`, `region` = number of jumps at or before `x` (periodic
    wrap: points before the first jump belong to the last interval)."""

    def __init__(
        self, jumps: Float[np.ndarray, " J"], values: Float[np.ndarray, " J"], x_span
    ):
        self.jumps = np.sort(np.asarray(jumps, dtype=float))
        self.values = np.asarray(values, dtype=float)
        self.x_span = (float(x_span[0]), float(x_span[1]))
        if self.jumps.shape != self.values.shape or self.jumps.ndim != 1:
            raise ValueError("jumps and values must be 1-D of equal length")

    @property
    def n_eqn(self) -> int:
        return 1

    @property
    def n_spatial_dims(self) -> int:
        return 1

    def at_points(self, coords: Coords) -> Float[np.ndarray, "1 *shape"]:
        (x,) = self.check_coords(coords)
        lo, hi = self.x_span
        xp = lo + np.mod(x - lo, hi - lo)  # into the periodic domain
        region = (
            np.searchsorted(self.jumps, xp, side="right") - 1
        )  # -1 before the first jump
        region = np.mod(region, len(self.jumps))
        return self.values[region][None]


class PeriodicRandomStepFunction1D(InitialConditionSampler):
    def __init__(
        self,
        num_jumps_min: int = 1,
        num_jumps_max: int = 5,
        value_min: float = -1.0,
        value_max: float = 1.0,
    ) -> None:
        if num_jumps_min < 1 or num_jumps_max < num_jumps_min:
            raise ValueError("need 1 <= num_jumps_min <= num_jumps_max")
        self.num_jumps_min = int(num_jumps_min)
        self.num_jumps_max = int(num_jumps_max)
        self.value_min = float(value_min)
        self.value_max = float(value_max)

    @property
    def n_eqn(self) -> int:
        return 1

    @property
    def n_spatial_dims(self) -> int:
        return 1

    def sample(
        self, key: PRNGKeyArray, x_spans: Sequence[Sequence[float]]
    ) -> PeriodicStep1D:
        ((lo, hi),) = (tuple(s) for s in x_spans)
        key_n, key_pos, key_val = jax.random.split(key, 3)
        num_jumps = int(
            jax.random.randint(key_n, (), self.num_jumps_min, self.num_jumps_max + 1)
        )
        jumps = jax.random.uniform(key_pos, (num_jumps,), minval=lo, maxval=hi)
        values = jax.random.uniform(
            key_val, (num_jumps,), minval=self.value_min, maxval=self.value_max
        )
        return PeriodicStep1D(np.asarray(jumps), np.asarray(values), (lo, hi))


class PeriodicVoronoi2D(InitialCondition):
    """`u0(x) = values[:, nearest seed]` with the periodic (torus) distance."""

    def __init__(
        self,
        seeds: Float[np.ndarray, "R 2"],
        values: Float[np.ndarray, "C R"],
        x_spans: Sequence[Sequence[float]],
    ) -> None:
        self.seeds = np.asarray(seeds, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.x_spans = tuple((float(lo), float(hi)) for lo, hi in x_spans)
        if self.seeds.ndim != 2 or self.seeds.shape[1] != 2 or len(self.x_spans) != 2:
            raise ValueError("seeds must be [R, 2] on a 2-D domain")
        if self.values.shape[1] != self.seeds.shape[0]:
            raise ValueError("values must be [channels, R]")

    @property
    def n_eqn(self) -> int:
        return self.values.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        return 2

    def at_points(self, coords: Coords) -> Float[np.ndarray, "n_eqn *shape"]:
        x, y = self.check_coords(coords)
        dist_sq = np.zeros(x.shape + (self.seeds.shape[0],))
        for axis, c in enumerate((x, y)):
            lo, hi = self.x_spans[axis]
            cp = lo + np.mod(c - lo, hi - lo)  # into the periodic domain
            d = np.abs(cp[..., None] - self.seeds[:, axis])  # [*shape, R]
            d = np.minimum(d, (hi - lo) - d)  # torus distance along this axis
            dist_sq += d**2
        region = np.argmin(dist_sq, axis=-1)
        return self.values[:, region]


class PeriodicRandomStepFunction2D(InitialConditionSampler):
    def __init__(
        self,
        channels: int,
        num_regions_min: int = 1,
        num_regions_max: int = 8,
        value_min: float | Sequence[float] = -1.0,
        value_max: float | Sequence[float] = 1.0,
    ) -> None:
        self.channels = int(channels)
        if num_regions_min < 1 or num_regions_max < num_regions_min:
            raise ValueError("need 1 <= num_regions_min <= num_regions_max")
        self.num_regions_min = int(num_regions_min)
        self.num_regions_max = int(num_regions_max)
        self.value_min = np.broadcast_to(np.asarray(value_min, float), (self.channels,))
        self.value_max = np.broadcast_to(np.asarray(value_max, float), (self.channels,))

    @property
    def n_eqn(self) -> int:
        return self.channels

    @property
    def n_spatial_dims(self) -> int:
        return 2

    def sample(
        self, key: PRNGKeyArray, x_spans: Sequence[Sequence[float]]
    ) -> PeriodicVoronoi2D:
        spans = tuple((float(lo), float(hi)) for lo, hi in x_spans)
        if len(spans) != 2:
            raise ValueError("PeriodicRandomStepFunction2D needs a 2-D domain")
        key_n, key_seed, key_val = jax.random.split(key, 3)
        num_regions = int(
            jax.random.randint(
                key_n, (), self.num_regions_min, self.num_regions_max + 1
            )
        )
        unit = jax.random.uniform(key_seed, (num_regions, 2))
        lo = jnp.array([s[0] for s in spans])
        hi = jnp.array([s[1] for s in spans])
        seeds = lo + (hi - lo) * unit
        values = jax.random.uniform(
            key_val,
            (self.channels, num_regions),
            minval=jnp.asarray(self.value_min)[:, None],
            maxval=jnp.asarray(self.value_max)[:, None],
        )
        return PeriodicVoronoi2D(np.asarray(seeds), np.asarray(values), spans)

"""A truncated Fourier series on `[x0, x0 + L)` as an initial condition (analytic; periodic),
and the sampler that draws its coefficients."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.simulations.initial_conditions.base import (
    Coords,
    InitialCondition,
    InitialConditionSampler,
)


class TruncatedFourier1D(InitialCondition):
    def __init__(
        self,
        an: Float[Array, " K"],
        bn: Float[Array, " K"],
        a0: float = 0.0,
        L: float = 1.0,
    ) -> None:
        super().__init__(n_eqn=1, n_spatial_dims=1)
        self.an = np.asarray(an, dtype=float)
        self.bn = np.asarray(bn, dtype=float)
        self.a0 = float(a0)
        self.L = float(L)
        if self.an.shape != self.bn.shape or self.an.ndim != 1:
            raise ValueError("an and bn must be 1-D of equal length")

    @property
    def num_modes(self) -> int:
        return len(self.an)

    @property
    def wavenumbers(self) -> Float[np.ndarray, " K"]:
        return np.arange(1, self.num_modes + 1) * 2 * np.pi / self.L

    def at_points(self, coords: Coords) -> Float[np.ndarray, "1 *shape"]:
        (x,) = self.check_coords(coords)
        kx = x[..., None] * self.wavenumbers
        series = np.mean(self.an * np.cos(kx) + self.bn * np.sin(kx), axis=-1)
        return (self.a0 + series / np.sqrt(2))[None]


class RandomTruncatedFourier1D(InitialConditionSampler):
    """`TruncatedFourier1D` with `an`, `bn` drawn uniformly in `coeff_range` and `a0` in
    `offset_range` (0 if `None`); the period is the domain length."""

    def __init__(
        self,
        num_modes: int,
        coeff_range: tuple[float, float] = (-1.0, 1.0),
        offset_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(n_eqn=1, n_spatial_dims=1)
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        self.num_modes = int(num_modes)
        self.coeff_range = (float(coeff_range[0]), float(coeff_range[1]))
        self.offset_range = (
            None
            if offset_range is None
            else (float(offset_range[0]), float(offset_range[1]))
        )

    def sample(
        self, key: PRNGKeyArray, x_spans: Sequence[Sequence[float]]
    ) -> TruncatedFourier1D:
        ((lo, hi),) = (tuple(s) for s in x_spans)
        key_coeff, key_offset = jax.random.split(key, 2)
        an, bn = np.asarray(
            jax.random.uniform(
                key_coeff,
                shape=(2, self.num_modes),
                minval=self.coeff_range[0],
                maxval=self.coeff_range[1],
            )
        )
        a0 = (
            0.0
            if self.offset_range is None
            else float(
                jax.random.uniform(
                    key_offset, minval=self.offset_range[0], maxval=self.offset_range[1]
                )
            )
        )
        return TruncatedFourier1D(an, bn, a0, L=hi - lo)

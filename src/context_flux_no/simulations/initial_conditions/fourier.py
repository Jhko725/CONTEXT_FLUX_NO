"""A truncated Fourier series on `[0, L)` as an initial condition (analytic; periodic)."""

from __future__ import annotations

import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.simulations.initial_conditions.base import Coords, InitialCondition


class TruncatedFourier1D(InitialCondition):
    def __init__(
        self,
        an: Float[Array, " K"],
        bn: Float[Array, " K"],
        a0: float = 0.0,
        L: float = 1.0,
    ) -> None:
        self.an = np.asarray(an, dtype=float)
        self.bn = np.asarray(bn, dtype=float)
        self.a0 = float(a0)
        self.L = float(L)
        if self.an.shape != self.bn.shape or self.an.ndim != 1:
            raise ValueError("an and bn must be 1-D of equal length")

    @property
    def n_eqn(self) -> int:
        return 1

    @property
    def n_spatial_dims(self) -> int:
        return 1

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

    @classmethod
    def with_uniform_rand_coeffs(
        cls,
        num_modes: int,
        L: float = 1.0,
        coeff_range: tuple[float, float] = (1, 1),
        offset_range: tuple[float, float] | None = None,
        *,
        key: PRNGKeyArray,
    ) -> TruncatedFourier1D:
        key_coeff, key_offset = jax.random.split(key, 2)
        an, bn = jax.random.uniform(
            key_coeff,
            shape=(2, num_modes),
            minval=coeff_range[0],
            maxval=coeff_range[1],
        )
        a0 = (
            0.0
            if offset_range is None
            else float(
                jax.random.uniform(
                    key_offset, minval=offset_range[0], maxval=offset_range[1]
                )
            )
        )
        return cls(an, bn, a0, L)

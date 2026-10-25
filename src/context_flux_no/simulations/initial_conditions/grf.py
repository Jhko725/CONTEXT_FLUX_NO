"""Gaussian random fields as initial conditions, through gstools' Fourier generator (periodic
by construction, `period` = the domain length; evaluable at any point, so the drawn field is a
function and not a grid array)."""

from __future__ import annotations

from collections.abc import Sequence

import gstools as gs
import jax
import numpy as np
from jaxtyping import Float, PRNGKeyArray

from context_flux_no.simulations.initial_conditions.base import (
    Coords,
    InitialCondition,
    InitialConditionSampler,
)


class GaussianRandomFieldCondition(InitialCondition):
    """One drawn field per channel: `scales * srf(x; seed) + means`."""

    def __init__(
        self,
        srfs: Sequence[gs.SRF],
        seeds: Sequence[int],
        means: Float[np.ndarray, " channels"],
        scales: Float[np.ndarray, " channels"],
        n_spatial_dims: int,
    ) -> None:
        self.srfs = tuple(srfs)
        self.seeds = tuple(int(s) for s in seeds)
        self.means = np.asarray(means, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self._n_spatial_dims = int(n_spatial_dims)

    @property
    def n_eqn(self) -> int:
        return len(self.srfs)

    @property
    def n_spatial_dims(self) -> int:
        return self._n_spatial_dims

    def at_points(self, coords: Coords) -> Float[np.ndarray, "n_eqn *shape"]:
        arrays = self.check_coords(coords)
        shape = arrays[0].shape
        pos = tuple(a.ravel() for a in arrays)
        fields = [
            srf(pos, seed=seed, mesh_type="unstructured").reshape(shape)
            for srf, seed in zip(self.srfs, self.seeds)
        ]
        broadcast = (-1,) + (1,) * len(shape)
        return np.stack(fields, axis=0) * self.scales.reshape(
            broadcast
        ) + self.means.reshape(broadcast)


class GaussianRandomField(InitialConditionSampler):
    """One covariance model per channel (all of one dimension), optional gstools
    normalizers (`transforms`), affine `means`/`scales` per channel. `period` must equal
    the domain length of every axis, which `sample` checks."""

    def __init__(
        self,
        covariance_fns: Sequence[gs.CovModel],
        means: Sequence[float] | None = None,
        scales: Sequence[float] | None = None,
        period: float = 1.0,
        num_modes: int = 32,
        transforms: Sequence[object | None] | None = None,
    ) -> None:
        dims = {cov.dim for cov in covariance_fns}
        if len(covariance_fns) == 0 or len(dims) != 1:
            raise ValueError("covariance_fns must be non-empty and of one dimension")
        self.covariance_fns = tuple(covariance_fns)
        self._n_spatial_dims = int(dims.pop())
        self.period = float(period)
        self.num_modes = int(num_modes)
        n = len(self.covariance_fns)
        normalizers = tuple(transforms) if transforms is not None else (None,) * n
        if len(normalizers) != n:
            raise ValueError("one transform per covariance function (or None)")
        self.means = np.ones(n) * 0.0 if means is None else np.asarray(means, float)
        self.scales = np.ones(n) if scales is None else np.asarray(scales, float)
        if self.means.shape != (n,) or self.scales.shape != (n,):
            raise ValueError("means and scales must have one entry per channel")
        self.srfs = tuple(
            gs.SRF(
                cov,
                normalizer=norm,
                generator="Fourier",
                period=self.period,
                mode_no=self.num_modes,
            )
            for cov, norm in zip(self.covariance_fns, normalizers)
        )

    @property
    def n_eqn(self) -> int:
        return len(self.covariance_fns)

    @property
    def n_spatial_dims(self) -> int:
        return self._n_spatial_dims

    def sample(
        self, key: PRNGKeyArray, x_spans: Sequence[Sequence[float]]
    ) -> GaussianRandomFieldCondition:
        if len(x_spans) != self.n_spatial_dims:
            raise ValueError(
                f"sampler is {self.n_spatial_dims}-D, domain has {len(x_spans)} axes"
            )
        for lo, hi in x_spans:
            if not np.isclose(hi - lo, self.period):
                raise ValueError(
                    f"GaussianRandomField period {self.period} != domain length {hi - lo}"
                )
        with jax.default_device(jax.devices("cpu")[0]):
            seeds = [int(jax.random.bits(k)) for k in jax.random.split(key, self.n_eqn)]
        return GaussianRandomFieldCondition(
            self.srfs, seeds, self.means, self.scales, self.n_spatial_dims
        )

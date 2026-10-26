from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def generate_periodic_random_step_function_1d(
    x: Float[Array, " Nx"],
    key: PRNGKeyArray,
    *,
    num_jumps_min: int = 1,
    num_jumps_max: int = 5,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> Float[Array, " Nx"]:
    """
    Generate a periodic random step function on a 1D grid.

    The function is piecewise constant on the periodic domain, so the left and right
    boundaries are connected.

    Args:
        x: 1D equispaced grid, shape (Nx,)
        key: JAX random key
        num_jumps_min: minimum number of jump points
        num_jumps_max: maximum number of jump points
        value_min, value_max: range of step heights

    Returns:
        u0: shape (Nx,)
    """
    Nx = x.shape[0]
    idx = jnp.arange(Nx)

    key_n, key_bp, key_val, key_shift = jax.random.split(key, 4)

    # number of jumps on the periodic domain
    num_jumps = jax.random.randint(
        key_n, shape=(), minval=num_jumps_min, maxval=num_jumps_max + 1
    )

    # choose jump locations among grid indices
    perm = jax.random.permutation(key_bp, idx)
    jump_idx = jnp.sort(perm[:num_jumps])

    # step values for each interval
    values = jax.random.uniform(
        key_val, shape=(num_jumps,), minval=value_min, maxval=value_max
    )

    # periodic random shift so that boundary x[0] is not special
    shift = jax.random.randint(key_shift, shape=(), minval=0, maxval=Nx)
    idx_shifted = (idx + shift) % Nx

    # count how many jump points are <= each shifted index
    # region label in {0,1,...,num_jumps-1}
    counts = jnp.sum(idx_shifted[:, None] >= jump_idx[None, :], axis=1)
    region = counts % num_jumps

    u0 = values[region]
    return u0


class PeriodicRandomStepFunction1D(eqx.Module):
    num_jumps_min: int = eqx.field(static=True, default=1)
    num_jumps_max: int = eqx.field(static=True, default=5)
    value_min: float = eqx.field(static=True, default=-1.0)
    value_max: float = eqx.field(static=True, default=1.0)

    def sample(self, x: Float[Array, " Nx"], key: PRNGKeyArray) -> Float[Array, " Nx"]:
        return jnp.expand_dims(
            generate_periodic_random_step_function_1d(
                x,
                key,
                num_jumps_min=self.num_jumps_min,
                num_jumps_max=self.num_jumps_max,
                value_min=self.value_min,
                value_max=self.value_max,
            ),
            axis=0,
        )


class MultichannelWaveform(eqx.Module):
    waveforms: tuple

    def __init__(self, waveforms):
        self.waveforms = tuple(waveforms)

    @property
    def channels(self) -> int:
        return len(self.waveforms)

    def sample(self, x: Float[Array, " Nx"], key: PRNGKeyArray) -> Float[Array, "C Nx"]:
        keys = jax.random.split(key, self.channels)
        return jnp.concatenate(
            [wave.sample(x, k) for wave, k in zip(self.waveforms, keys)], axis=0
        )


# ============================================================
# 1. Periodic Voronoi random step function
# ============================================================


def generate_periodic_random_step_function_2d(
    x: Float[Array, " Nx"],
    y: Float[Array, " Ny"],
    key: PRNGKeyArray,
    *,
    channels: int = 1,
    num_regions_min: int = 2,
    num_regions_max: int = 8,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> Float[Array, "C Nx Ny"]:
    """
    Generate a periodic piecewise-constant random field on a 2D domain.

    The domain is partitioned using a Voronoi tessellation on a torus.
    Therefore:

        - discontinuities can have arbitrary orientations,
        - left/right boundaries are periodic,
        - top/bottom boundaries are periodic,
        - all channels share the same discontinuity geometry.

    Each Voronoi region is assigned an independent C-dimensional
    constant state.

    Args:
        x:
            Cell-center x coordinates, shape (Nx,).
        y:
            Cell-center y coordinates, shape (Ny,).
        key:
            JAX random key.
        channels:
            Number of physical/state channels.
        num_regions_min:
            Minimum number of Voronoi regions.
        num_regions_max:
            Maximum number of Voronoi regions.
        value_min:
            Minimum state value.
        value_max:
            Maximum state value.

    Returns:
        u0:
            Initial condition, shape (C, Nx, Ny).
    """
    nx = x.shape[0]
    ny = y.shape[0]

    if nx < 2 or ny < 2:
        raise ValueError("2D grid requires Nx >= 2 and Ny >= 2.")

    if channels < 1:
        raise ValueError("channels must be >= 1.")

    if num_regions_min < 1:
        raise ValueError("num_regions_min must be >= 1.")

    if num_regions_max < num_regions_min:
        raise ValueError("num_regions_max must be >= num_regions_min.")

    key_n, key_seed, key_value = jax.random.split(key, 3)

    # --------------------------------------------------------
    # Recover periodic physical domain from cell centers
    # --------------------------------------------------------

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    lx = dx * nx
    ly = dy * ny

    x_left = x[0] - 0.5 * dx
    y_left = y[0] - 0.5 * dy

    # Grid:
    # X, Y -> (Nx, Ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # --------------------------------------------------------
    # Number of active Voronoi regions
    #
    # num_regions is dynamic, but all arrays have static size
    # num_regions_max so that this remains JIT-friendly.
    # --------------------------------------------------------

    num_regions = jax.random.randint(
        key_n,
        shape=(),
        minval=num_regions_min,
        maxval=num_regions_max + 1,
    )

    active = jnp.arange(num_regions_max) < num_regions

    # --------------------------------------------------------
    # Uniformly sample Voronoi centers on the periodic domain
    # --------------------------------------------------------

    seeds_unit = jax.random.uniform(
        key_seed,
        shape=(num_regions_max, 2),
        minval=0.0,
        maxval=1.0,
    )

    seed_x = x_left + lx * seeds_unit[:, 0]
    seed_y = y_left + ly * seeds_unit[:, 1]

    # --------------------------------------------------------
    # Periodic Euclidean distance
    # --------------------------------------------------------

    dist_x = jnp.abs(X[..., None] - seed_x[None, None, :])
    dist_y = jnp.abs(Y[..., None] - seed_y[None, None, :])

    dist_x = jnp.minimum(dist_x, lx - dist_x)
    dist_y = jnp.minimum(dist_y, ly - dist_y)

    dist_sq = dist_x**2 + dist_y**2

    # Ignore inactive seeds.
    dist_sq = jnp.where(active[None, None, :], dist_sq, jnp.inf)

    # (Nx, Ny)
    region = jnp.argmin(dist_sq, axis=-1)

    # --------------------------------------------------------
    # Random state vector assigned to every region
    #
    # region_values:
    #   (num_regions_max, C)
    #
    # This means all channels share the same region geometry.
    # --------------------------------------------------------

    region_values = jax.random.uniform(
        key_value, shape=(num_regions_max, channels), minval=value_min, maxval=value_max
    )

    # (Nx, Ny, C)
    u0 = region_values[region]
    # -> (C, Nx, Ny)
    u0 = jnp.moveaxis(u0, -1, 0)

    return u0


# ============================================================
# 2. Equinox initial-condition module
# ============================================================


class PeriodicRandomStepFunction2D(eqx.Module):
    channels: int = eqx.field(static=True)
    num_regions_min: int = eqx.field(static=True)
    num_regions_max: int = eqx.field(static=True)
    value_min: tuple[float, ...] = eqx.field(static=True)
    value_max: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_regions_min: int = 1,
        num_regions_max: int = 8,
        value_min: float | Sequence[float] = -1.0,
        value_max: float | Sequence[float] = 1.0,
    ):
        self.channels = channels
        self.num_regions_min = num_regions_min
        self.num_regions_max = num_regions_max
        self.value_min = (
            (value_min,) if isinstance(value_min, float) else tuple(value_min)
        )
        self.value_max = (
            (value_max,) if isinstance(value_max, float) else tuple(value_max)
        )

    def sample(
        self,
        xs: tuple[Float[Array, " #Nx"], ...],
        key: PRNGKeyArray,
    ) -> Float[Array, "C Nx Ny"]:
        return generate_periodic_random_step_function_2d(
            *xs,
            key,
            channels=self.channels,
            num_regions_min=self.num_regions_min,
            num_regions_max=self.num_regions_max,
            value_min=jnp.asarray(self.value_min),
            value_max=jnp.asarray(self.value_max),
        )

    def __call__(
        self,
        grid,
        key: PRNGKeyArray,
    ) -> Float[Array, "C Nx Ny"]:
        x, y = grid
        return self.sample(x, y, key)

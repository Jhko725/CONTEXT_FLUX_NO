"""Error metrics and diagnostics of predicted fields and rollouts.

Every function acts on one snapshot, one rollout or one segment and is left undecorated,
so that the caller chooses where to vmap and jit. The two functions on NumPy arrays,
`first_true_step` and `valid_horizon`, run on the host on arrays that rollouts produced.
"""

import functools
import operator
from math import prod, sqrt
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int


def l2_norm(
    u: Float[Array, " channels *spatial_dims"], dxs: tuple[float, ...] | None = None
) -> Float[Array, ""]:
    """Compute the l2 norm of a field, pooled over channels and cells.

    **Arguments:**

    - `u`: The field.
    - `dxs`: The cell widths along the spatial axes. When given, the squared norm is
        multiplied by the cell volume, so the result approximates the L2 norm of the
        function on the domain. When `None`, the result is the plain Euclidean norm.

    **Returns:**

    The norm, a scalar.
    """
    # Sum over channels for inner product, sum over spatial_dims for integration
    norm_squared = jnp.sum(jnp.abs(u) ** 2)
    if dxs is not None:
        norm_squared = norm_squared * prod(dxs)
    return jnp.sqrt(norm_squared)


def l_infty_norm(u: Float[Array, " channels *spatial_dims"]) -> Float[Array, ""]:
    """Compute the l-infinity norm of a field: the largest absolute value of any cell.

    **Arguments:**

    - `u`: The field.

    **Returns:**

    The norm, a scalar.
    """
    return jnp.max(jnp.abs(u))


def relative_l2_error(
    u_pred: Float[Array, " channels *spatial_dims"],
    u_data: Float[Array, " channels *spatial_dims"],
) -> Float[Array, ""]:
    """Compute the relative l2 error between the predicted and data fields.

    The norm pools the channels, so a channel with a large mean dominates the error.

    **Arguments:**

    - `u_pred`: The predicted field.
    - `u_data`: The reference field.

    **Returns:**

    `l2_norm(u_pred - u_data) / l2_norm(u_data)`, a scalar.
    """
    return l2_norm(u_pred - u_data) / l2_norm(u_data)


def relative_l_infty_error(
    u_pred: Float[Array, " channels *spatial_dims"],
    u_data: Float[Array, " channels *spatial_dims"],
) -> Float[Array, ""]:
    """Compute the relative l-infinity error between the predicted and data fields.

    **Arguments:**

    - `u_pred`: The predicted field.
    - `u_data`: The reference field.

    **Returns:**

    `l_infty_norm(u_pred - u_data) / l_infty_norm(u_data)`, a scalar.
    """
    return l_infty_norm(u_pred - u_data) / l_infty_norm(u_data)


def relative_l2_error_per_channel(
    u_pred: Float[Array, " channels *spatial_dims"],
    u_data: Float[Array, " channels *spatial_dims"],
) -> Float[Array, " channels"]:
    """Compute the relative l2 error of every channel separately.

    `relative_l2_error` pools the channels into one norm, so a channel with a large mean
    dominates it. This function gives each channel its own norm.

    **Arguments:**

    - `u_pred`: The predicted field.
    - `u_data`: The reference field.

    **Returns:**

    One relative error per channel.
    """
    spatial_axes = tuple(range(1, u_data.ndim))
    error_norm = jnp.sqrt(jnp.sum(jnp.abs(u_pred - u_data) ** 2, axis=spatial_axes))
    data_norm = jnp.sqrt(jnp.sum(jnp.abs(u_data) ** 2, axis=spatial_axes))
    return error_norm / data_norm


def spatial_mean_field(
    u: Float[Array, " channels *spatial_dims"],
) -> Float[Array, " channels *spatial_dims"]:
    """Return the field that holds the cell mean of each channel of `u` everywhere.

    A conservation law with periodic boundaries keeps this mean constant, so the field
    is the prediction that has the right mean and no structure.

    **Arguments:**

    - `u`: The field.

    **Returns:**

    A field of the shape of `u`, constant in space within each channel.
    """
    spatial_axes = tuple(range(1, u.ndim))
    return jnp.broadcast_to(jnp.mean(u, axis=spatial_axes, keepdims=True), u.shape)


def error_area_fraction(
    u_pred: Float[Array, " *spatial_dims"],
    u_data: Float[Array, " *spatial_dims"],
    thresholds: Float[Array, " thresholds"],
) -> Float[Array, " thresholds"]:
    """Compute the fraction of cells whose absolute error exceeds a threshold.

    **Arguments:**

    - `u_pred`: The predicted scalar field.
    - `u_data`: The reference scalar field.
    - `thresholds`: The thresholds, as fractions of the range (maximum minus minimum)
        of `u_data`.

    **Returns:**

    One fraction per threshold. A small fraction together with a large l2 error marks
    an error that is confined to few cells.
    """
    data_range = jnp.max(u_data) - jnp.min(u_data)
    abs_error = jnp.abs(u_pred - u_data)
    exceeds = abs_error[..., None] > thresholds * data_range
    return jnp.mean(exceeds, axis=tuple(range(u_data.ndim)))


def total_variation(u: Float[Array, " *spatial_dims"]) -> Float[Array, ""]:
    """Compute the total variation of a scalar field on a periodic grid.

    The total variation is the sum of the absolute differences between neighboring
    cells along every axis, including the pair that wraps around the boundary.

    **Arguments:**

    - `u`: The scalar field.

    **Returns:**

    The total variation, a scalar.
    """
    differences = [jnp.abs(jnp.roll(u, -1, axis=axis) - u) for axis in range(u.ndim)]
    return functools.reduce(operator.add, (jnp.sum(d) for d in differences))


def num_wavenumber_bins(spatial_shape: tuple[int, ...]) -> int:
    """Return the number of bins that `radial_power_spectrum` uses for a grid.

    **Arguments:**

    - `spatial_shape`: The number of cells along each spatial axis.

    **Returns:**

    One more than the rounded magnitude of the largest wavenumber on the grid.
    """
    largest = sqrt(sum((n // 2) ** 2 for n in spatial_shape))
    return round(largest) + 1


def radial_power_spectrum(
    u: Float[Array, " *spatial_dims"],
) -> Float[Array, " wavenumbers"]:
    """Compute the power of a periodic scalar field per wavenumber magnitude.

    Wavenumbers are counted in cycles per domain length. Bin `k` collects the Fourier
    modes whose wavenumber magnitude rounds to `k`, so bin 0 holds the mean. The
    normalization follows Parseval's identity: the bins sum to `jnp.sum(u**2)`.

    **Arguments:**

    - `u`: The scalar field.

    **Returns:**

    The power per bin, `num_wavenumber_bins(u.shape)` bins.
    """
    power = jnp.abs(jnp.fft.fftn(u)) ** 2 / u.size
    frequencies = jnp.meshgrid(
        *[jnp.fft.fftfreq(n, d=1 / n) for n in u.shape], indexing="ij"
    )
    magnitude = jnp.sqrt(sum(f**2 for f in frequencies))
    bins = jnp.round(magnitude).astype(int)
    spectrum = jnp.zeros(num_wavenumber_bins(u.shape), dtype=power.dtype)
    return spectrum.at[bins].add(power)


def error_power_spectrum(
    u_pred: Float[Array, " *spatial_dims"],
    u_data: Float[Array, " *spatial_dims"],
) -> Float[Array, " wavenumbers"]:
    """Compute the power spectrum of the error, relative to the reference power.

    The spectrum of `u_pred - u_data` is divided by the total power of `u_data`, mean
    included. The bins therefore sum to the squared relative l2 error of the field.

    **Arguments:**

    - `u_pred`: The predicted scalar field.
    - `u_data`: The reference scalar field.

    **Returns:**

    The relative error power per bin, as `radial_power_spectrum` bins it.
    """
    return radial_power_spectrum(u_pred - u_data) / jnp.sum(u_data**2)


def conservation_drift(
    u: Float[Array, " channels *spatial_dims"],
    u_initial: Float[Array, " channels *spatial_dims"],
) -> Float[Array, " channels"]:
    """Compute the normalized drift of the conserved quantity of every channel.

    The drift of a channel is `|I(u) - I(u_initial)| / L1(u_initial)`, where `I` is
    the integral of the channel over the domain and `L1` the integral of its absolute
    value. On a uniform grid the cell volume cancels, so both are sums over the cells.
    The L1 norm replaces the integral itself in the denominator because the integral
    is zero for the mean-subtracted initial conditions of the 2D cubic family.

    A conservation law with periodic boundaries keeps `I` constant, so the drift of
    the exact solution is zero.

    **Arguments:**

    - `u`: The state whose drift is measured.
    - `u_initial`: The state that the drift is measured from. For a rollout it is the
        last state of the context window.

    **Returns:**

    One drift per channel.
    """
    spatial_axes = tuple(range(1, u.ndim))
    integral_change = jnp.sum(u, axis=spatial_axes) - jnp.sum(
        u_initial, axis=spatial_axes
    )
    return jnp.abs(integral_change) / jnp.sum(jnp.abs(u_initial), axis=spatial_axes)


class RolloutReferences(eqx.Module):
    """Quantities of one rollout that depend on the reference solution only.

    **Attributes:**

    - `l2_identity`: The relative l2 error of the prediction that holds the last
        context state over the whole rollout.
    - `l2_spatial_mean`: The relative l2 error of the prediction that holds the cell
        mean of each channel of the last context state.
    - `total_variation`: The total variation of the first channel of the reference.
    - `conservation_drift`: The conservation drift of the reference from the last
        context state, per channel. It is the level that the data itself reaches.
    """

    l2_identity: Float[Array, " steps"]
    l2_spatial_mean: Float[Array, " steps"]
    total_variation: Float[Array, " steps"]
    conservation_drift: Float[Array, " steps channels"]


def rollout_references(
    u_context_last: Float[Array, " channels *spatial_dims"],
    u_data: Float[Array, " steps channels *spatial_dims"],
) -> RolloutReferences:
    """Compute the reference quantities of one rollout.

    **Arguments:**

    - `u_context_last`: The last state of the context window.
    - `u_data`: The reference solution at the steps of the rollout.

    **Returns:**

    The `RolloutReferences` of the rollout, one entry per step.
    """
    identity = jnp.broadcast_to(u_context_last, u_data.shape)
    spatial_mean = jnp.broadcast_to(spatial_mean_field(u_context_last), u_data.shape)
    references = RolloutReferences(
        l2_identity=jax.vmap(relative_l2_error)(identity, u_data),
        l2_spatial_mean=jax.vmap(relative_l2_error)(spatial_mean, u_data),
        total_variation=jax.vmap(total_variation)(u_data[:, 0]),
        conservation_drift=jax.vmap(conservation_drift, in_axes=(0, None))(
            u_data, u_context_last
        ),
    )
    # Type-level only: ty 0.0.84 types an instance built through equinox's unannotated
    # `_ModuleMeta.__call__` as `eqx.Module`, not as the class that was called.
    return cast(RolloutReferences, references)


class RolloutDiagnostics(eqx.Module):
    """Per-step diagnostics of one predicted rollout.

    The area fraction and the total variation are those of the first channel, which is
    the first conserved variable of a system.

    **Attributes:**

    - `l2`: The relative l2 error, pooled over channels.
    - `l_inf`: The relative l-infinity error, pooled over channels.
    - `l2_per_channel`: The relative l2 error of every channel.
    - `area_fraction`: The error area fraction at every threshold.
    - `total_variation`: The total variation of the prediction.
    - `nonfinite`: Whether any cell of the prediction is NaN or infinite.
    - `conservation_drift`: The conservation drift of the prediction from the last
        context state, per channel.
    """

    l2: Float[Array, " steps"]
    l_inf: Float[Array, " steps"]
    l2_per_channel: Float[Array, " steps channels"]
    area_fraction: Float[Array, " steps thresholds"]
    total_variation: Float[Array, " steps"]
    nonfinite: Bool[Array, " steps"]
    conservation_drift: Float[Array, " steps channels"]


def rollout_diagnostics(
    u_context_last: Float[Array, " channels *spatial_dims"],
    u_pred: Float[Array, " steps channels *spatial_dims"],
    u_data: Float[Array, " steps channels *spatial_dims"],
    area_thresholds: Float[Array, " thresholds"],
) -> RolloutDiagnostics:
    """Compute the per-step diagnostics of one predicted rollout.

    **Arguments:**

    - `u_context_last`: The last state of the context window.
    - `u_pred`: The predicted states.
    - `u_data`: The reference states at the same steps.
    - `area_thresholds`: The thresholds of the error area fraction, as fractions of the
        range of the reference field.

    **Returns:**

    The `RolloutDiagnostics` of the rollout, one entry per step.
    """
    diagnostics = RolloutDiagnostics(
        l2=jax.vmap(relative_l2_error)(u_pred, u_data),
        l_inf=jax.vmap(relative_l_infty_error)(u_pred, u_data),
        l2_per_channel=jax.vmap(relative_l2_error_per_channel)(u_pred, u_data),
        area_fraction=jax.vmap(error_area_fraction, in_axes=(0, 0, None))(
            u_pred[:, 0], u_data[:, 0], area_thresholds
        ),
        total_variation=jax.vmap(total_variation)(u_pred[:, 0]),
        nonfinite=jnp.any(~jnp.isfinite(u_pred), axis=tuple(range(1, u_pred.ndim))),
        conservation_drift=jax.vmap(conservation_drift, in_axes=(0, None))(
            u_pred, u_context_last
        ),
    )
    return cast(
        RolloutDiagnostics, diagnostics
    )  # type-level only: see rollout_references


def first_true_step(
    flags: Bool[np.ndarray, " *batch steps"],
) -> Int[np.ndarray, " *batch"]:
    """Return the first step at which each row of `flags` is true.

    This function runs on the host, on the arrays that the rollouts have produced.

    **Arguments:**

    - `flags`: One flag per step, along the last axis.

    **Returns:**

    The first true step of each row, counted from 1. A row that is never true receives
    `steps + 1`.
    """
    num_steps = flags.shape[-1]
    return np.where(flags.any(axis=-1), flags.argmax(axis=-1) + 1, num_steps + 1)


def valid_horizon(
    l2: Float[np.ndarray, " *batch steps"],
    l2_identity: Float[np.ndarray, " *batch steps"],
    nonfinite: Bool[np.ndarray, " *batch steps"],
) -> Int[np.ndarray, " *batch"]:
    """Return the first step at which a rollout is no better than the identity baseline.

    The valid horizon is the first step `n` with `l2[n] >= l2_identity[n]`, where the
    identity baseline holds the last context state over the whole rollout. A step with
    a non-finite prediction or a non-finite error also ends it. This function runs on
    the host.

    **Arguments:**

    - `l2`: The relative l2 error of the prediction at every step.
    - `l2_identity`: The relative l2 error of the identity baseline at every step.
    - `nonfinite`: Whether the prediction is non-finite at every step.

    **Returns:**

    The valid horizon of each rollout, counted from 1. A rollout that stays below the
    identity-baseline error receives `steps + 1`, so the caller can tell it apart from
    a rollout that crosses at the last step.
    """
    with np.errstate(invalid="ignore"):
        no_better = l2 >= l2_identity
    return first_true_step(no_better | nonfinite | ~np.isfinite(l2))


class SegmentErrors(eqx.Module):
    """Single-step and rollout errors of one prediction over one segment.

    A segment is a context window followed by the frames that a model predicts from it.

    **Attributes:**

    - `l2_onestep`: The relative l2 error of the first predicted frame.
    - `l_inf_onestep`: The relative l-infinity error of the first predicted frame.
    - `l2_rollout`: The relative l2 error of all predicted frames, with one norm over
        time, channels and space.
    - `l_inf_rollout`: The relative l-infinity error of all predicted frames.
    """

    l2_onestep: Float[Array, ""]
    l_inf_onestep: Float[Array, ""]
    l2_rollout: Float[Array, ""]
    l_inf_rollout: Float[Array, ""]


def segment_errors(
    u_pred: Float[Array, " steps channels *spatial_dims"],
    u_data: Float[Array, " steps channels *spatial_dims"],
) -> SegmentErrors:
    """Compute the single-step and rollout errors of one predicted segment.

    **Arguments:**

    - `u_pred`: The predicted frames.
    - `u_data`: The reference frames at the same steps.

    **Returns:**

    The `SegmentErrors` of the segment.
    """
    errors = SegmentErrors(
        l2_onestep=relative_l2_error(u_pred[0], u_data[0]),
        l_inf_onestep=relative_l_infty_error(u_pred[0], u_data[0]),
        l2_rollout=relative_l2_error(u_pred, u_data),
        l_inf_rollout=relative_l_infty_error(u_pred, u_data),
    )
    return cast(SegmentErrors, errors)  # type-level only: see rollout_references


def largest_jump(u: Float[Array, " x"]) -> Float[Array, ""]:
    """Return the largest absolute difference between neighboring cells of a periodic
    1D field, including the pair that wraps around the boundary.

    The number measures the front strength of a snapshot. A limited solver spreads a
    shock over two to three cells, so the number is a lower bound on the jump across
    the shock.

    **Arguments:**

    - `u`: The periodic 1D field.

    **Returns:**

    The largest jump, a scalar.
    """
    return jnp.max(jnp.abs(jnp.roll(u, -1) - u))


def count_jumps(u: Float[Array, " x"]) -> Int[Array, ""]:
    """Count the neighboring pairs of a periodic 1D field whose values differ.

    The comparison is exact, so the count is meaningful for a piecewise-constant field
    such as a step initial condition, where it is the number of discontinuities.

    **Arguments:**

    - `u`: The periodic 1D field.

    **Returns:**

    The number of differing pairs, the one across the boundary included. A constant
    field gives 0.
    """
    return jnp.sum(jnp.roll(u, -1) != u)

"""Rollouts of a model over a test set, run on the accelerator.

`evaluate_rollouts` rolls a model out from the start of every trajectory and returns the
per-step diagnostics of `metrics.rollout_diagnostics`, and `evaluate_references` returns
the model-independent quantities of the same rollouts. `evaluate_segments` and
`evaluate_segment_baselines` score a model and the two baselines on segments that begin
at chosen frames of every trajectory. Each driver moves the trajectories to the device in
chunks and returns NumPy arrays on the host, which `summary` reduces.

A model reaches this module through the `Operator` protocol, so that nothing here depends
on a model class.
"""

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from .metrics import (
    error_power_spectrum,
    rollout_diagnostics,
    rollout_references,
    RolloutDiagnostics,
    RolloutReferences,
    segment_errors,
    SegmentErrors,
    spatial_mean_field,
)


class Operator(Protocol):
    """A model that predicts future frames autoregressively from a context window."""

    def rollout(
        self,
        u: Float[Array, " time channels *spatial_dims"],
        args: Any,
        num_steps: int,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " {num_steps} channels *spatial_dims"], Any]:
        """Predict `num_steps` frames that follow the context window `u`.

        **Arguments:**

        - `u`: The context window, oldest frame first.
        - `args`: The arguments of one step, such as `(dt, dx)`.
        - `num_steps`: The number of frames to predict.
        - `key`: The random key of a stochastic model, or `None`.
        - `inference`: Whether the model runs in inference mode, or `None` for its own
            setting.

        **Returns:**

        The predicted frames and the auxiliary output of every step.
        """
        ...


PositiveQuantities = Callable[
    [Float[Array, " channels *spatial_dims"]],
    Float[Array, " quantities *spatial_dims"],
]
"""A function that returns the quantities of one frame that must stay positive.

It reads the frame in the channel order of the trajectories it is evaluated on, so a
caller composes `AbstractPDEFamily.positive_quantities` with the store's channel order.
"""


@dataclasses.dataclass(frozen=True)
class RolloutEvaluation:
    """The result of rolling one model out over a set of trajectories.

    The result lives on the host and holds NumPy arrays, the fields of `diagnostics`
    included.

    **Attributes:**

    - `diagnostics`: The per-step diagnostics of every trajectory.
    - `minima`: The per-step minimum over the cells of each quantity that must stay
        positive, or `None` when no `positive_quantities` was given.
    - `error_spectrum`: The error power spectrum of the first channel at the requested
        steps, averaged over the trajectories whose prediction is finite at that step.
        `None` when no spectrum was requested.
    - `num_finite_spectra`: The number of trajectories in that average, per step.
    """

    diagnostics: RolloutDiagnostics
    minima: Float[np.ndarray, " trajectories steps quantities"] | None
    error_spectrum: Float[np.ndarray, " spectrum_steps wavenumbers"] | None
    num_finite_spectra: Float[np.ndarray, " spectrum_steps"] | None


def _rollout_one(
    model: Operator,
    u: Float[Array, " frames channels *spatial_dims"],
    model_args: Any,
    context_length: int,
    area_thresholds: Float[Array, " thresholds"],
    spectrum_indices: tuple[int, ...],
    positive_quantities: PositiveQuantities | None,
) -> tuple[
    RolloutDiagnostics,
    Float[Array, " steps quantities"] | None,
    Float[Array, " spectrum_steps wavenumbers"] | None,
]:
    """Roll `model` out from the first `context_length` frames of `u` and compare the
    prediction with the remaining frames."""
    context, u_data = u[:context_length], u[context_length:]
    u_pred = model.rollout(context, model_args, num_steps=len(u_data))[0]

    diagnostics = rollout_diagnostics(context[-1], u_pred, u_data, area_thresholds)
    minima = None
    if positive_quantities is not None:
        quantities = jax.vmap(positive_quantities)(u_pred)
        minima = jnp.min(quantities, axis=tuple(range(2, quantities.ndim)))
    spectrum = None
    if spectrum_indices:
        indices = jnp.asarray(spectrum_indices)
        spectrum = jax.vmap(error_power_spectrum)(
            u_pred[indices, 0], u_data[indices, 0]
        )
    return diagnostics, minima, spectrum


def evaluate_rollouts(
    model: Operator,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    model_args: Any,
    *,
    context_length: int,
    area_thresholds: tuple[float, ...],
    spectrum_steps: tuple[int, ...] = (),
    positive_quantities: PositiveQuantities | None = None,
    chunk_size: int = 1000,
    batch_size: int = 1000,
) -> RolloutEvaluation:
    """Roll a model out over every trajectory and compute the rollout diagnostics.

    Each trajectory gives one rollout. Its first `context_length` frames are the
    context, and the model predicts all remaining frames autoregressively.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `model_args`: The `args` of `model.rollout`, such as `(dt, dx)`.
    - `context_length`: The number of context frames.
    - `area_thresholds`: The thresholds of the error area fraction.
    - `spectrum_steps`: The rollout steps, counted from 1, at which the error spectrum
        of the first channel is computed. Empty for no spectrum.
    - `positive_quantities`: The quantities of a frame that must stay positive, in the
        channel order of `trajectories`, such as the bound
        `AbstractPDEFamily.positive_quantities` of the test set's family composed with
        the store's channel order. `None` for none.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of rollouts that run in parallel on the device.

    **Returns:**

    A `RolloutEvaluation` whose arrays are NumPy arrays on the host.

    **Raises:**

    `ValueError` if the trajectories leave no step after the context, or if a step of
    `spectrum_steps` lies outside the rollout.
    """
    num_trajectories, num_frames = trajectories.shape[:2]
    num_steps = num_frames - context_length
    if num_steps < 1:
        raise ValueError(
            f"Trajectories of {num_frames} frames leave no step after a context of "
            f"{context_length} frames."
        )
    if any(step < 1 or step > num_steps for step in spectrum_steps):
        raise ValueError(
            f"spectrum_steps={spectrum_steps} must lie within 1 and {num_steps}."
        )
    spectrum_indices = tuple(step - 1 for step in spectrum_steps)
    thresholds = jnp.asarray(area_thresholds)

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk):
        return jax.lax.map(
            lambda u: _rollout_one(
                model,
                u,
                model_args,
                context_length,
                thresholds,
                spectrum_indices,
                positive_quantities,
            ),
            u_chunk,
            batch_size=batch_size,
        )

    diagnostics_chunks, minima_chunks = [], []
    spectrum_sum, spectrum_count = 0.0, 0.0
    for start in range(0, num_trajectories, chunk_size):
        diagnostics, minima, spectrum = evaluate_chunk(
            model, jnp.asarray(trajectories[start : start + chunk_size])
        )
        diagnostics_chunks.append(jax.tree.map(np.asarray, diagnostics))
        if minima is not None:
            minima_chunks.append(np.asarray(minima))
        if spectrum is not None:
            spectrum = np.asarray(spectrum)
            is_finite = np.all(np.isfinite(spectrum), axis=-1, keepdims=True)
            spectrum_sum = spectrum_sum + np.where(is_finite, spectrum, 0.0).sum(axis=0)
            spectrum_count = spectrum_count + is_finite[..., 0].sum(axis=0)

    has_spectrum = bool(spectrum_indices)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_spectrum = spectrum_sum / spectrum_count[:, None] if has_spectrum else None
    return RolloutEvaluation(
        diagnostics=jax.tree.map(lambda *xs: np.concatenate(xs), *diagnostics_chunks),
        minima=np.concatenate(minima_chunks) if minima_chunks else None,
        error_spectrum=mean_spectrum,
        num_finite_spectra=spectrum_count if has_spectrum else None,
    )


def _map_over_chunks(
    evaluate_chunk: Callable[
        [Float[Array, " chunk frames channels *spatial_dims"]], PyTree[Array]
    ],
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    chunk_size: int,
) -> PyTree[np.ndarray]:
    """Apply a jitted function to chunks of trajectories and join the results on the
    host."""
    chunks = [
        jax.tree.map(
            np.asarray,
            evaluate_chunk(jnp.asarray(trajectories[start : start + chunk_size])),
        )
        for start in range(0, trajectories.shape[0], chunk_size)
    ]
    return jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)


def evaluate_references(
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    *,
    context_length: int,
    chunk_size: int = 1000,
) -> RolloutReferences:
    """Compute the model-independent reference quantities of every rollout.

    The rollouts are those of `evaluate_rollouts`: the context is the first
    `context_length` frames, and the remaining frames are the reference.

    **Arguments:**

    - `trajectories`: The test trajectories. The array stays on the host.
    - `context_length`: The number of context frames.
    - `chunk_size`: The number of trajectories that are moved to the device at once.

    **Returns:**

    A `RolloutReferences` whose fields are NumPy arrays with a leading trajectory axis.
    """

    @jax.jit
    def evaluate_chunk(u_chunk):
        return jax.vmap(rollout_references)(
            u_chunk[:, context_length - 1], u_chunk[:, context_length:]
        )

    return _map_over_chunks(evaluate_chunk, trajectories, chunk_size)


class SegmentBaselines(eqx.Module):
    """Errors of the two baselines on the segments of a test set.

    **Attributes:**

    - `identity`: The errors of the identity baseline, which holds the last context
        frame over the whole segment.
    - `spatial_mean`: The errors of the spatial-mean baseline, which holds the cell
        mean of each channel of the last context frame.
    """

    identity: SegmentErrors
    spatial_mean: SegmentErrors


def _check_window_starts(
    window_starts: tuple[int, ...], num_frames: int, segment_length: int
) -> None:
    """Raise a `ValueError` unless every segment fits within the trajectory."""
    if not window_starts:
        raise ValueError("window_starts is empty.")
    last_start = num_frames - segment_length
    if min(window_starts) < 0 or max(window_starts) > last_start:
        raise ValueError(
            f"window_starts must lie within 0 and {last_start} for {num_frames} "
            f"frames and segments of {segment_length} frames, got {window_starts}."
        )


def _segments_of(
    u: Float[Array, " frames channels *spatial_dims"],
    window_starts: Int[Array, " windows"],
    segment_length: int,
) -> Float[Array, " windows {segment_length} channels *spatial_dims"]:
    """Return the segments of one trajectory that begin at `window_starts`."""
    return u[window_starts[:, None] + jnp.arange(segment_length)]


def evaluate_segments(
    model: Operator,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    window_starts: tuple[int, ...],
    model_args: Any,
    *,
    context_length: int,
    rollout_steps: int,
    chunk_size: int = 1000,
    batch_size: int = 16,
) -> SegmentErrors:
    """Compute the errors of a model on every segment of every trajectory.

    A segment begins at one of `window_starts`. Its first `context_length` frames are
    the context, and the model predicts the next `rollout_steps` frames
    autoregressively. The errors of the segments are returned one by one, not
    averaged.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `window_starts`: The frame indices at which segments begin.
    - `model_args`: The `args` of `model.rollout`, such as `(dt, dx)`.
    - `context_length`: The number of context frames.
    - `rollout_steps`: The number of predicted frames.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of trajectories whose segments run in parallel on the
        device. The number of parallel rollouts is `batch_size * len(window_starts)`.

    **Returns:**

    A `SegmentErrors` whose fields are NumPy arrays of shape `(trajectories, windows)`.

    **Raises:**

    `ValueError` if `window_starts` is empty or a segment ends past the last frame.
    """
    segment_length = context_length + rollout_steps
    _check_window_starts(window_starts, trajectories.shape[1], segment_length)
    starts = jnp.asarray(window_starts)

    def one_segment(model, segment):
        context, u_data = segment[:context_length], segment[context_length:]
        u_pred = model.rollout(context, model_args, num_steps=rollout_steps)[0]
        return segment_errors(u_pred, u_data)

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk):
        return jax.lax.map(
            lambda u: jax.vmap(lambda segment: one_segment(model, segment))(
                _segments_of(u, starts, segment_length)
            ),
            u_chunk,
            batch_size=batch_size,
        )

    return _map_over_chunks(
        lambda u_chunk: evaluate_chunk(model, u_chunk), trajectories, chunk_size
    )


def evaluate_segment_baselines(
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    window_starts: tuple[int, ...],
    *,
    context_length: int,
    rollout_steps: int,
    chunk_size: int = 1000,
) -> SegmentBaselines:
    """Compute the errors of the two baselines on the segments of `evaluate_segments`.

    **Arguments:**

    - `trajectories`: The test trajectories. The array stays on the host.
    - `window_starts`: The frame indices at which segments begin.
    - `context_length`: The number of context frames.
    - `rollout_steps`: The number of predicted frames.
    - `chunk_size`: The number of trajectories that are moved to the device at once.

    **Returns:**

    A `SegmentBaselines` whose arrays have the shape `(trajectories, windows)`.

    **Raises:**

    `ValueError` if `window_starts` is empty or a segment ends past the last frame.
    """
    segment_length = context_length + rollout_steps
    _check_window_starts(window_starts, trajectories.shape[1], segment_length)
    starts = jnp.asarray(window_starts)

    def one_segment(segment):
        last, u_data = segment[context_length - 1], segment[context_length:]
        return SegmentBaselines(
            identity=segment_errors(jnp.broadcast_to(last, u_data.shape), u_data),
            spatial_mean=segment_errors(
                jnp.broadcast_to(spatial_mean_field(last), u_data.shape), u_data
            ),
        )

    @jax.jit
    def evaluate_chunk(u_chunk):
        return jax.vmap(
            lambda u: jax.vmap(one_segment)(_segments_of(u, starts, segment_length))
        )(u_chunk)

    return _map_over_chunks(evaluate_chunk, trajectories, chunk_size)

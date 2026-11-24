"""Checks of the rollout and segment drivers in `context_flux_no.evaluation.rollout`.

The drivers run the stub models of `tests/helpers.py` on traveling waves, whose errors
are known in closed form.
"""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from context_flux_no.evaluation.rollout import (
    evaluate_references,
    evaluate_rollouts,
    evaluate_segment_baselines,
    evaluate_segments,
)
from jaxtyping import Array, Float

from tests.helpers import HoldLastFrame, ROLLOUT_ATOL, ShiftByOneCell


NUM_CELLS = 64
NUM_FRAMES = 12
CONTEXT_LENGTH = 3
ROLLOUT_STEPS = 4
# The wave moves by this fraction of its period per frame.
SHIFT_PER_FRAME = 1 / 32
MODEL_ARGS = (0.005, 1 / NUM_CELLS)
# Type-level only: ty 0.0.84 types an instance built through equinox's unannotated
# `_ModuleMeta.__call__` as `eqx.Module`, not as the class that was called.
HOLD_LAST_FRAME = cast(HoldLastFrame, HoldLastFrame())
SHIFT_BY_ONE_CELL = cast(ShiftByOneCell, ShiftByOneCell())


def _traveling_sines(shift_per_frame: float) -> Float[np.ndarray, " 2 frames 1 x"]:
    """Two trajectories of a sine wave that moves by `shift_per_frame` of its period per
    frame. The second has twice the amplitude, which no relative error depends on."""
    x = np.arange(NUM_CELLS) / NUM_CELLS
    frames = np.arange(NUM_FRAMES)[:, None] * shift_per_frame
    wave = np.sin(2 * np.pi * (x - frames))[:, None, :]
    return np.stack((wave, 2 * wave)).astype(np.float32)


def _identity_step_errors(num_steps: int) -> Float[np.ndarray, " {num_steps}"]:
    """Relative l2 error of the identity baseline at each step on the sine wave.

    A sine shifted by the phase phi differs from itself by 2 |sin(phi / 2)| in relative
    l2, and step n of a rollout has the phase 2 pi n SHIFT_PER_FRAME.
    """
    steps = np.arange(1, num_steps + 1)
    return 2 * np.abs(np.sin(np.pi * steps * SHIFT_PER_FRAME))


def _identity_errors() -> tuple[float, float]:
    """Single-step and rollout l2 error of the identity baseline on a segment.

    The rollout error is the root mean square of the step errors, since every frame has
    the same norm.
    """
    per_step = _identity_step_errors(ROLLOUT_STEPS)
    return float(per_step[0]), float(np.sqrt(np.mean(np.square(per_step))))


def test_baselines_on_a_traveling_sine():
    """The identity baseline has the closed-form errors, and the spatial-mean baseline
    predicts zero, the cell mean of a sine."""
    window_starts = (0, 2, 5)
    baselines = evaluate_segment_baselines(
        _traveling_sines(SHIFT_PER_FRAME),
        window_starts,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
    )
    onestep, rollout = _identity_errors()
    assert baselines.identity.l2_onestep.shape == (2, len(window_starts))
    assert np.allclose(baselines.identity.l2_onestep, onestep, atol=ROLLOUT_ATOL)
    assert np.allclose(baselines.identity.l2_rollout, rollout, atol=ROLLOUT_ATOL)
    assert np.allclose(baselines.spatial_mean.l2_onestep, 1.0, atol=ROLLOUT_ATOL)
    assert np.allclose(baselines.spatial_mean.l2_rollout, 1.0, atol=ROLLOUT_ATOL)
    assert np.allclose(baselines.spatial_mean.l_inf_rollout, 1.0, atol=ROLLOUT_ATOL)


def test_a_model_that_holds_the_last_frame_has_the_identity_errors():
    """On every segment, the stub that holds the last frame has the errors of the
    identity baseline."""
    window_starts = (0, 1, 5)
    errors = evaluate_segments(
        HOLD_LAST_FRAME,
        _traveling_sines(SHIFT_PER_FRAME),
        window_starts,
        MODEL_ARGS,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
        batch_size=1,
    )
    onestep, rollout = _identity_errors()
    assert errors.l2_rollout.shape == (2, len(window_starts))
    assert np.allclose(errors.l2_onestep, onestep, atol=ROLLOUT_ATOL)
    assert np.allclose(errors.l2_rollout, rollout, atol=ROLLOUT_ATOL)


def test_each_window_is_scored_against_its_own_frames():
    """Each window's error is computed from the frames that begin at its own start.

    Frame n holds the constant n + 1, so the identity baseline of a segment that starts
    at frame s predicts s + CONTEXT_LENGTH and its single-step error is
    1 / (s + CONTEXT_LENGTH + 1). The error identifies the window.
    """
    frames = np.arange(1, NUM_FRAMES + 1, dtype=np.float32)
    trajectories = np.broadcast_to(frames[:, None, None], (1, NUM_FRAMES, 1, 8)).copy()
    window_starts = (5, 0, 3)
    baselines = evaluate_segment_baselines(
        trajectories,
        window_starts,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
    )
    expected = [1 / (start + CONTEXT_LENGTH + 1) for start in window_starts]
    assert np.allclose(baselines.identity.l2_onestep[0], expected, atol=ROLLOUT_ATOL)


def test_a_window_past_the_last_frame_is_rejected():
    """A window whose segment ends past the last frame raises."""
    last_start = NUM_FRAMES - CONTEXT_LENGTH - ROLLOUT_STEPS
    with pytest.raises(ValueError, match="window_starts"):
        evaluate_segment_baselines(
            _traveling_sines(SHIFT_PER_FRAME),
            (0, last_start + 1),
            context_length=CONTEXT_LENGTH,
            rollout_steps=ROLLOUT_STEPS,
        )


def test_a_model_that_moves_with_the_wave_has_zero_error():
    """The stub that shifts by one cell per step reproduces a wave that moves by one
    cell per frame, so every error and the error spectrum are zero."""
    evaluation = evaluate_rollouts(
        SHIFT_BY_ONE_CELL,
        _traveling_sines(1 / NUM_CELLS),
        MODEL_ARGS,
        context_length=CONTEXT_LENGTH,
        area_thresholds=(0.1,),
        spectrum_steps=(1, NUM_FRAMES - CONTEXT_LENGTH),
    )
    num_steps = NUM_FRAMES - CONTEXT_LENGTH
    assert evaluation.diagnostics.l2.shape == (2, num_steps)
    assert np.allclose(evaluation.diagnostics.l2, 0.0, atol=ROLLOUT_ATOL)
    assert np.allclose(evaluation.diagnostics.area_fraction, 0.0)
    assert not evaluation.diagnostics.nonfinite.any()
    assert evaluation.minima is None
    assert evaluation.error_spectrum is not None
    assert evaluation.num_finite_spectra is not None
    assert np.allclose(evaluation.error_spectrum, 0.0, atol=ROLLOUT_ATOL)
    assert evaluation.num_finite_spectra.tolist() == [2, 2]


def test_a_rollout_that_holds_the_last_frame_has_the_identity_errors_in_chunks():
    """With one trajectory per chunk, the stub that holds the last frame has the
    closed-form identity errors at every step, and so has the reference baseline."""
    trajectories = _traveling_sines(SHIFT_PER_FRAME)
    num_steps = NUM_FRAMES - CONTEXT_LENGTH
    expected = _identity_step_errors(num_steps)
    evaluation = evaluate_rollouts(
        HOLD_LAST_FRAME,
        trajectories,
        MODEL_ARGS,
        context_length=CONTEXT_LENGTH,
        area_thresholds=(0.1, 0.5),
        chunk_size=1,
    )
    references = evaluate_references(
        trajectories, context_length=CONTEXT_LENGTH, chunk_size=1
    )
    assert evaluation.diagnostics.area_fraction.shape == (2, num_steps, 2)
    assert evaluation.error_spectrum is None
    assert evaluation.num_finite_spectra is None
    for l2 in (evaluation.diagnostics.l2, references.l2_identity):
        assert l2.shape == (2, num_steps)
        assert np.allclose(l2, expected, atol=ROLLOUT_ATOL)


def _shifted_and_negated(
    u: Float[Array, " 1 x"],
) -> Float[Array, " 2 x"]:
    """Two quantities of a frame, in this order: the field plus 1.5, and its negation."""
    return jnp.stack((u[0] + 1.5, -u[0]))


def test_minima_of_the_positive_quantities_follow_their_order():
    """The minima are those of each positive quantity over the cells, in its order.

    The two sines have the amplitudes 1 and 2, and a wave that moves by whole cells
    keeps a cell at its trough, so the minima are 1.5 - A and -A at every step.
    """
    evaluation = evaluate_rollouts(
        SHIFT_BY_ONE_CELL,
        _traveling_sines(1 / NUM_CELLS),
        MODEL_ARGS,
        context_length=CONTEXT_LENGTH,
        area_thresholds=(0.1,),
        positive_quantities=_shifted_and_negated,
    )
    num_steps = NUM_FRAMES - CONTEXT_LENGTH
    expected = np.broadcast_to(
        np.array([[0.5, -1.0], [-0.5, -2.0]])[:, None, :], (2, num_steps, 2)
    )
    assert evaluation.minima is not None
    assert np.allclose(evaluation.minima, expected, atol=ROLLOUT_ATOL)


def test_a_spectrum_step_outside_the_rollout_is_rejected():
    """A spectrum step of 0 or past the last step raises."""
    num_steps = NUM_FRAMES - CONTEXT_LENGTH
    for steps in ((0,), (num_steps + 1,)):
        with pytest.raises(ValueError, match="spectrum_steps"):
            evaluate_rollouts(
                HOLD_LAST_FRAME,
                _traveling_sines(SHIFT_PER_FRAME),
                MODEL_ARGS,
                context_length=CONTEXT_LENGTH,
                area_thresholds=(0.1,),
                spectrum_steps=steps,
            )


def test_a_context_without_a_step_after_it_is_rejected():
    """A context that covers every frame leaves no step to predict and raises."""
    with pytest.raises(ValueError, match="leave no step"):
        evaluate_rollouts(
            HOLD_LAST_FRAME,
            _traveling_sines(SHIFT_PER_FRAME),
            MODEL_ARGS,
            context_length=NUM_FRAMES,
            area_thresholds=(0.1,),
        )

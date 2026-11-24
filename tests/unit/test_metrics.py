"""Analytic checks of the metrics in `context_flux_no.evaluation.metrics`.

Each test compares one definition with a value that is known in closed form.
"""

import jax
import jax.numpy as jnp
import numpy as np
from context_flux_no.evaluation.metrics import (
    conservation_drift,
    count_jumps,
    error_area_fraction,
    error_power_spectrum,
    first_true_step,
    largest_jump,
    radial_power_spectrum,
    relative_l2_error,
    relative_l2_error_per_channel,
    rollout_diagnostics,
    rollout_references,
    segment_errors,
    total_variation,
    valid_horizon,
)
from jaxtyping import Array, Float


# Agreement of a closed form with its float32 evaluation, for sums over at most
# 64 x 64 cells of order-one values.
ATOL = 1e-4
RTOL = 1e-4


def _grid(n: int) -> Float[Array, " n"]:
    return jnp.linspace(0.0, 1.0, n, endpoint=False)


def test_total_variation_of_a_sine():
    """The total variation of a sine is four times its amplitude per period."""

    # A sine of amplitude A with m periods rises and falls by 2A per half period. The
    # grid of 64 cells samples the extrema of m = 2 exactly.
    amplitude, periods = 1.5, 2
    u = amplitude * jnp.sin(2 * jnp.pi * periods * _grid(64))
    assert jnp.allclose(total_variation(u), 4 * amplitude * periods, atol=ATOL)


def test_total_variation_sums_over_axes():
    """The total variation of a 2D field sums the variation along every axis."""

    # u(x, y) = f(x) varies along x only, so every one of the 16 rows along y
    # contributes the 1D total variation once.
    f = jnp.sin(2 * jnp.pi * _grid(64))
    u = jnp.broadcast_to(f[:, None], (64, 16))
    assert jnp.allclose(total_variation(u), 16 * 4.0, atol=ATOL)


def test_error_area_fraction_counts_cells_above_threshold():
    """The area fraction counts the cells whose error exceeds each threshold."""

    # The reference has range 2. The error is 0.3 on 5 of 100 cells, which exceeds
    # 0.1 * 2 and not 0.2 * 2.
    u_data = jnp.linspace(-1.0, 1.0, 100)
    u_pred = u_data.at[10:15].add(0.3)
    fractions = error_area_fraction(u_pred, u_data, jnp.array([0.05, 0.1, 0.2]))
    assert jnp.allclose(fractions, jnp.array([0.05, 0.05, 0.0]), atol=ATOL)


def test_radial_power_spectrum_of_single_modes():
    """Each Fourier mode lies in the bin of its wavenumber, with its mean square."""

    # cos(2 pi (3x + 4y)) has wavenumber magnitude 5 and mean square 1/2. The constant
    # 2 lies in bin 0 with mean square 4.
    x, y = jnp.meshgrid(_grid(32), _grid(32), indexing="ij")
    u = 2.0 + jnp.cos(2 * jnp.pi * (3 * x + 4 * y))
    spectrum = radial_power_spectrum(u)
    expected = jnp.zeros_like(spectrum).at[0].set(4.0 * u.size).at[5].set(0.5 * u.size)
    assert jnp.allclose(spectrum, expected, rtol=RTOL, atol=ATOL * u.size)


def test_error_power_spectrum_sums_to_squared_relative_error():
    """The error spectrum sums to the squared relative l2 error (Parseval)."""

    key_pred, key_data = jax.random.split(jax.random.key(0))
    u_data = 1.0 + jax.random.normal(key_data, (32, 48))
    u_pred = u_data + 0.1 * jax.random.normal(key_pred, (32, 48))
    spectrum = error_power_spectrum(u_pred, u_data)
    expected = relative_l2_error(u_pred[None], u_data[None]) ** 2
    assert jnp.allclose(jnp.sum(spectrum), expected, rtol=RTOL)


def test_relative_l2_error_per_channel():
    """Each channel's error is normalized by that channel's own norm."""

    u_data = jnp.stack((jnp.ones(50), 10.0 * jnp.ones(50)))
    u_pred = u_data + jnp.array([[0.1], [0.0]])
    errors = relative_l2_error_per_channel(u_pred, u_data)
    assert jnp.allclose(errors, jnp.array([0.1, 0.0]), atol=ATOL)


def test_rollout_references_of_a_translating_wave():
    """The baseline errors of a translating sine match their closed forms."""

    # u(x, t) = 1 + sin(2 pi (x - t)) has mean 1 and fluctuation power 1/2, so the
    # spatial-mean error is sqrt((1/2) / (3/2)). After a shift of half a period the
    # wave is 1 - sin, so the identity-baseline error is sqrt(2 / (3/2)).
    x = _grid(64)
    shifts = jnp.array([0.0, 0.25, 0.5])
    u = (1.0 + jnp.sin(2 * jnp.pi * (x - shifts[:, None])))[:, None]
    references = rollout_references(u[0], u[1:])
    assert jnp.allclose(references.l2_spatial_mean, jnp.sqrt(1 / 3), atol=ATOL)
    assert jnp.allclose(references.l2_identity[1], jnp.sqrt(4 / 3), atol=ATOL)
    assert jnp.allclose(references.total_variation, 4.0, atol=ATOL)


def test_rollout_diagnostics_flags_nonfinite_steps():
    """A step with a NaN cell is flagged, and the other steps are not."""

    u_data = jnp.ones((3, 2, 16)) + jnp.sin(2 * jnp.pi * _grid(16))
    u_pred = u_data.at[1, 1, 3].set(jnp.nan)
    diagnostics = rollout_diagnostics(u_data[0], u_pred, u_data, jnp.array([0.1]))
    assert diagnostics.nonfinite.tolist() == [False, True, False]
    assert jnp.allclose(diagnostics.l2[0], 0.0, atol=ATOL)
    assert diagnostics.area_fraction.shape == (3, 1)
    assert diagnostics.l2_per_channel.shape == (3, 2)


def test_first_true_step_counts_from_one():
    """Steps count from 1, and a row that is never true receives `steps + 1`."""

    flags = np.array([[False, True, True], [False, False, False], [True, False, False]])
    assert first_true_step(flags).tolist() == [2, 4, 1]


def test_valid_horizon():
    """The horizon ends at the first step that reaches the baseline or is non-finite."""

    # One identity baseline for all four rollouts, broadcast to the shape of `l2`.
    l2 = np.array(
        [
            [0.05, 0.1, 0.3, 0.2],  # reaches the baseline at step 3
            [0.2, 0.1, 0.1, 0.1],  # above the baseline at step 1 only
            [0.0, 0.1, 0.2, 0.3],  # never reaches it
            [0.0, np.nan, np.nan, np.nan],  # non-finite from step 2
        ]
    )
    l2_identity = np.broadcast_to(np.array([0.1, 0.2, 0.3, 0.4]), l2.shape)
    nonfinite = np.isnan(l2)
    assert valid_horizon(l2, l2_identity, nonfinite).tolist() == [3, 1, 5, 2]


def test_segment_errors_of_a_constant_offset():
    """The errors of a constant offset per frame match their closed forms."""

    # The reference is 1 in the first frame and 2 in the second, and the prediction is
    # off by 0.5 in the first frame and by 1 in the second. The single-step errors are
    # 0.5 / 1. The rollout l2 error is sqrt(0.25 + 1) / sqrt(1 + 4) = 0.5, and the
    # rollout l-infinity error is 1 / 2.
    u_data = jnp.stack((jnp.ones((2, 8)), 2 * jnp.ones((2, 8))))
    u_pred = u_data + jnp.asarray([0.5, 1.0])[:, None, None]
    errors = segment_errors(u_pred, u_data)
    assert jnp.allclose(errors.l2_onestep, 0.5, atol=ATOL)
    assert jnp.allclose(errors.l_inf_onestep, 0.5, atol=ATOL)
    assert jnp.allclose(errors.l2_rollout, 0.5, atol=ATOL)
    assert jnp.allclose(errors.l_inf_rollout, 0.5, atol=ATOL)


def test_jumps_of_a_step_function():
    """A periodic step function has one jump per plateau, the largest exact."""

    # Three plateaus on a periodic grid have three discontinuities, the one across
    # the boundary included, and the largest is the one from 4 down to 0.5.
    u = jnp.asarray([0.5, 0.5, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
    assert int(count_jumps(u)) == 3
    assert jnp.allclose(largest_jump(u), 3.5, atol=ATOL)
    assert int(count_jumps(jnp.full(8, 1.5))) == 0


def test_conservation_drift_of_a_shift_and_of_an_offset():
    """A periodic shift has no drift; an offset has its closed-form drift."""

    # Channel 0 is a sine with mean zero. The sum of |sin(2 pi k / 64)| over 64 cells
    # is 2 / tan(pi / 64), by the closed form of a sum of sines in arithmetic
    # progression. Channel 1 is the constant 2, with an L1 norm of 64 * 2.
    x = _grid(64)
    u_initial = jnp.stack((jnp.sin(2 * jnp.pi * x), jnp.full(64, 2.0)))
    # A periodic shift keeps both integrals.
    shifted = jnp.roll(u_initial, 5, axis=1)
    assert jnp.allclose(conservation_drift(shifted, u_initial), 0.0, atol=ATOL)
    # An offset of 0.1 changes each integral by 64 * 0.1.
    drift = conservation_drift(u_initial + 0.1, u_initial)
    expected = jnp.array([6.4 * jnp.tan(jnp.pi / 64) / 2, 0.05])
    assert jnp.allclose(drift, expected, rtol=RTOL)


def test_rollout_fields_measure_the_drift_from_the_last_context_state():
    """Both rollout records measure the drift from the last context state."""

    # The prediction gains 0.01 per cell and step on a state of constant 1, so its
    # drift at step n is 0.01 * n. The reference is a shifted copy and has none.
    u_context_last = jnp.ones((1, 32))
    u_data = jnp.ones((4, 1, 32))
    u_pred = u_data + 0.01 * jnp.arange(1, 5)[:, None, None]
    diagnostics = rollout_diagnostics(u_context_last, u_pred, u_data, jnp.array([0.1]))
    references = rollout_references(u_context_last, u_data)
    expected = 0.01 * jnp.arange(1, 5)[:, None]
    assert jnp.allclose(diagnostics.conservation_drift, expected, rtol=RTOL)
    assert jnp.allclose(references.conservation_drift, 0.0, atol=ATOL)

"""Checks of `evaluation.cost`: the FLOP count and the measurement of a matrix product,
against the closed form `2 m k n` of its multiply-adds and the size of its output."""

import jax.numpy as jnp
import numpy as np
from context_flux_no.evaluation import count_flops, measure
from jaxtyping import Array, Float


def _matmul(a: Float[Array, "m k"], b: Float[Array, "k n"]) -> Float[Array, "m n"]:
    return a @ b


def test_count_flops_of_a_matrix_product():
    a, b = jnp.ones((6, 5)), jnp.ones((5, 4))
    assert count_flops(_matmul, a, b) == 2 * 6 * 5 * 4


def test_measure_times_every_repeat():
    a, b = jnp.ones((6, 5)), jnp.ones((5, 4))
    measurement = measure(_matmul, a, b, repeats=3, warmup=1)
    assert measurement.seconds.shape == (3,)
    assert np.all(np.isfinite(measurement.seconds) & (measurement.seconds > 0))
    assert measurement.flops == 2 * 6 * 5 * 4
    # Six by four float32 entries.
    assert measurement.output_bytes == 6 * 4 * 4

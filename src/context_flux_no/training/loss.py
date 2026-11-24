"""Training losses: the one-step mean squared error and two variants of it, one with
noise on the input and one that pushes the input forward through the model first."""

from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from context_flux_no.custom_types import FloatScalar


class AbstractTrainLoss[M: eqx.Module, B: PyTree[Array]](Protocol):
    """What `Trainer` calls a loss: `loss(model, batch, args, key)` returns the scalar
    loss and a dictionary of further metrics."""

    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        """Return the scalar loss of `model` on `batch`, where `args` is what the model
        needs besides the state, and a dictionary of further metrics."""
        pass


class OneStepLoss[M: eqx.Module, B: PyTree[Array]](eqx.Module):
    """The mean squared error of the model's prediction of the last frame of each window
    from the frames before it."""

    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        """Return the loss of `model` on `batch` and an empty dictionary of further
        metrics."""
        u: Float[Array, "batch time dim ..."] = batch
        u0, u1 = u[:, :-1], u[:, -1]

        keys = jax.random.split(key, u0.shape[0])
        u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
            lambda u_, key_: model(u_, args, key=key_)
        )(u0, keys)[0]
        return jnp.mean((u1 - u1_pred) ** 2), dict()


class DenoisingOneStepLoss[M: eqx.Module, B: PyTree[Array]](eqx.Module):
    """`OneStepLoss` with Gaussian noise added to the input frames. The noise of each
    frame and channel has standard deviation `noise_scale` times the frame's norm
    over the spatial axes."""

    noise_scale: float = eqx.field(static=True, default=5e-5)

    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        """Return the loss of `model` on `batch` and an empty dictionary of further
        metrics."""
        u: Float[Array, "batch time dim ..."] = batch
        u0, u1 = u[:, :-1], u[:, -1]
        u0_norm = jnp.sqrt(jnp.sum(u0**2, axis=tuple(range(3, u0.ndim)), keepdims=True))

        key, key_noise = jax.random.split(key)
        noise = self.noise_scale * u0_norm * jax.random.normal(key_noise, u0.shape)

        keys = jax.random.split(key, u0.shape[0])
        u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
            lambda u_, key_: model(u_, args, key=key_)
        )(u0 + noise, keys)[0]
        return jnp.mean((u1 - u1_pred) ** 2), dict()


class PushforwardOneStepLoss[M: eqx.Module, B: PyTree[Array]](eqx.Module):
    """`OneStepLoss` on an input whose last frame is the model's own prediction. The
    model first predicts the second-to-last frame from the frames before it, without
    gradient; the loss is then the error on the last frame from the input shifted by
    one frame. A window therefore needs one frame more than for `OneStepLoss`."""

    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        """Return the loss of `model` on `batch` and an empty dictionary of further
        metrics."""
        u: Float[Array, "batch time dim ..."] = batch
        # Different from other methods; thus to use this method dataloader segment size
        # must be set to be one larger usual
        u0_prev, u1 = u[:, :-2], u[:, -1]

        key_prev, key = jax.random.split(key)
        u1_prev = jax.lax.stop_gradient(
            eqx.filter_vmap(lambda u_, key_: model(u_, args, key=key_))(
                u0_prev, jax.random.split(key_prev, u0_prev.shape[0])
            )[0]
        )

        u0 = jnp.concatenate((u0_prev[:, 1:], jnp.expand_dims(u1_prev, axis=1)), axis=1)
        keys = jax.random.split(key, u0.shape[0])
        u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
            lambda u_, key_: model(u_, args, key=key_)
        )(u0, keys)[0]
        return jnp.mean((u1 - u1_pred) ** 2), dict()

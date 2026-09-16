from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.multiphysics.hyperfluxfno.target_networks import (
    AbstractTargetNetwork,
)
from context_flux_no.nn.channelwise import ChannelwiseMLP
from context_flux_no.nn.misc import apply_along_axis
from context_flux_no.nn.operators.fourier_utils import append_grid_channels


class GlobalResponseNorm(eqx.Module):
    gamma: Float[Array, " channels"]
    beta: Float[Array, " channels"]

    def __init__(
        self,
        channels: int,
        dtype=None,
    ):
        self.gamma = jnp.zeros(shape=(channels,), dtype=dtype)
        self.beta = jnp.zeros(shape=(channels,), dtype=dtype)

    def __call__(self, x: Float[Array, " channels *rest"], *, key=None):
        g: Float[Array, " channels *1"] = jax.vmap(
            lambda _x: jnp.linalg.norm(_x, keepdims=True)
        )(x)
        n: Float[Array, " channels *1"] = g / (jnp.mean(g, keepdims=True) + 1e-6)
        gamma = jnp.reshape(self.gamma, n.shape)
        beta = jnp.reshape(self.beta, n.shape)
        return x * n * gamma + beta + x


class ConvNeXtV2(eqx.Module):
    num_spatial_dims: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    expansion_ratio: int = eqx.field(static=True)

    conv_dw: eqx.nn.Conv
    norm: eqx.nn.LayerNorm
    mlp: eqx.nn.Sequential

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        kernel_size: int,
        expansion_ratio: int = 4,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)

        self.num_spatial_dims = num_spatial_dims
        self.channels = channels
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size

        self.conv_dw = eqx.nn.Conv(
            num_spatial_dims=self.num_spatial_dims,
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.channels,
            padding_mode="CIRCULAR",
            dtype=dtype,
            key=keys[0],
        )
        self.norm = eqx.nn.LayerNorm(shape=(self.channels,), dtype=dtype)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Conv(
                    num_spatial_dims=self.num_spatial_dims,
                    in_channels=self.channels,
                    out_channels=self.hidden_channels,
                    kernel_size=1,
                    dtype=dtype,
                    key=keys[1],
                ),
                eqx.nn.Lambda(jax.nn.gelu),
                GlobalResponseNorm(channels=self.hidden_channels),
                eqx.nn.Conv(
                    num_spatial_dims=self.num_spatial_dims,
                    in_channels=self.hidden_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    dtype=dtype,
                    key=keys[2],
                ),
            ]
        )

    @property
    def hidden_channels(self) -> int:
        return self.channels * self.expansion_ratio

    def __call__(
        self, x: Float[Array, " channels *rest"]
    ) -> Float[Array, " channels *rest"]:
        y = self.conv_dw(x)
        y = apply_along_axis(self.norm, y, axis=0)
        y = self.mlp(y)
        return x + y


class ConvNeXtV2TargetNetwork(AbstractTargetNetwork):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)

    lift_layer: ChannelwiseMLP
    blocks: tuple[ConvNeXtV2, ...]
    project_layer: ChannelwiseMLP

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        lift_dim: int,
        kernel_size: int,
        depth: int,
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_grid = stack_grid

        in_channels_ = in_channels + num_spatial_dims if stack_grid else in_channels
        self.lift_layer = ChannelwiseMLP(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels_,
            out_channels=lift_dim,
            hidden_channels=lift_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=keys[0],
        )
        self.blocks = tuple(
            [
                ConvNeXtV2(
                    num_spatial_dims=num_spatial_dims,
                    channels=lift_dim,
                    kernel_size=3,
                    key=k,
                )
                for k in jax.random.split(keys[1], depth)
            ]
        )
        self.project_layer = ChannelwiseMLP(
            num_spatial_dims=num_spatial_dims,
            in_channels=lift_dim,
            out_channels=in_channels * num_spatial_dims,
            hidden_channels=lift_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=keys[2],
        )

    def __call__(
        self,
        u: Float[Array, " in_channels *spatial_dims"],
        args: tuple[float, ...],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        del key, inference
        dt, *dxs = args

        v = append_grid_channels(u) if self.stack_grid else u
        v = jnp.pad(
            v, pad_width=[(0, 0)] + [(1, 0)] * self.num_spatial_dims, mode="wrap"
        )
        v: Float[Array, " lift_dim *spatial_dims_plus_1"] = self.lift_layer(v)
        for convnextv2 in self.blocks:
            v: Float[Array, " lift_dim *spatial_dims_plus_1"] = convnextv2(v)
        f_: Float[Array, "num_spatial_dims in_channels *spatial_dims_plus_1"] = (
            rearrange(
                self.project_layer(v), "(N C) ... -> N C ...", N=self.num_spatial_dims
            )
        )
        spatial_slices = tuple(slice(0, s) for s in u.shape[1:])
        for i, dx in enumerate(dxs):
            df = jnp.diff(f_[i], axis=i + 1)[:, *spatial_slices]
            u = u - dt * df / dx
        return u

from collections.abc import Callable, Sequence
from itertools import accumulate

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, rearrange, unpack
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.multiphysics.hyperfluxfno.target_networks import (
    AbstractTargetNetwork,
)
from context_flux_no.nn.channelwise import ChannelwiseMLP
from context_flux_no.nn.operators.fourier_utils import append_grid_channels


class LargeKernelSelection(eqx.Module):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    kernel_sizes: tuple[int, ...] = eqx.field(static=True)
    dilations: tuple[int, ...] = eqx.field(static=True)

    spatial_kernels: tuple[eqx.nn.Conv, ...]
    conv1x1: eqx.nn.Conv
    conv_squeeze: eqx.nn.Conv
    conv_out: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        hidden_channels: int,
        kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        dilations: Sequence[int] | None = None,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        # TODO: check kernel_sizes is nondecreasing
        # TODO: check whether dilations satisfy the no gap condition
        self.kernel_sizes = tuple(kernel_sizes)
        self.dilations = (
            tuple(dilations)
            if dilations is not None
            else tuple(range(1, self.num_kernels + 1))
        )
        keys = jax.random.split(key, self.num_kernels + 3)
        self.spatial_kernels = tuple(
            [
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=k,
                    dilation=d,
                    groups=in_channels,
                    padding=d * (k - 1) // 2,
                    padding_mode="CIRCULAR",  # Need to change wrt BCs
                    dtype=dtype,
                    key=keys[i],
                )
                for i, (k, d) in enumerate(zip(self.kernel_sizes, self.dilations))
            ]
        )
        self.conv1x1 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels * self.num_kernels,
            out_channels=hidden_channels * self.num_kernels,
            kernel_size=1,
            groups=self.num_kernels,
            dtype=dtype,
            key=keys[-3],
        )
        self.conv_squeeze = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=2,
            out_channels=self.num_kernels,
            kernel_size=7,
            padding=3,
            padding_mode="CIRCULAR",
            key=keys[-2],
        )
        self.conv_out = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=1,
            key=keys[-1],
        )
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels

    @property
    def num_kernels(self) -> int:
        return len(self.kernel_sizes)

    @property
    def receptive_field_sizes(self) -> list[int]:
        k0, ks = self.kernel_sizes[0], self.kernel_sizes[1:]
        return list(
            accumulate(
                (d * (k - 1) for k, d in zip(ks, self.dilations[1:])),
                initial=k0,
            )
        )

    def __call__(
        self, x: Float[Array, " in_channels *grids"]
    ) -> Float[Array, " in_channels *grids_plus_1"]:
        us = []
        u = x
        for spatial_kernel in self.spatial_kernels:
            u = spatial_kernel(u)
            us.append(u)

        U: Float[Array, " num_kernels*in_channels *grids"] = jnp.concatenate(us, axis=0)
        U: Float[Array, " num_kernels*hidden_channels *grids"] = self.conv1x1(U)
        P: Float[Array, " 2 *grids"] = jnp.stack(
            (jnp.max(U, axis=0), jnp.mean(U, axis=0)), axis=0
        )
        weights: Float[Array, " num_kernels *grids"] = jax.nn.sigmoid(
            self.conv_squeeze(P)
        )
        S: Float[Array, " hidden_channels *grids"] = jnp.einsum(
            "i...,ij...->j...",
            weights,
            rearrange(U, "(K C) ... -> K C ...", K=self.num_kernels),
        )
        S: Float[Array, " in_channels *grids"] = self.conv_out(S)
        return x * S


class KernelSelectionBlock(eqx.Module):
    num_spatial_dims: int = eqx.field(static=True)

    activation: Callable
    fc1: eqx.nn.Conv
    kernel_select: LargeKernelSelection
    fc2: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        hidden_channels,
        kernel_sizes,
        dilations,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        keys = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[0],
        )
        self.kernel_select = LargeKernelSelection(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dtype=dtype,
            key=keys[1],
        )
        self.activation = activation
        self.fc2 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[2],
        )

    def __call__(
        self, x: Float[Array, " in_channels *grids"]
    ) -> Float[Array, " in_channels *grids"]:
        y: Float[Array, " in_channels *grids"] = self.activation(self.fc1(x))
        y: Float[Array, " in_channels *grids"] = self.fc2(self.kernel_select(y))
        return x + y


class MLPBlock(eqx.Module):
    num_spatial_dims: int = eqx.field(static=True)
    fc1: eqx.nn.Conv
    conv_depthwise: eqx.nn.Conv
    activation: Callable
    fc2: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        keys = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[0],
        )
        self.conv_depthwise = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="CIRCULAR",
            groups=hidden_channels,
            dtype=dtype,
            key=keys[1],
        )
        self.activation = activation
        self.fc2 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[2],
        )

    def __call__(
        self, x: Float[Array, " in_channels *grids"]
    ) -> Float[Array, " in_channels *grids"]:
        x = self.fc1(x)
        x = self.activation(self.conv_depthwise(x))
        return self.fc2(x)


class LSKBlock(eqx.Module):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)

    norm1: eqx.nn.RMSNorm
    kernel_select: KernelSelectionBlock
    norm2: eqx.nn.RMSNorm
    mlp: MLPBlock
    layer_scale1: Float[Array, " in_channels"]
    layer_scale2: Float[Array, " in_channels"]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        hidden_channels: int,
        kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        dilations: Sequence[int] | None = None,
        activation: Callable = jax.nn.gelu,
        layer_scale_init: float = 1e-2,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        keys = jax.random.split(key, 2)

        self.norm1 = eqx.nn.RMSNorm(
            shape=(in_channels,), use_weight=True, use_bias=True, dtype=dtype
        )
        self.norm2 = eqx.nn.RMSNorm(
            shape=(in_channels,), use_weight=True, use_bias=True, dtype=dtype
        )
        self.kernel_select = KernelSelectionBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            hidden_channels=1,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            activation=activation,
            dtype=dtype,
            key=keys[0],
        )
        self.mlp = MLPBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            dtype=dtype,
            key=keys[1],
        )
        dtype = default_floating_dtype() if dtype is None else dtype
        layerscale_shape = (in_channels,) + (1,) * num_spatial_dims
        self.layer_scale1 = jnp.ones(layerscale_shape, dtype=dtype) * layer_scale_init
        self.layer_scale2 = jnp.ones(layerscale_shape, dtype=dtype) * layer_scale_init

    def __call__(
        self, x: Float[Array, " in_channels *grids"]
    ) -> Float[Array, " in_channels *grids"]:
        y = self.kernel_select(self.apply_norm(self.norm1, x))
        y = x + self.layer_scale1 * y

        z = self.mlp(self.apply_norm(self.norm2, y))
        z = y + self.layer_scale2 * z
        return z

    def apply_norm(
        self, norm: eqx.nn.RMSNorm, x: Float[Array, " in_channels *grids"]
    ) -> Float[Array, " in_channels *grids"]:
        x_, ps = pack([x], "C *")
        normed_ = eqx.filter_vmap(norm, in_axes=-1, out_axes=-1)(x_)
        return unpack(normed_, ps, "C *")[0]


class LSKFluxNOTargetNetwork(AbstractTargetNetwork):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)

    lift_layer: ChannelwiseMLP
    lskblock: LSKBlock
    project_layer: ChannelwiseMLP

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        lift_dim: int,
        depth: int,
        kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        dilations: Sequence[int] | None = None,
        activation: Callable = jax.nn.gelu,
        layer_scale_init: float = 1e-2,
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
        self.lskblock = LSKBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=lift_dim,
            hidden_channels=lift_dim,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            activation=activation,
            layer_scale_init=layer_scale_init,
            dtype=dtype,
            key=keys[1],
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
        v: Float[Array, " lift_dim *spatial_dims_plus_1"] = self.lskblock(v)
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

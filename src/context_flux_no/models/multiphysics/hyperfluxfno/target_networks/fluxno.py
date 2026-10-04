from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, rearrange, unpack
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn.channelwise import ChannelwiseMLP
from context_flux_no.nn.operators.fourier_utils import append_grid_channels

from .base import AbstractTargetNetwork


class NDNeuralNetworkFlux(eqx.Module):
    """An N-D version of the Flux NO that uses full ND hyperrectangular finite volume
    stencil, instead of using dimensional splitting.

    For the flux along `normal_axis`, the stencil spans (a + 1, b) cells to the
    left/right of each face along the normal axis, and is centered with halfwidth s
    along every transverse axis. The same lift kernel is shared across directions and
    is oriented so that its first spatial axis lies along the normal axis.
    """

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    normal_stencil_halfwidths: tuple[int, int] = eqx.field(static=True)
    transverse_stencil_halfwidth: int = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    lift_layer: eqx.nn.Conv
    mlp: ChannelwiseMLP

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        normal_stencil_halfwidths: tuple[int, int],
        transverse_stencil_halfwidth: int,
        lift_dim: int,
        hidden_dim: int,
        depth: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        if any(
            w < 0 for w in (*normal_stencil_halfwidths, transverse_stencil_halfwidth)
        ):
            raise ValueError("Stencil widths must be non-negative")
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normal_stencil_halfwidths = normal_stencil_halfwidths
        self.transverse_stencil_halfwidth = transverse_stencil_halfwidth
        self.lift_dim = lift_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        keys = jax.random.split(key, 2)
        # Kernel layout: (normal, transverse, ..., transverse).
        kernel_size = (self.normal_stencil_width,) + (
            self.transverse_stencil_width,
        ) * (num_spatial_dims - 1)
        self.lift_layer = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=lift_dim,
            kernel_size=kernel_size,
            dtype=dtype,
            key=keys[0],
        )
        self.mlp = ChannelwiseMLP(
            num_spatial_dims=num_spatial_dims,
            in_channels=lift_dim,
            out_channels=out_channels * num_spatial_dims,
            hidden_channels=hidden_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=keys[1],
        )

    @property
    def normal_stencil_width(self) -> int:
        return sum(self.normal_stencil_halfwidths) + 1

    @property
    def transverse_stencil_width(self) -> int:
        return 2 * self.transverse_stencil_halfwidth + 1

    def pad_ghost_cells(
        self, u: Float[Array, " in_channels *grids"], normal_axis: int
    ) -> Float[Array, " in_channels *grids_padded"]:
        a, b = self.normal_stencil_halfwidths
        s = self.transverse_stencil_halfwidth

        pad = [(s, s)] * self.num_spatial_dims
        pad[normal_axis] = (a + 1, b)  # N+1 faces along the normal axis
        # Need to change mode if not periodic boundary condition
        return jnp.pad(u, [(0, 0)] + pad, mode="wrap")

    def __call__(
        self,
        u: Float[Array, " in_channels *grids"],
        normal_axis: int,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_channels *faces"]:
        """Flux through the faces normal to `normal_axis`: N+1 entries along the
        normal axis, N along every transverse axis."""
        del key
        u_padded = self.pad_ghost_cells(u, normal_axis)
        # Orient the (normal, transverse, ...) kernel along `normal_axis` by moving the
        # normal axis of the input to the front, and back again after the convolution.
        u_stencils: Float[Array, " lift_dim *faces"] = jnp.moveaxis(
            self.lift_layer(jnp.moveaxis(u_padded, 1 + normal_axis, 1)),
            1,
            1 + normal_axis,
        )
        f: Float[Array, " num_spatial_dims*out_channels *faces"] = self.mlp(u_stencils)
        return rearrange(f, "(D C) ... -> D C ...", D=self.num_spatial_dims)[
            normal_axis
        ]


class NDFluxNOTargetNetwork(AbstractTargetNetwork):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)

    fluxes: NDNeuralNetworkFlux

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        lift_dim: int,
        depth: int,
        normal_stencil_halfwidths: tuple[int, int],
        transverse_stencil_halfwidth: int,
        hidden_dim: int,
        stack_grid: bool = True,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        in_channels_ = in_channels + num_spatial_dims if stack_grid else in_channels
        self.fluxes = NDNeuralNetworkFlux(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels_,
            out_channels=out_channels,
            normal_stencil_halfwidths=normal_stencil_halfwidths,
            transverse_stencil_halfwidth=transverse_stencil_halfwidth,
            lift_dim=lift_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=key,
        )
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_grid = stack_grid

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

        # All directional fluxes are evaluated at the same (un-updated) state v.
        v = append_grid_channels(u) if self.stack_grid else u
        for i, dx in enumerate(dxs):
            f_i: Float[Array, " out_channels *faces"] = self.fluxes(v, normal_axis=i)
            u = u - dt * jnp.diff(f_i, axis=i + 1) / dx
        return u


class NeuralNetworkFlux(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stencil_widths: tuple[int, int] = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    lift_layer: eqx.nn.Conv1d
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stencil_widths: tuple[int, int],
        lift_dim: int,
        hidden_dim: int,
        depth: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 2)
        kernel_size = stencil_widths[0] + stencil_widths[1] + 1
        self.lift_layer = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=lift_dim,
            kernel_size=kernel_size,
            dtype=dtype,
            key=keys[0],
        )
        self.mlp = eqx.nn.MLP(
            in_size=lift_dim,
            out_size=out_channels,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=keys[1],
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stencil_widths = stencil_widths
        self.lift_dim = lift_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

    @property
    def stencil_size(self) -> int:
        return sum(self.stencil_widths) + 1

    def __call__(
        self,
        u: Float[Array, "in_channels grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "out_channels grids+1"]:
        del key
        a, b = self.stencil_widths
        pad_widths = [(0, 0), (a + 1, b)]
        # Need to change mode if not periodic boundary condition
        u_padded = jnp.pad(u, pad_widths, mode="wrap")
        u_stencils: Float[Array, "lift_dim grids_x+1"] = self.lift_layer(u_padded)
        f: Float[Array, "out_channels grids_x+1"] = eqx.filter_vmap(
            self.mlp, in_axes=-1, out_axes=-1
        )(u_stencils)
        return f


class FluxNOTargetNetwork(AbstractTargetNetwork):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)

    fluxes: tuple[NeuralNetworkFlux]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        lift_dim: int,
        depth: int,
        stencil_widths: tuple[int, int],
        hidden_dim: int,
        stack_grid: bool = True,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        in_channels_ = in_channels + num_spatial_dims if stack_grid else in_channels
        fluxes = [
            NeuralNetworkFlux(
                in_channels=in_channels_,
                out_channels=out_channels,
                stencil_widths=stencil_widths,
                lift_dim=lift_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for k in jax.random.split(key, num_spatial_dims)
        ]
        self.fluxes = tuple(fluxes)
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_grid = stack_grid

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
        for i, (flux_func, dx) in enumerate(zip(self.fluxes, dxs)):
            df = self.compute_flux_difference(flux_func, v, spatial_axis=i)
            u = u - dt * df / dx
        return u

    def compute_flux_difference(
        self,
        flux_func: Callable[
            [Float[Array, "in_channels x"]], Float[Array, "in_channels x+1"]
        ],
        v: Float[Array, " in_channels *spatial_dims"],
        spatial_axis: int,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        v = jnp.swapaxes(v, spatial_axis + 1, 1)
        v_, ps = pack([v], "C S *")
        f_ = eqx.filter_vmap(flux_func, in_axes=-1, out_axes=-1)(v_)
        f = unpack(f_, ps, "C S *")[0]
        df = jnp.diff(f, axis=1)
        df = jnp.swapaxes(df, spatial_axis + 1, 1)
        return df

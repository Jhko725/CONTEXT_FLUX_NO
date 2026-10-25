from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from clawpack import riemann
from jaxtyping import Float

from context_flux_no.simulations.pde.base import AbstractPDEFamily
from context_flux_no.simulations.pyclaw import PyClawSpec
from context_flux_no.simulations.trixi import TrixiSpec


class Burgers1D(AbstractPDEFamily):
    """Inviscid Burgers through Clawpack's compiled `riemann.burgers_1D` (entropy fix on by
    default), van Leer limiter."""

    n_spatial_dims: ClassVar[int] = 1
    field_rank_names: ClassVar[tuple[tuple[int, str], ...]] = ((0, "u"),)
    nu: float = eqx.field(static=True, default=0.0)
    entropy_fix: bool = eqx.field(static=True, default=True)

    @property
    def parameters(self) -> dict[str, float]:
        return {"nu": self.nu}

    @property
    def pyclaw_spec(self) -> PyClawSpec:
        return PyClawSpec(
            riemann_solver=riemann.burgers_1D,
            num_waves=1,
            kernel_language="Fortran",
            problem_data={"efix": self.entropy_fix},
            classic_options={"limiters": "vanleer"},
        )

    @property
    def trixi_spec(self) -> TrixiSpec:
        return TrixiSpec(
            "InviscidBurgersEquation1D"
        )  # Trixi's own; `nu` unused, as in PyClaw


class ViscousBurgers1D(AbstractPDEFamily):
    """
    Parametric viscous Burgers-type equation:

        u_t + a (u^2)_x = b u_xx

    Equivalently:

        u_t + f(u)_x = b u_xx,
        f(u) = a u^2.

    Parameters:
        a: nonlinear advection coefficient
        b: diffusion/viscosity coefficient

    This solver assumes periodic boundary conditions.
    """

    n_spatial_dims: ClassVar[int] = 1
    field_rank_names: ClassVar[tuple[tuple[int, str], ...]] = ((0, "u"),)

    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0, b: float = 0.01):
        self.a = a
        self.b = b

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b}

    def flux(
        self,
        u: Float[np.ndarray, " Nx"],
    ) -> Float[np.ndarray, " Nx"]:
        return self.a * u**2

    def solve(
        self,
        ic,
        x_spans,
        Nxs,
        t_span: tuple[float, float],
        Nt: int,
        bc: str = "periodic",
        cfl: float = 0.4,
        max_steps: int = 10_000_000,
        **kwargs,
    ) -> tuple[
        Float[np.ndarray, "time dim x_grid"],
        Float[np.ndarray, " time"],
        Float[np.ndarray, " x_grid"],
    ]:
        """
        Solves

            u_t + a (u^2)_x = b u_xx

        using:
            - finite volume Rusanov flux for the nonlinear hyperbolic term
            - centered finite difference for diffusion
            - explicit time stepping
            - periodic boundary condition

        Returns:
            (u, t, xs) with u.shape = (Nt, 1, Nx), xs = (x_grid,)
        """
        if kwargs.pop("backend", None) is not None:
            raise ValueError(
                f"{type(self).__name__} uses its own stepper; backend= is not supported"
            )

        if bc != "periodic":
            raise NotImplementedError(
                "Currently only periodic boundary conditions are supported."
            )

        ((x0, x1),) = (tuple(s) for s in x_spans)  # 1-D only
        (Nx,) = Nxs
        t0, t1 = t_span

        dx = (x1 - x0) / Nx
        x_grid = x0 + (np.arange(Nx) + 0.5) * dx  # cell centres (F8: was left edges)

        t_save = np.linspace(t0, t1, Nt + 1)

        u = jnp.squeeze(jnp.asarray(ic.at_points([x_grid])))

        if u.ndim != 1:
            raise ValueError(
                f"initial condition must have shape (1, Nx), got {u.shape}."
            )

        u_save = np.zeros((Nt + 1, 1, Nx), dtype=float)
        u_save[0, 0, :] = u

        t = t0
        save_idx = 1
        step = 0

        @eqx.filter_jit
        def _step_fn(u, dt):
            return self._step_periodic(u, dx, dt)

        while save_idx <= Nt:  # F8: `< Nt` never wrote the last frame
            if step > max_steps:
                raise RuntimeError(
                    f"Maximum number of steps exceeded: {max_steps}. "
                    "Try increasing b, decreasing Nx, or increasing cfl carefully."
                )

            # Characteristic speed for f(u) = a u^2:
            # f'(u) = 2 a u
            max_adv_speed = np.max(np.abs(2.0 * self.a * u))

            if max_adv_speed > 1e-14:
                dt_adv = dx / max_adv_speed
            else:
                dt_adv = np.inf

            if self.b > 1e-14:
                dt_diff = 0.5 * dx**2 / self.b
            else:
                dt_diff = np.inf

            dt = cfl * min(dt_adv, dt_diff)

            next_save_t = t_save[save_idx]
            if t + dt > next_save_t:
                dt = next_save_t - t

            if dt <= 0:
                u_save[save_idx, 0, :] = u
                save_idx += 1
                continue

            u = _step_fn(u, dt)

            t += dt
            step += 1

            if abs(t - next_save_t) < 1e-12:
                u_save[save_idx, 0, :] = u
                save_idx += 1

        return u_save, t_save, (x_grid,)

    def _step_periodic(
        self,
        u: Float[np.ndarray, " Nx"],
        dx: float,
        dt: float,
    ) -> Float[np.ndarray, " Nx"]:
        """
        One explicit finite-volume / finite-difference step.

        Hyperbolic part:
            u_t + f(u)_x = 0

        Diffusion part:
            u_t = b u_xx
        """

        u_l = u
        u_r = jnp.roll(u, -1)

        f_l = self.flux(u_l)
        f_r = self.flux(u_r)

        # Local Rusanov speed at each interface i+1/2.
        # max |f'(u)| = max |2 a u|
        smax = jnp.maximum(
            jnp.abs(2.0 * self.a * u_l),
            jnp.abs(2.0 * self.a * u_r),
        )

        # Numerical flux F_{i+1/2}
        flux_half = 0.5 * (f_l + f_r) - 0.5 * smax * (u_r - u_l)

        # Conservative hyperbolic update:
        # -(F_{i+1/2} - F_{i-1/2}) / dx
        hyperbolic_rhs = -(flux_half - jnp.roll(flux_half, 1)) / dx

        # Centered diffusion:
        # b * (u_{i+1} - 2u_i + u_{i-1}) / dx^2
        diffusion_rhs = self.b * (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)) / dx**2

        return u + dt * (hyperbolic_rhs + diffusion_rhs)

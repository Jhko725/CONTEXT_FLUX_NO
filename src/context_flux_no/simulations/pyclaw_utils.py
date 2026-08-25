from typing import Sequence
from collections.abc import Callable

import numpy as np
from clawpack import pyclaw
from jaxtyping import Float


def bc_from_string(bc_name: str) -> pyclaw.BC:
    match bc_name:
        case "periodic":
            bc_pyclaw = pyclaw.BC.periodic
        case _:
            raise ValueError("Unrecognized boundary condition")
    return bc_pyclaw


def make_controller(
    state: pyclaw.State, domain: pyclaw.Domain, t_span: tuple[float, float], Nt: int
) -> pyclaw.Controller:
    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.tfinal = t_span[1] - t_span[0]
    # Manually specify output times
    controller.output_style = 2
    controller.out_times = list(np.linspace(*t_span, Nt + 1, endpoint=True))
    # By default, keep the solution in memory and do not save
    # See https://www.clawpack.org/pyclaw/output.html for details
    controller.keep_copy = True
    controller.output_format = None

    return controller


def solution_from_controller(
    controller: pyclaw.Controller,
) -> Float[np.ndarray, "time dim *grids"]:
    return np.stack([f.state.q for f in controller.frames], axis=0)


def grid_centers_from_state(
    state: pyclaw.State,
) -> list[Float[np.ndarray, " grid"]]:
    grid = state.grid
    grid_centers = [getattr(grid, dim_name).centers for dim_name in grid._dimensions]
    return grid_centers


def make_domain(x_spans: Sequence[tuple[float, float]], Nxs: Sequence[int]):
    if (n_spatial_dims := len(x_spans)) != len(Nxs):
        raise ValueError("x_spans and Nxs must have the same length.")

    if n_spatial_dims <= 3:
        dim_names = ["x", "y", "z"][:n_spatial_dims]
    else:
        dim_names = [f"x_{i}" for i in range(n_spatial_dims)]

    dims = [
        pyclaw.Dimension(*x_span, Nx, name=name)
        for x_span, Nx, name in zip(x_spans, Nxs, dim_names)
    ]
    print(dims)
    return pyclaw.Domain(dims)


def apply_initial_condition(
    state: pyclaw.State,
    ic_factory: Callable[
        [Float[np.ndarray, " *x_grid"]], Float[np.ndarray, " *x_grid"]
    ],
) -> None:
    grid_centers = grid_centers_from_state(state)
    state.q = np.asfortranarray(ic_factory(grid_centers))

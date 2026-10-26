import logging
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from clawpack import pyclaw
from jaxtyping import Float

from .pyclaw_utils import (
    apply_initial_condition,
    bc_from_string,
    grid_centers_from_state,
    make_controller,
    make_domain,
    solution_from_controller,
)


ClawSolver = pyclaw.ClawSolver1D | pyclaw.ClawSolver2D | pyclaw.ClawSolver3D


def pdesolve_pyclaw(
    solver: ClawSolver,
    problem_data: dict[str, float],
    ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
    x_spans: Sequence[tuple[float, float]],
    Nxs: Sequence[int],
    t_span: tuple[float, float],
    Nt: int,
    bc: Literal["periodic"],
    *,
    verbose: bool = True,
) -> tuple[
    Float[np.ndarray, "time dim *xs"],
    Float[np.ndarray, " time"],
    list[Float[np.ndarray, " ?x"]],
]:
    if not verbose:
        # See https://www.clawpack.org/pyclaw/output.html#logging
        logger = logging.getLogger("pyclaw")
        logger.setLevel(logging.CRITICAL)

    # Need to change for per-axis boundary conditions
    solver.all_bcs = bc_from_string(bc)
    domain = make_domain(x_spans, Nxs)

    state = pyclaw.State(domain, solver.num_eqn)
    state.problem_data.update(problem_data)
    apply_initial_condition(state, ic_factory)

    controller = make_controller(state, domain, t_span, Nt)
    controller.solver = solver

    _ = controller.run()

    u = solution_from_controller(controller)
    t = np.asarray(controller.out_times)
    xs = grid_centers_from_state(state)
    return u, t, xs

"""`generate_dataset`: sample parameters and initial conditions, run a `SolverBackend`, write
a Well store with its `DatasetRecord` (replaces `simulations/utils.py`).

Sampling is unchanged from the ICLR script: `n_coeffs` parameter sets (uniform or linspace in
`coeff_range_dict`) × `n_ics_per_coeff` initial conditions, keys derived from
`(split, seed)`. What is new: the backend is an argument (default Classic PyClaw), per-
trajectory solver failures are counted into the record instead of silently dropped, and the
store is finalised with the number actually written."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from context_flux_no.simulations.backend import normalize_bcs, SolverBackend
from context_flux_no.simulations.grid import Grid
from context_flux_no.simulations.record import DatasetRecord
from context_flux_no.simulations.writer import WellWriter


log = logging.getLogger(__name__)

SPLIT_KEYS = {"train": 0, "valid": 1, "test": 2}


def sample_coefficients(
    key: PRNGKeyArray,
    coeff_range_dict: Mapping[str, tuple[float, float]],
    strategy: Literal["uniform", "linspace"],
    n_coeffs: int,
) -> dict[str, Array]:
    subkeys = jax.random.split(key, len(coeff_range_dict))
    if strategy == "uniform":

        def _sample(key_, rng):
            return jax.random.uniform(key_, (n_coeffs,), minval=rng[0], maxval=rng[1])

    elif strategy == "linspace":

        def _sample(key_, rng):
            return jnp.linspace(rng[0], rng[1], n_coeffs)

    else:
        raise ValueError(f"unknown coefficient sampling strategy {strategy!r}")
    return {
        name: _sample(k, rng)
        for k, (name, rng) in zip(subkeys, coeff_range_dict.items())
    }


def generate_dataset(
    dataset_name: str,
    n_coeffs: int,
    n_ics_per_coeff: int,
    pde_factory: Callable[..., Any],
    initial_condition_fn: Callable[[Sequence[np.ndarray], PRNGKeyArray], Array],
    coeff_range_dict: Mapping[str, tuple[float, float]],
    x_spans: Sequence[Sequence[float]],
    Nxs: Sequence[int],
    t_span: tuple[float, float],
    Nt: int,
    bc: str | Sequence[Any] = "periodic",
    dataset_type: Literal["train", "valid", "test"] = "train",
    coeff_sampling_strategy: Literal["uniform", "linspace"] = "uniform",
    savedir: str | Path = "./",
    filename: str | None = None,
    codec: Any | None = "default",
    seed: int = 0,
    backend: SolverBackend | None = None,
    ic_description: Mapping[str, Any] | None = None,
    **solve_kwargs: Any,
) -> WellWriter:
    """Writes `<savedir>/<dataset_name>/data/<split>/<filename>` and returns the writer
    (finalised). `backend=None` means each PDE class's default (Classic PyClaw for the
    PyClaw families, the class's own stepper otherwise)."""
    filename = filename or f"{dataset_name}_{seed=}.zarr"
    savepath = Path(savedir) / dataset_name / "data" / dataset_type / filename
    grid = Grid(x_spans, Nxs)
    t_out = np.linspace(t_span[0], t_span[1], Nt + 1, endpoint=True)
    bcs = normalize_bcs(bc, grid.n_spatial_dims)

    base_key = jax.random.key(SPLIT_KEYS[dataset_type])
    keys = jax.random.split(jax.random.fold_in(base_key, seed), n_coeffs + 1)
    coeffs = sample_coefficients(
        keys[0], coeff_range_dict, coeff_sampling_strategy, n_coeffs
    )

    pde0 = pde_factory(**{k: float(v[0]) for k, v in coeffs.items()})
    record = DatasetRecord.create(
        family=type(pde0).__name__,
        dataset_name=dataset_name,
        split=dataset_type,
        n_trajectories=n_coeffs * n_ics_per_coeff,
        grid=grid.to_dict(),
        time={
            "t_span": [float(t_span[0]), float(t_span[1])],
            "Nt": int(Nt),
            "dt": float(t_out[1] - t_out[0]),
        },
        bcs=bcs,
        field_rank_names=pde0.field_rank_names,
        params={
            "names": list(coeff_range_dict),
            "ranges": {
                k: [float(a), float(b)] for k, (a, b) in coeff_range_dict.items()
            },
            "sampling": coeff_sampling_strategy,
            "n_coeffs": int(n_coeffs),
            "n_ics_per_coeff": int(n_ics_per_coeff),
        },
        ic=dict(
            ic_description
            or {
                "name": getattr(
                    initial_condition_fn, "__qualname__", str(initial_condition_fn)
                )
            }
        ),
        backend=(backend or pde0.default_backend()).describe(),
        seed=seed,
        extra={"solve_kwargs": {k: _jsonable(v) for k, v in solve_kwargs.items()}},
    )
    writer = WellWriter.create(
        savepath, pde=pde0, grid=grid, t_out=t_out, bcs=bcs, record=record, codec=codec
    )

    write_idx = 0
    n_failed = 0
    for i, key in enumerate(tqdm(keys[1:], disable=n_coeffs < 2)):
        pde = pde_factory(**{k: float(v[i]) for k, v in coeffs.items()})
        for key_ic in jax.random.split(key, n_ics_per_coeff):
            try:
                u, t, xs = pde.solve(
                    lambda centers: initial_condition_fn(centers, key_ic),  # noqa: B023
                    x_spans,
                    Nxs,
                    t_span,
                    Nt,
                    bc,
                    backend=backend,
                    **solve_kwargs,
                )
                if not np.all(np.isfinite(u)):
                    raise FloatingPointError("non-finite values in the solution")
                np.testing.assert_allclose(t, t_out)
                for x_i, x_ref in zip(xs, grid.centers):
                    np.testing.assert_allclose(x_i, x_ref)
            except Exception:  # noqa: BLE001 — a failed trajectory is recorded, not fatal
                n_failed += 1
                log.exception("solve failed for coefficients %s", pde.parameters)
                continue
            writer.write_trajectory(
                write_idx, pde.solution_to_well_schema(np.asarray(u))
            )
            write_idx += 1

    writer.finalize(write_idx, n_failed=n_failed)
    return writer


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

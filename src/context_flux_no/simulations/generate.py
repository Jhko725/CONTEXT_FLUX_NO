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

from context_flux_no.simulations.backends.base import (
    normalize_bcs,
    SolverBackend,
    SolverFailure,
)
from context_flux_no.simulations.grid import Grid
from context_flux_no.simulations.initial_conditions.base import (
    Conserved,
    InitialConditionSampler,
)
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
    initial_condition: InitialConditionSampler,
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
    primitive_ic: bool = False,
    **solve_kwargs: Any,
) -> WellWriter:
    """Generates one Well-schema store of trajectories for a family of PDEs and returns its
    writer, finalized.

    `n_coeffs` parameter sets are drawn from `coeff_range_dict` (split-specific key folded
    with `seed`); for each, `n_ics_per_coeff` initial conditions are sampled on the domain,
    solved by `backend` and written in order. A trajectory whose solve raises
    `SolverFailure` (or is non-finite) is counted in the record's `n_failed`, logged with its
    parameters, and skipped; any other exception is a bug and propagates.

    **Arguments:**

    - `dataset_name`: Name of the dataset; the store is
      `<savedir>/<dataset_name>/data/<dataset_type>/<filename>`.
    - `n_coeffs`: Number of parameter sets.
    - `n_ics_per_coeff`: Number of initial conditions per parameter set.
    - `pde_factory`: Builds the family from one parameter set, called as
      `pde_factory(**{name: value})` with the names of `coeff_range_dict`.
    - `initial_condition`: Sampler of initial conditions, `sample(key, x_spans)`.
    - `coeff_range_dict`: `{parameter name: (low, high)}` for every parameter that varies.
    - `x_spans`: Domain, one `(low, high)` per axis.
    - `Nxs`: Cells per axis.
    - `t_span`: `(t0, t1)`.
    - `Nt`: Number of output steps; `Nt + 1` frames at `linspace(t0, t1, Nt + 1)`.
    - `bc`: Boundary condition name, or a `(lower, upper)` pair of names per axis
      (`normalize_bcs`).
    - `dataset_type`: `"train"`, `"valid"` or `"test"`; picks the split's base key.
    - `coeff_sampling_strategy`: `"uniform"` draws, `"linspace"` grids each range.
    - `savedir`: Root directory.
    - `filename`: Store name; `"<dataset_name>_seed=<seed>.zarr"` by default.
    - `codec`: Zarr compressor, `"default"`, or `None` for uncompressed.
    - `seed`: Folded into the split's key; distinct seeds give disjoint draws.
    - `backend`: `SolverBackend` for every trajectory; `None` means the family's default
      (Classic PyClaw for the PyClaw families, the family's own stepper otherwise).
    - `ic_description`: What goes into the record's `ic` field (the Hydra IC config);
      `{"name": <sampler class>}` by default.
    - `primitive_ic`: Whether the sampler draws primitive variables (Euler's
      `(rho, u, v, p)`); each draw is then wrapped in `Conserved(ic, pde)` before solving.
    - `**solve_kwargs`: Passed to the family's `solve` (recorded in `extra.solve_kwargs`).

    **Returns:**

    The `WellWriter`, finalized: its `record` carries the counts, provenance and
    `backend.describe()`.
    """
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
        ic={
            **dict(ic_description or {"name": type(initial_condition).__name__}),
            "primitive": bool(primitive_ic),
        },
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
                ic = initial_condition.sample(key_ic, x_spans)
                if primitive_ic:
                    ic = Conserved(ic, pde)
                u, t, xs = pde.solve(
                    ic,
                    x_spans,
                    Nxs,
                    t_span,
                    Nt,
                    bc,
                    backend=backend,
                    **solve_kwargs,
                )
                if not np.all(np.isfinite(u)):
                    raise SolverFailure(
                        type(pde).__name__, "non-finite values in the solution"
                    )
                np.testing.assert_allclose(t, t_out)
                for x_i, x_ref in zip(xs, grid.centers):
                    np.testing.assert_allclose(x_i, x_ref)
            except SolverFailure:  # a failed trajectory is recorded; a bug propagates
                n_failed += 1
                log.exception("solve failed for coefficients %s", pde.parameters)
                continue
            writer.write_trajectory(write_idx, np.asarray(u), pde.parameters)
            write_idx += 1

    writer.finalize(write_idx, n_failed=n_failed)
    return writer


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

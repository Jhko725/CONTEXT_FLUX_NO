import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import zarr
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from context_flux_no.simulations.dataset import ZarrWellDataset
from context_flux_no.simulations.pde.base import AbstractHyperbolicConservationLaw


logger = logging.getLogger(__name__)


def sample_coefficients(
    key: PRNGKeyArray,
    coeff_range_dict: dict[str, tuple[float, float]],
    strategy: Literal["uniform", "linspace"],
    n_coeffs: int = 1,
):
    if strategy == "uniform":

        def _sample(key_, coeff_range_: tuple[float, float]):
            return jax.random.uniform(
                key_, shape=(n_coeffs,), minval=coeff_range_[0], maxval=coeff_range_[1]
            )
    else:

        def _sample(key_, coeff_range_: tuple[float, float]):
            return jnp.linspace(*coeff_range_, n_coeffs)

    subkeys = jax.random.split(key, len(coeff_range_dict))
    return {
        name: _sample(subkey, coeff_range)
        for subkey, (name, coeff_range) in zip(subkeys, coeff_range_dict.items())
    }


def sample_coefficients_uniform(
    key: PRNGKeyArray, coeff_range_dict: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """
    Given a dictionary of coefficient names and a tuple indicating the range to sample
    from, return dictionary of coefficient names and values that are sampled from
    uniform distributions of the given range.
    """
    subkeys = jax.random.split(key, len(coeff_range_dict))
    return {
        name: float(
            jax.random.uniform(subkey, minval=coeff_range[0], maxval=coeff_range[1])
        )
        for subkey, (name, coeff_range) in zip(subkeys, coeff_range_dict.items())
    }


BASE_KEY_DICT = {"train": 0, "validation": 1, "test": 2}


# TODO: change x_span, Nx, etc to be compatible with N-D simulations
def generate_dataset(
    dataset_name: str,
    n_coeffs: int,
    n_ics_per_coeff: int,
    pde_factory: Callable[[Any], eqx.Module],
    initial_condition_fn: Callable[[Array, PRNGKeyArray], Array],
    coeff_range_dict: dict,
    x_spans: Sequence[tuple[float, float]],
    Nxs: Sequence[int],
    t_span: tuple[float, float],
    Nt: int,
    bc: Literal["periodic"] = "periodic",
    dataset_type: Literal["train", "valid", "test"] = "train",
    coeff_sampling_strategy: Literal["uniform", "linspace"] = "uniform",
    savedir: str | Path = "./",
    filename: str | None = None,
    codec: zarr.abc.codec.Codec | None = zarr.codecs.BloscCodec(
        cname="zstd", clevel=3, shuffle="bitshuffle"
    ),
    seed: int = 0,
    **solve_kwargs,
):
    if filename is None:
        filename = f"{dataset_name}_{seed=}.zarr"
    savepath = Path(savedir) / dataset_name / "data" / dataset_type / filename
    base_key = jax.random.key({"train": 0, "valid": 1, "test": 2}[dataset_type])
    keys = jax.random.split(jax.random.fold_in(base_key, seed), n_coeffs + 1)
    coeffs = sample_coefficients(
        keys[0], coeff_range_dict, coeff_sampling_strategy, n_coeffs
    )

    write_idx = 0
    for i, key in enumerate(tqdm(keys[1:])):
        key_ics = jax.random.split(key, n_ics_per_coeff)
        coeffs_i = {k: float(v[i]) for k, v in coeffs.items()}

        pde: AbstractHyperbolicConservationLaw = pde_factory(**coeffs_i)

        if i == 0:
            well_writer = ZarrWellDataset(
                dataset_name=dataset_name,
                savepath=savepath,
                n_trajectories=n_coeffs * n_ics_per_coeff,
                codec=codec,
            )
            well_writer.initialize_with_pde(
                pde, t_span=t_span, Nt=Nt, x_spans=x_spans, Nxs=Nxs
            )

        for key_ic in key_ics:
            try:
                u, t, xs = pde.solve(
                    lambda u0: initial_condition_fn(u0, key_ic),
                    x_spans,  # TODO: check if it works for both 1D and 2D
                    Nxs,
                    t_span,
                    Nt,
                    bc=bc,
                    verbose=False,
                    **solve_kwargs,
                )
                assert not np.any(np.isnan(u))

                u_well_schema = pde.solution_to_well_schema(u)
                assert np.allclose(well_writer.root["dimensions"]["time"], t)
                # Implement n-D check for the spatial axes
                for x_i, x_i_name in zip(xs, ["x", "y", "z"]):  # Works up to 3D
                    assert np.allclose(well_writer.root["dimensions"][x_i_name], x_i)
                well_writer[write_idx] = u_well_schema

                write_idx += 1

            except Exception:  # TODO: Need to narrow to the Exception pyclaw raises
                logging.exception("solve failed for coeffs=%s", coeffs)
                continue

    well_writer.resize_dataset(write_idx)
    return well_writer

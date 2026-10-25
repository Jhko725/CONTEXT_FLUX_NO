"""Generate a Well-schema dataset for one PDE family (ADR-0004 / ADR-0011).

    uv run python scripts/data/generate_dataset.py pde=cubic_1d initial_condition=grf
    uv run python scripts/data/generate_dataset.py pde=cubic_1d backend=sharpclaw dataset_type=test
    uv run python scripts/data/generate_dataset.py pde=sine_1d n_coeffs=2 n_ics_per_coeff=2 pde.Nxs=[32]

Writes `<savedir>/<pde.name>/data/<dataset_type>/<savename>` plus `<savename>.dataset.json`
(the `DatasetRecord`). `backend=default` keeps each PDE class's own default solver (Classic
PyClaw for the PyClaw families), which is what the ICLR datasets used.
"""

import hydra
import jax
from context_flux_no.simulations import generate_dataset
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    solve_kwargs = dict(cfg.pde.get("solve_kwargs") or {})
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])

    backend = None
    if not cfg.backend.get("null_backend", False):
        backend = hydra.utils.instantiate(cfg.backend)

    generate_dataset(
        dataset_name=cfg.pde.name,
        n_coeffs=cfg.n_coeffs,
        n_ics_per_coeff=cfg.n_ics_per_coeff,
        pde_factory=hydra.utils.get_method(cfg.pde.pde_factory),
        initial_condition=hydra.utils.instantiate(cfg.initial_condition.sampler),
        coeff_range_dict=cfg.pde.coeff_range_dict,
        x_spans=cfg.pde.x_spans,
        Nxs=cfg.pde.Nxs,
        t_span=tuple(cfg.pde.t_span),
        Nt=cfg.pde.Nt,
        bc=OmegaConf.to_container(cfg.bc) if not isinstance(cfg.bc, str) else cfg.bc,
        dataset_type=cfg.dataset_type,
        coeff_sampling_strategy=cfg.coeff_sampling_strategy,
        seed=cfg.seed,
        savedir=cfg.savedir,
        filename=cfg.savename,
        backend=backend,
        primitive_ic=bool(cfg.pde.get("primitive_ic", False)),
        ic_description={
            "name": cfg.initial_condition.name,
            "config": OmegaConf.to_container(
                cfg.initial_condition.sampler, resolve=True
            ),
        },
        **solve_kwargs,
    )


if __name__ == "__main__":
    main()

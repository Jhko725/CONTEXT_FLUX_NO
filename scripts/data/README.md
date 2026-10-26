# Data generation scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

## Families, initial conditions, backends

- `pde=` one of `cubic_1d`, `sine_1d`, `burgers_1d` (viscous, own JAX stepper), `shallow_water_1d`
  (use `initial_condition=grf_shallow`/`step_shallow`), `cubic_2d` (own JAX stepper), `euler_2d`
  (`initial_condition=grf_euler_2d`/`step_euler_2d`).
- `initial_condition=` `grf` or `step` (and the family-specific variants above).
- `backend=` `default` (each family's own default: PyClaw Classic for the PyClaw families — what
  the ICLR datasets used), `classic`, or `sharpclaw` (WENO5 + SSP104; the higher-order reference).
  The JAX-stepped families accept only `default`.
- `bc=` `periodic` (default) or per-axis pairs, e.g. `bc=[[extrap,wall]]`.

Every store gets a sidecar `<name>.dataset.json` (`DatasetRecord`: grid, parameters, initial
condition, backend, seed, git SHA, number written/failed); see `docs/design/simulation-backends.md`.

## Example usage
To generate training data with default settings (Gaussian random field initial conditions and cubic flux) for multiple seeds using slurm job arrays (sbatch),
```bash
uv run python ./scripts/data/generate_dataset.py seed=0,1,2,3,4,5,6,7,8,9 --multirun
```

To generate test data for cubic flux and step initial conditions in the local terminal (recommand using tmux so that the process is not canceled when the terminal is exited),
```bash
uv run python ./scripts/data/generate_dataset.py initial_condition=step dataset_type=test seed=10
```

Similarly, for test data with sine flux and step initial conditions in the local terminal,
```bash
uv run python ./scripts/data/generate_dataset.py initial_condition=step pde=sine_1d dataset_type=test seed=12
```
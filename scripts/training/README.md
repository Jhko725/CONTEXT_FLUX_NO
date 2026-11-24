# Model training scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

`scripts/train.py` trains every model (HyperFluxFNO, HyperNO, DPOT, DISCO) on every dataset. The `source:` node of the data config (`configs/data/`) names the source class and its keyword arguments: `ZarrWellDatasetSource` for the Zarr stores (`cubic_1d`, `euler_2d`) and `TheWellDataSource` for The Well's HDF5 files (`well_euler_2d`). Train on `well_euler_2d` with `training=the_well`, which adds gradient clipping to the optimizer. Run the script from the project top level directory:

(Running in the current terminal)
```bash
uv run python ./scripts/train.py model=hyperfluxno
```
Note that the `--multirun` flag is need to submit the training as a slurm job.
(Submitting as a slurm job)
```bash
uv run python ./scripts/train.py model=hyperfluxno --multirun
```
If submitting to a particular partition, or modifying any other sbatch arguments, use the hydra.launcher keyword:

(Submitting to a partition named h100)
```bash
uv run python ./scripts/train.py model=hyperfluxno hydra.launcher.partition=h100 --multirun
```
Supported values for `model` are the file names under `scripts/training/configs/model/` (e.g. `hyperfluxno`, `hyperfluxno_2d_euler_v2`, `dpot`, `disco`); `tests/unit/test_configs.py` lists which model × data pairings instantiate.

Additional parameters can be passed to alter the model configuration. Alternatively, one can change the yaml files in scripts/training/configs to change default configuration values.

## Where a run goes

Every run lives in Hydra's output directory — `runs/<date>/<time>/` (a `--multirun` sweep:
`runs/multirun/<date>/<time>/<job>/`) — holding `config.yaml` (the resolved config),
`metadata.json` (the run's identity, provenance and outcome), `checkpoints/` (whole training
state; the best by validation loss and the final step are kept) and Hydra's `.hydra/`. See
`docs/design/run-identity.md`.

wandb is **off by default**. `wandb=online` (or `offline`) creates a wandb run that mirrors the
run directory: its config is only the dozen index columns (model, data, seed, loss, …), the
full config is attached as a file, and the wandb id/url are written back into `metadata.json`.
Set the entity/project in `configs/wandb/online.yaml`; `experiment=<name>` groups runs.

```bash
uv run python ./scripts/train.py model=hyperfluxno wandb=online experiment=ablation-A
uv run python ./scripts/train.py resume=runs/2026-10-24/14-03-12       # continue in place
uv run python ./scripts/train.py warm_start=runs/2026-10-24/14-03-12 model.key.seed=1
```

To list and load trained runs from Python: `from context_flux_no import runs` —
`runs.index("runs")`, `runs.select(...)`, `runs.load_run(run_dir)`. Checkpoints from before
this layout (`checkpoints/<data>/<Model>/...`) are indexed after
`uv run python scripts/backfill_run_records.py --root ./checkpoints`.

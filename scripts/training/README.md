# Model training scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

To train multiphysics neural operator models (HyperFluxFNO, DPOT, DISCO), run the following script from the project top level directory:

(Running in the current terminal)
```bash
uv run python ./scripts/training/train_multiphysics.py model=hyperfluxno
```
Note that the `--multirun` flag is need to submit the training as a slurm job.
(Submitting as a slurm job)
```bash
uv run python ./scripts/training/train_multiphysics.py model=hyperfluxno --multirun
```
If submitting to a particular partition, or modifying any other sbatch arguments, use the hydra.launcher keyword:

(Submitting to a partition named h100)
```bash
uv run python ./scripts/training/train_multiphysics.py model=hyperfluxno hydra.launcher.partition=h100 --multirun
```
Supported values for `model` are the file names under `scripts/training/configs/model/` (e.g. `hyperfluxno`, `hyperfluxno_2d_euler_v2`, `dpot`, `disco`); run `uv run python scripts/check_configs.py` to see which model × data pairings instantiate.

Additional parameters can be passed to alter the model configuration. Alternatively, one can change the yaml files in scripts/training/configs to change default configuration values.

Once training starts, progress will be logged in wandb, corresponding to the wandb entity and project specified in /configs/config.yaml. 
Make sure the wandb entity value is appropriately set to the user's value.

The resulting checkpoint (corresponding to the best training loss) will be saved at /checkpoints directory.

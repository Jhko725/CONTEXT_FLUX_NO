# context_flux_no — HFluxNO

In-context learning of finite-volume solvers for *families* of hyperbolic conservation laws.
A context encoder reads a short observed trajectory, a conditioning mechanism (hypernetwork)
specialises a flux network to the unknown coefficients, and a finite-volume update rolls the
solution forward. Families: cubic, sine and Burgers scalar laws, shallow water, 2D Euler.
Baselines: DISCO, DPOT, FNO variants. Paper: submitted to ICLR 2027 (tag `iclr2027-submission`).

**The repository is being restructured** (Oct 2026). The plan, the current state of the code and
every decision are under [`docs/`](docs/index.md) — start with `docs/index.md`. Conventions for
working here (including for Claude) are in [`CLAUDE.md`](CLAUDE.md).

## Install

```bash
uv sync --group dev                 # CPU JAX; add --extra cuda on a GPU machine
uv run nbstripout --install         # once per clone: notebook outputs never reach git
git submodule update --init         # third_party/icon (baseline; not imported by src)
uv run pre-commit install
```

## The three commands

```bash
# generate a dataset (Hydra; see scripts/data/README.md)
uv run python scripts/data/generate_dataset.py pde=cubic_1d initial_condition=grf seed=0

# train (Hydra; see scripts/training/README.md); --multirun submits to slurm via submitit
uv run python scripts/training/train_multiphysics.py model=hyperfluxno_2d_euler_v2 data=euler_2d

# evaluate: currently notebooks under notebooks/evaluation/; scripts/evaluate.py arrives in Phase 6
```

Checkpoints are written under `./checkpoints/<data>/<Model>/<Loss>/seed=<n>/<timestamp>/`
(orbax); runs are logged to wandb (`configs/config.yaml: wandb_kwargs`). Datasets live under
`./data/datasets/` in The Well's zarr schema.

## Tests

| Tier | Command | Needs |
|---|---|---|
| unit + property (fast; CI) | `uv run pytest` | — |
| regression (ICLR goldens) | `JAX_PLATFORMS=cpu uv run pytest tests/regression -m regression --checkpoints-root ./checkpoints` | checkpoints |
| verification (numerics) | `uv run pytest -m slow` | x64; slow |

`uv run python scripts/check_docs.py` checks the documentation bookkeeping; `uv run python
scripts/check_configs.py` instantiates every model × data config pairing (Phase 0.6e).

## Layout

```
src/context_flux_no/   the library (models, nn, training, data, simulations, waveforms)
scripts/               Hydra entry points + configs; check_docs.py
notebooks/             data generation, evaluation, prototypes (outputs stripped)
tests/                 unit · property · regression · verification
docs/                  roadmap, architecture, decisions (ADRs), design notes, change documents
third_party/icon       ICON baseline (submodule)
```

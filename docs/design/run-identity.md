---
type: design
status: agreed
updated: 2026-10-24
verified_by: joon (Phase 5 ledger, 2026-10-24)
sources: ADR-0010; code on phase5-run-identity
---

# Run identity — layout, records, how to find and load a run

The decision is ADR-0010; this is the working reference. Everything here is **[code]** on
`phase5-run-identity` unless marked.

## The run directory

```
runs/2026-10-24/14-03-12/              ← hydra.run.dir  (sweep: runs/multirun/<date>/<time>/<job>/)
├── .hydra/                            Hydra's own record (config, overrides, hydra.yaml)
├── config.yaml                        the resolved config; rewritten on an in-place resume
├── metadata.json                      RunRecord (below)
├── checkpoints/                       OrbaxCheckpointer: whole TrainerState per step dir
│   ├── 100/ 200/ … 50000/             LatestN(1) + BestN(1 by valid_loss); final step forced
│   └── _CHECKPOINT_METADATA           custom_metadata = {run_record, model_config}
└── wandb/                             only with wandb=offline|online
```

`runs.Run` is the object over this directory: `Run.from_config(cfg, dir, dataset_id=…)`
creates it (refusing an existing run unless `resume=True`), `Run.open(dir)` reads it,
`run.checkpointer(save_every=…)` gives the trainer its `OrbaxCheckpointer`, `run.restore(template,
step)` the `TrainerState`, `run.finish(status=…)` closes the record with final/best step.

## `RunRecord` (`metadata.json`)

| Field | Meaning |
|---|---|
| `run_id` | `YYYYMMDD-HHMMSS-<8 hex>`, generated locally at creation; legacy runs `legacy-<timestamp>` |
| `created_at`, `status` | ISO UTC; `running` · `finished` · `failed` · `legacy` |
| `index` | the columns of the run table (below) |
| `model_config` | `cfg.model` resolved — what `hydra.utils.instantiate` needs to rebuild the architecture |
| `git` `{sha, dirty}`, `package_version`, `x64` | provenance of the code that trained it |
| `wandb` `{id, url, entity, project, mode}` | the mirror, if any |
| `metric_name`, `best_step`, `best_metric`, `final_step` | outcome (filled by `Run.finish`) |
| `resumed` `{from_step, config_changes, previous_resumes}` | in-place continuation history |
| `resumed_from` `{run_dir, run_id, step}` | warm start from another run |
| `extra` | anything else; unknown fields of a newer schema land in `extra._unknown` |
| `schema_version` | 1 |

**Index columns** (`run_record.INDEX_COLUMNS`): `model` (class), `encoder`, `target_network`,
`conditioning` (implied by the class until Phase 3 makes it a config field), `data`,
`dataset_id`, `seed`, `loss`, `context_length`, `batch_size`, `peak_lr`, `max_steps`. These
and only these go to `wandb.config`. Adding a column = one line in `INDEX_COLUMNS` and one in
`index_from_config`.

`dataset_id` is `WellDatasetSourceBase.dataset_id`: `<dataset>:<split>:<8 hex of the common
metadata + window parameters>`, stable across machines (no paths in the hash). It is
provisional until Phase 4's `DatasetRecord.id` [deduced: the hash covers what today
distinguishes two data versions; a regenerated dataset with identical metadata but different
samples would collide — `DatasetRecord` will carry a content hash].

## Finding and loading runs

```python
from context_flux_no import runs

table = runs.index("runs")                       # pandas.DataFrame, one row per metadata.json
best = runs.select(table, model="HyperFluxFNOLocal", data="cubicflux_1d", status="finished")
best = best.sort_values("best_metric").iloc[0]

model, record = runs.load_run(best.run_dir)      # trained model at the latest step
model, record = runs.load_run(best.run_dir, step=record.best_step)
state, run = runs.load_state(best.run_dir)       # the whole TrainerState, for analysis of opt state
table_wb = runs.from_wandb("jhko725", "hyperfluxfno_revised")   # the same columns from wandb
```

`load_run` rebuilds the model from `config.yaml` (`cfg.model`) and the `TrainerState`
template from `cfg.training.optimizer`; a run whose optimizer was built in code rather than
in the config cannot be restored this way, which is why `train_the_well.py`'s clipping chain
moved into `configs/training/the_well.yaml`.

Legacy ICLR checkpoints: run `scripts/backfill_run_records.py --root ./checkpoints --audit
docs/bias_hyperinit_audit.csv` once; they then appear in `runs.index("checkpoints")` with
`status == "legacy"` and `load_run` loads them through `training.io.load_model`.

## Training, resuming, warm-starting

```bash
uv run python scripts/training/train_multiphysics.py model=hyperfluxno data=cubic_1d          # runs/<date>/<time>, wandb off
uv run python scripts/training/train_multiphysics.py ... wandb=online experiment=ablation-A    # wandb group = ablation-A
uv run python scripts/training/train_multiphysics.py resume=runs/2026-10-24/14-03-12           # continue in place
uv run python scripts/training/train_multiphysics.py warm_start=runs/2026-10-24/14-03-12 model.key.seed=1
uv run python scripts/training/train_multiphysics.py -m model=hyperfluxno,disco model.key.seed=0,1,2   # sweep → one group
```

`experiment.train(cfg, model=…, loss_fn=…, loaders…, dataset_id=…)` is the shared
assembly; the scripts only build the data source and loaders. Cadence keys in
`configs/training/*.yaml`: `eval_every` (one-batch validation; 1 as in the ICLR loop),
`save_every` (checkpoint offer; 100; the final step is always saved), `log_every` (wandb).

**Resume caveat** (ADR-0010 §5): the data stream restarts on resume; only the `TrainerState`
is exact. Exact continuation through grain's iterator checkpoint is an open roadmap item.

## What the trainer sees

```
Trainer(optimizer, loss_fn, loss_args).train(model | state, loader,
        num_steps, key, logger, checkpointer, evaluate, eval_every)
   │  per step:  state, metrics = train_step(state, batch);  logger.log(metrics, step)
   │  every eval_every:  val = evaluate(state);  logger.log(val);  checkpointer.save(step, state, val)
   └  end:  final state evaluated + saved with force=True (unless the last step already did)
```

Loggers: `PrintLogger` · `WandbLogger(run)` · `ListLogger` (tests) · `DelayedLogger(inner)`
(one-step delay so `float()` never waits on the step just enqueued) · `MultiLogger`.
Non-finite validation raises `FloatingPointError` *before* checkpointing, so the last
checkpoint on disk is finite; `experiment.train` then marks the record `failed`.

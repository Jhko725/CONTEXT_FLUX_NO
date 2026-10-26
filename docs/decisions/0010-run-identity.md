---
type: decision
id: ADR-0010
status: accepted
updated: 2026-10-25
verified_by: joon (ledger confirmed 2026-10-24, "Looks good. Go ahead")
sources: deep_isochron training/checkpoint.py, loggers.py, experiment/run.py, ADR-0011 §6; orbax-checkpoint 0.11 v1 API docs; JAX training cookbook (RecordWriter)
---

# ADR-0010 — A run is a directory; the trainer has no side effects

**Status**: accepted (2026-10-24). Implemented on `phase5-run-identity`.

## Context

The ICLR trainer (`training/trainer.py` at `iclr2027-submission`) created its own wandb run
and its own orbax directory (`checkpoints/<data>/<Model>/<Loss>/seed=<n>/<timestamp>/`) and
nothing tied the two together: the checkpoint folder was named by a wall-clock timestamp, the
wandb run by wandb, and the evaluation notebooks kept hand-maintained `CHECKPOINT_DICT`s of
timestamps (pain points P9, P10; finding F7). wandb's `config` was the whole resolved Hydra
config, so its run table had hundreds of columns and was unusable for the question "which
runs do I have, and which is best?" (P10). Checkpoints held model weights only, so a run
could not be continued. The same loop also mixed the training step with logging and
checkpointing, so it could not be tested without wandb.

`deep_isochron` (Joon's other project) had already solved this: injected `Logger` /
`Checkpointer` protocols, whole-`TrainerState` checkpoints, a `Run` directory object with
explicit resume semantics. Joon asked to study it and adopt what fits.

## Decision

1. **The run directory is the identity** [code]. A run is `<run_dir>/{config.yaml,
   metadata.json, checkpoints/}` and `<run_dir>` is Hydra's output directory
   (`runs/<date>/<time>`, `runs/multirun/<date>/<time>/<job>` under a sweep), so Hydra's
   `.hydra/`, the wandb files (`dir=run_dir`) and the checkpoints live together and nothing
   has to be joined afterwards. `metadata.json` is a `RunRecord`: a **locally generated**
   `run_id` (UTC stamp + uuid4 suffix, so a run has an identity with wandb off), git SHA and
   dirty flag, package version, `x64`, `dataset_id`, the **index columns**, status, best /
   final step, and resume notes. The full config is *not* duplicated into the record;
   `config.yaml` is the latest truth.

2. **wandb is a mirror, never the source** [code]. `wandb.config` is `RunRecord.index` only
   (a dozen columns: model, encoder, target network, conditioning, data, dataset_id, seed,
   loss, context length, batch size, peak LR, max steps); the full config is uploaded as the
   file `config.yaml`. wandb's id/url are written back into `metadata.json`. `group` =
   `cfg.experiment` → sweep directory → dataset name; `job_type` = model class; `tags` = the
   string-valued index columns. `wandb=disabled` is the default config group.

3. **The trainer has no side effects** [code]. `Trainer(optimizer, loss_fn).train(...)` takes
   a `Logger` and a `Checkpointer` (protocols; `Null`, `List`, `Print`, `Wandb`, `Delayed`,
   `Multi` loggers; `Null` and `Orbax` checkpointers) and an `evaluate(state)` callable. The
   loop never reads a device value: metrics go to the logger as JAX scalars and
   `DelayedLogger` converts the *previous* step's, the cookbook `RecordWriter` pattern the old
   loop implemented by hand with `state_prev`.

4. **Whole `TrainerState` is checkpointed** [code] — model, optimizer state, step, PRNG key —
   under `LatestN(1) + BestN(1 by valid_loss)`; the final step is always saved (`force=True`),
   so `Run.restore()` with no step is the trained state whatever `save_every` is. Typed PRNG
   keys restore natively on orbax ≥ 0.12.8 (0.11 could not; the pin was raised in review round 1). The
   record and the model config are also written to orbax `custom_metadata`, so a
   `checkpoints/` directory alone identifies its run.

5. **Resume is explicit** [code]. `Run.from_config` **refuses** a directory that already
   holds a run; `cfg.resume=<run_dir>` continues *that* run in place (same `run_id`;
   `resumed = {from_step, config_changes}` recorded, `config.yaml` rewritten);
   `cfg.warm_start=<run_dir>` starts a *new* run from another run's latest state
   (`resumed_from` recorded). The optimizer must be fully described by
   `cfg.training.optimizer` (chains included) so `runs.state_template(cfg)` can rebuild the
   opt-state structure from `config.yaml` alone.

6. **Legacy checkpoints join the index** [code]. `scripts/backfill_run_records.py` writes a
   `status="legacy"` record (`run_id = legacy-<timestamp>`) into each ICLR timestamp
   directory, matching wandb runs through the F1 audit CSV; `runs.load_run` routes those to
   the legacy loader `training/io.py::load_model`, which stays until the regression tier no
   longer needs it.

## Rejected alternatives

- **Checkpoint folder named by the wandb run id** (the Phase 5 ledger's first draft). Makes
  the local artifact depend on a remote service and on `wandb=online`; a locally generated id
  recorded alongside wandb's gives the same join without the dependency. [deduced]
- **A local SQLite / Parquet database of runs** (Joon's own suggestion, "feels a bit like
  overkill"). `runs.index(root)` reads the `metadata.json` files directly — fast enough for
  hundreds of runs, no schema to migrate, `git`-free. Revisit if the count reaches thousands.
- **Model-only checkpoints** (as before). Cheaper on disk but cannot resume; the optimizer
  state of adamw is 2× the model, acceptable with `LatestN(1) + BestN(1)`.
- **Trainer owns wandb and orbax** (as before). Untestable without network and disk; the
  injected version runs the same loop in tests with `ListLogger` and a temporary
  `OrbaxCheckpointer`.
- **Exact data-stream continuation on resume.** grain's `.shuffle().repeat()` iterator state
  is not restored; a resumed run restores the state exactly and *restarts* the stream with a
  seed derived from the step. Honest version shipped; exact continuation via grain iterator
  checkpoints is an open item (roadmap Phase 5).

## Where the code lives (review round 1)

`training/` — building blocks (trainer, losses, loggers, checkpoint), importing nothing from the
other two; `runs/` — run identity (`record`, `run` + `RunLayout`, `index`);
`experiment.py` — config → objects (`build_model`, `build_optimizer`, `state_template`, the
wandb mirror `build_wandb_run`/`finish_wandb_run`, the shared script assembly; review round 2
moved the wandb functions here from `runs/wandb_mirror.py` since they are config-driven
assembly too). Dependency direction `training ← runs ← experiment`.

## Consequences

- One training run yields `metadata.json`, orbax `custom_metadata` and (if enabled) a wandb
  run that agree on `run_id`; `runs.index()` replaces notebook `CHECKPOINT_DICT`s (F7).
- `save_every` now bounds how fine "best by `valid_loss`" is (the ICLR loop saved weights
  every step and kept the best): the default is 100 with the final step forced. Set
  `training.save_every=1` to recover the old selection exactly, at 3× the write volume.
- Multi-device: `Trainer` carries an unused `mesh` argument and never reads device values,
  which is the one line of design Phase 5.5 asked for; sharding itself is parked.

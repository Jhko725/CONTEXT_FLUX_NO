---
type: change
status: landed
updated: 2026-10-24
branch: phase5-run-identity
roadmap: Phase 5
base: phase1-prune
---

# 2026-10-24 — `phase5-run-identity`

Phase 5 of `docs/roadmap.md` (P9, P10, F7): every training run gets an identity that its
checkpoints, its wandb run and the analysis code agree on, and the trainer becomes a pure loop
with injected logging and checkpointing. Executed by Claude on 2026-10-24 from the revised
ledger Joon confirmed after the `deep_isochron` study. Decision record: ADR-0010; working
reference: `docs/design/run-identity.md`.

**Base.** Branched from `phase1-prune` (not yet merged, gated on the regression tier), so
`scripts/check_docs.py --files --base phase1-prune` is the right check; against `main` the diff
also shows Phase 1's files. **Merge order**: `phase1-prune` first, then this.

**Merge gate.** Same as Phase 1: `tests/regression` green on the ICLR goldens. This branch does
not touch any model's forward pass; `training/io.py::load_model` (the loader the regression tier
uses) is unchanged.

## Summary

- `Trainer(optimizer, loss_fn)` loses wandb and orbax; `train()` takes a `Logger`, a
  `Checkpointer` and an `evaluate` callable, checkpoints the **whole `TrainerState`**, always saves
  the final step, and stops before checkpointing a non-finite validation.
- `training/loggers.py`, `training/checkpoint.py`: the protocols and implementations (ported from
  `deep_isochron`; `DelayedLogger` is the old `state_prev` trick made explicit).
- `training/run_record.py`, `training/run.py`: `RunRecord` (`metadata.json`) and the `Run`
  directory object in Hydra's output dir; explicit `resume` / `warm_start`.
- `training/runs.py`: `index(root) → DataFrame`, `select`, `load_run`, `load_state`, `from_wandb`.
- `training/wandb_run.py`: wandb as a mirror — `config` = index columns only, full config as a
  file, id/url written back; `wandb=disabled|offline|online` config group (disabled default).
- `training/experiment.py`: what the two scripts share; the scripts shrink to data + loaders.
- `WellDatasetSourceBase.dataset_id`; `provenance.git_state()`.
- `scripts/backfill_run_records.py`: legacy ICLR checkpoints get a `status="legacy"` record.
- 7 new unit-test files, 28 tests; `pandas` declared (was transitive).

## Files

Commit 1 — `6c67dbc` injected side effects (ledger 1–4, 12):
- `src/context_flux_no/provenance.py` — new: `git_state() -> (sha, dirty)`, `package_version()`.
- `src/context_flux_no/training/loggers.py` — new: `Logger` ABC (context manager), `NullLogger`, `ListLogger`, `PrintLogger`, `WandbLogger`, `DelayedLogger`, `MultiLogger`.
- `src/context_flux_no/training/checkpoint.py` — new: `Checkpointer` ABC, `NullCheckpointer`, `OrbaxCheckpointer` (`save(step, state, metrics, force)`, `restore(template, step)`, `steps`, `latest_step`, `best`, `custom_metadata`); typed-key unwrap/rewrap.
- `src/context_flux_no/training/trainer.py` — rewritten: `TrainerState` unchanged in shape; `Trainer.__init__(optimizer, loss_fn, *, loss_args, is_trainable, mesh)`, `init`, `train_step`/`valid_step`, `evaluator(loader)`, `train(...)` as described above.
- `src/context_flux_no/training/__init__.py` — exports for the above, later `Run`, `RunRecord`, `index_from_config`.
- `tests/unit/test_loggers.py`, `tests/unit/test_checkpoint.py`, `tests/unit/test_trainer.py` — new.

Commit 2 — `e792ecb` run identity (ledger 5–8):
- `src/context_flux_no/training/run_record.py` — new: `RunRecord`, `INDEX_COLUMNS`, `index_from_config`, `new_run_id`.
- `src/context_flux_no/training/run.py` — new: `Run` (`from_config`, `open`, `checkpointer`, `latest_step`, `restore`, `update`, `finish`), `config_changes`.
- `src/context_flux_no/training/runs.py` — new: `iter_runs`, `index`, `select`, `build_model`, `build_optimizer`, `state_template`, `load_state`, `load_run`, `from_wandb`.
- `src/context_flux_no/data/sources.py` — `WellDatasetSourceBase.dataset_id` cached property (only addition).
- `pyproject.toml`, `uv.lock` — `pandas>=2.2` declared.
- `tests/unit/test_run_record.py` (Hypothesis JSON round trip), `tests/unit/test_run.py` — new.

Commit 3 — `2b68e01` wandb, Hydra run dir, scripts, backfill (ledger 9–11):
- `scripts/training/configs/config.yaml` — `wandb: disabled` default group; `experiment`, `resume`, `warm_start` keys; `hydra.run.dir` / `hydra.sweep.dir` under `runs/`; `wandb_kwargs` removed.
- `scripts/training/configs/wandb/{disabled,offline,online}.yaml` — new (three files; committed in the docs commit because the old `wandb/` ignore pattern swallowed them).
- `.gitignore` — `wandb/` narrowed to wandb's run files (`**/wandb/run-*/` etc., `/wandb/`); `/runs/` added.
- `scripts/training/configs/training/default.yaml` — `checkpoint_dir` → `eval_every` / `save_every` / `log_every`.
- `scripts/training/configs/training/the_well.yaml`, `scripts/training/configs/config_the_well.yaml` — new: the clipping chain as `optax.chain` in the config; entry config for `train_the_well.py`.
- `src/context_flux_no/training/wandb_run.py` — new: `wandb_group`, `build_wandb_run`, `finish_wandb_run`.
- `src/context_flux_no/training/experiment.py` — new: `select_device`, `segment_length`, `run_directory`, `start_run`, `training_key`, `initial_state`, `train`.
- `scripts/training/train_multiphysics.py`, `scripts/training/train_the_well.py` — rewritten on `experiment.train`; only data source + loaders remain.
- `scripts/backfill_run_records.py` — new.
- `tests/unit/test_experiment.py`, `tests/unit/test_backfill_run_records.py` — new.

Commit 4 — docs (this commit):
- `docs/decisions/0010-run-identity.md`, `docs/design/run-identity.md` — new.
- `docs/changes/2026-10-24-phase5-run-identity.md` — this document.
- `docs/roadmap.md`, `docs/architecture.md`, `docs/glossary.md`, `docs/index.md`, `docs/findings.md` — Phase 5 rows, Done ledger, module tree, terms, index rows, F7 status.
- `scripts/training/README.md` — run directory, wandb group, resume.

## Design

Each with the alternative rejected; ADR-0010 has the full argument.

- **Run directory = Hydra output dir** rather than `checkpoints/<data>/<Model>/…/<wandb id>`: no dependency on wandb for the local artefact, and `.hydra/` + checkpoints + wandb files sit together. The index columns carry what the old path encoded.
- **Locally generated `run_id`, wandb's recorded alongside** rather than wandb's id as the identity (first ledger draft). Works with `wandb=disabled`.
- **Whole `TrainerState` checkpoints, `LatestN(1)+BestN(1)`, final forced** rather than weights only: resume becomes possible; disk cost ≈ 3× weights for two checkpoints. Legacy loader kept for the regression tier.
- **Injected `Logger` / `Checkpointer`** rather than the trainer owning wandb/orbax: the loop is testable (28 tests, CPU, under a minute) and a notebook can run it with `ListLogger`.
- **`DelayedLogger`** rather than the hand-rolled `state_prev`/`outputs_prev` pair: same effect (never `float()` the step just enqueued), one place.
- **`eval_every=1, save_every=100`** defaults rather than saving every step as the ICLR loop did: write volume; the final step is always saved and the record names the best *saved* step. `save_every=1` recovers the old selection exactly. See Open issues.
- **Optimizer fully in the config** (the_well's clipping chain moved from code to `configs/training/the_well.yaml`): `runs.state_template(cfg)` must rebuild the opt-state structure from `config.yaml` alone, so anything built in code would not be restorable.
- **`index_from_config` tolerates missing keys** and looks inside `optax.chain` `_args_` for `learning_rate`: legacy configs and the_well chain both index.
- **`Run.finish` opens its own checkpointer view** rather than reusing the trainer's: the trainer closes its checkpointer on exit (context manager), and `finish` runs after.
- **Resume semantics ported as is** (refuse-by-default, `resume=` in place, `warm_start=` new run) with the data-stream caveat shipped honestly rather than a half-exact continuation.
- **`experiment.py` lives in the package**, not `scripts/training/common.py`: Hydra's submitit launcher pickles `main` by module reference, and a sibling module of the script is not importable in the Slurm job.
- **wandb config group named `disabled`**, not `off`: `- wandb: off` in a defaults list is YAML `false`.

## Bugs fixed

- `train_the_well.py` built its optimizer with a clipping chain the saved config did not mention, so the recorded config could not have rebuilt the run. Moved into `configs/training/the_well.yaml`.

## Findings

- **F7 (checkpoint ↔ run link)**: addressed for new runs; legacy runs join via the backfill with wandb candidates from the F1 audit CSV, unique matches recorded as `wandb`, ambiguous ones in `extra.wandb_candidates` for Joon to resolve by hand. Appended to `docs/findings.md`.
- **Save cadence changes the "best checkpoint" semantics** (see Design). Not a reported-result concern — the ICLR checkpoints are untouched — but a protocol change to state in the paper's training section if any post-submission number uses the new scripts. Appended as F10.

## Tests

All CPU, fast tier (`JAX_PLATFORMS=cpu uv run pytest tests/unit`, ~70 s):

- `test_loggers.py` — every logger's contract; `DelayedLogger` one-step delay and flush on close; `MultiLogger` fan-out.
- `test_checkpoint.py` — save/restore round trip of a `TrainerState` incl. typed PRNG key; `save_every` and `force`; `LatestN+BestN` preservation; `custom_metadata`; `best`.
- `test_trainer.py` — logs at eval steps, forced final save under `save_every=100`; same seed ⇒ same state; resume from a state continues the step; loader-ends-early warning; non-finite validation raises before checkpointing.
- `test_run_record.py` — Hypothesis JSON round trip over scalar-valued index/model configs; unknown fields kept in `extra._unknown`; `index_from_config` on the real config layout and on partial configs; `row()` is flat JSON.
- `test_run.py` — layout written, reuse refused; train → `finish` → `load_run` returns the same weights; resume in place keeps `run_id`, records `from_step` and `config_changes`, continues the step; `index`/`select`; `config_changes` leaves only.
- `test_experiment.py` — `experiment.train` under a faked `HydraConfig`: run in the output dir, `resume=` and `warm_start=` (`resumed_from` names the source run), `failed` status on `FloatingPointError`.
- `test_backfill_run_records.py` — synthetic ICLR-layout checkpoint → `metadata.json` with audit match → `runs.index` lists it → `load_run` loads it through `io.load_model`.

Not tested here: a real `wandb.init` (network); `scripts/check_configs.py` still reports the same 14 dimension-mismatch pairings as before this branch.

## Open issues

- **Exact data-stream continuation on resume** — grain iterator state is not checkpointed; the stream restarts. Follow-up item on the roadmap (5.6).
- **`save_every` vs best-checkpoint fidelity** — default 100; Joon to decide whether the new scripts should save every step (`save_every=1`) to match the ICLR protocol.
- **`conditioning` index column is implied by the model class** until Phase 3 makes it a config field.
- **`dataset_id` is provisional** until Phase 4's `DatasetRecord` (content hash).
- **`training/io.py::load_model` stays** as the legacy loader for the regression tier and backfilled runs; retire with the ICLR shim (D2).
- **wandb `resume="allow"` on an in-place resume** reuses the recorded wandb id; untested against the service.
- **`check_docs.py --files`** defaults to `--base origin/main`; for this branch use `--base phase1-prune` (documented above) — a `base:` frontmatter key that the script reads would remove the footgun; left for a docs-tooling pass.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/context_flux_no/provenance.py` — `git_state`, `package_version` | Looks good. | None |
| `src/context_flux_no/training/loggers.py` — Logger protocol and six implementations | Looks good. `block` could easily be generalized into `tree_block_until_ready` via `jax.tree.map`. Would be need such generalization in the codebase? | To be discussed with Claude |
| `src/context_flux_no/training/checkpoint.py` — Checkpointer protocol, `OrbaxCheckpointer` (whole state, typed keys, `best`) | Currently random key array is being handled via additional logic involving `_unwrap_key` and `_rewrap_key`. Modern version of `orbax.checkpoint` should natively be able to handle this. How about upgrading the checkpoint version dependency? Alternatively, can one register custom PyTree nodes to getaround this? A third option is not to have `PRNGKeyArray` in the pytrees we are saving. Where are the `keys` currently? Could change to a integer seed value instead and wrap with `jax.random.key(seed)`, no? | To be discussed with Claude. |
| `src/context_flux_no/training/trainer.py` — pure loop with injected logger/checkpointer/evaluate; forced final save; non-finite stop | Looks good for now. Will revisit in the future (after the new dataset generation) to optimize trainer performance. Having an extra evaluator interface around validation dataloader also seems a bit bloated, but will be deferring for now. In the future, this may make more sense with the need for additional evalutions during training. | None |
| `src/context_flux_no/training/__init__.py` — exports | Trivial changes. | None |
| `src/context_flux_no/training/run_record.py` — `RunRecord`, `INDEX_COLUMNS`, `index_from_config` | Looks good for now. | None |
| `src/context_flux_no/training/run.py` — `Run` directory object, resume semantics, `config_changes` | Currently, the files 'Run' point to are determined by the module level constants `CONFIG_FILE`, `METADATA_FILE`, and `CHECKPOINT_DIR`. Seems better to make these default constructor arguments (and instance attributes) to avoid rigid hardcoded behavior. | To be discussed with Claude. |
| `src/context_flux_no/training/runs.py` — `index`/`select`/`load_run`/`load_state`/`from_wandb` | Contents look okay for now. The problem is that haveing `run.py`, `runs.py` and `run_record.py` is very confusing. Need better names and/or a `training/runs` (or a `src/runs`) (sub)module. | To be discussed with Claude.|
| `src/context_flux_no/data/sources.py` — `dataset_id` | Looks good. | None |
| `src/context_flux_no/training/wandb_run.py` — wandb as mirror | Likewise, will need to be grouped with the other `run` related functionalities. | To be discussed with Claude. |
| `src/context_flux_no/training/experiment.py` — shared script assembly, `resume`/`warm_start` | Look good, but seems like that `training` has two types of files - those representing the building blocks to be used in training, and helper functionalities that instantiate these objects from config. Feels like `training` should be split into two to deflect this dichotomy. | To be discussed with Claude. |
| `scripts/training/train_multiphysics.py`, `scripts/training/train_the_well.py` — on the new API | Scripts look lean, which is good. Don't like renaming `experiments` into `common` | Deferred to Claude for change. |
| `scripts/training/configs/config.yaml`, `configs/wandb/*.yaml`, `configs/training/{default,the_well}.yaml`, `configs/config_the_well.yaml` — run dir, wandb group, cadence, optimizer chain | Looks good. What is maant be cadence? | None |
| `scripts/backfill_run_records.py` — legacy records | Had a lookthrough. Will need to run once cluster comes back to properly confirm. | None for now. |
| `.gitignore` — wandb pattern narrowed, `/runs/` added | Confirmed. | None |
| `pyproject.toml`, `uv.lock` — pandas | Confirmed. | None |
| `tests/unit/test_loggers.py`, `test_checkpoint.py`, `test_trainer.py` | Looks good. | None |
| `tests/unit/test_run_record.py`, `test_run.py`, `test_experiment.py`, `test_backfill_run_records.py` | Had a lookthrough. Will scrutinize later. | None for now. |
| `docs/decisions/0010-run-identity.md`, `docs/design/run-identity.md` | Looks good. | None |
| `docs/roadmap.md`, `docs/architecture.md`, `docs/glossary.md`, `docs/index.md`, `docs/findings.md`, `scripts/training/README.md` | Confirmed | None |

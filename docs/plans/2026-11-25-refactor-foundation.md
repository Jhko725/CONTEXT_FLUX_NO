---
type: plan
status: executed
updated: 2026-11-25
sources:
  - "Project document claude/refactor-foundation-handoff-2026-11-25.md"
---

# Handoff: `refactor-foundation`

## 1. Header

**Goal.** Give `context_flux_no` the working conventions of `UniversalForceReconstruction`, remove everything that existed to keep the ICLR checkpoints loadable, move the code that awaits its rewrite under `legacy/`, and replace three existing pieces with clean ones: the training entry point, the data loader, and the evaluation code.

Implement with `orchestrate-implementation`.

- **Date:** 2026-11-25.
- **Design settled by:** Joon, in the planning session of 2026-11-25 (decisions 1 to 5, questions Q1 to Q20).
- **Repository:** `Jhko725/CONTEXT_FLUX_NO`.
- **Branch:** `refactor-foundation`, created from `origin/workflow-bootstrap` at `d60a19a`, which is `main` at `450f31d` plus one commit that adds `.claude/` and touches `CLAUDE.md` and `docs/index.md`. `workflow-bootstrap` is not merged separately; this branch carries it.
- **Line numbers** refer to `450f31d`. Confirm each by grep before editing.
- **Reference repositories,** read-only:
  - `Jhko725/UniversalForceReconstruction` at `b80a341` (private; called *UFR* below). Source of the workflow files and the checks.
  - `xx257xx/CONTEXT_FLUX_NO`, branch `joon`, at `4fe3323` (public; called *the rebuttal branch* below). Source of the evaluation code.
- **Earlier documents:** `docs/roadmap.md` at `450f31d`; Project documents `claude/Codebase cleanup plan - v3.md` and the ICLR 2027 rebuttal plans.
- **The run is unattended.** Section 9 gives the conditions that stop it.
- A reason that this document does not give is a gap in this document. Do not fill it silently; follow section 9.

## 2. Out of scope and assumptions

### Out of scope

- The finite-volume core and every numerical scheme (`fv/`, roadmap Phases 2b to 2f, WENO5). Joon writes these himself.
- The models. Nothing under `legacy/` changes except import paths (Q12). No model is rewritten, and the operator hierarchy of ADR-0006 is not started.
- The question-specific half of the rebuttal's `evaluation.py`: `rollout_with_solver`, `rollout_with_refresh`, `evaluate_refresh_rollouts`, `evaluate_context_interventions`, `effective_flux`, `evaluate_effective_flux`, `evaluate_time_steps`, `max_cubic_wave_speed`, `equal_count_bins`, `cubic_wave_speed_distance`, `state_coverage`, `coverage_bins`, the `cubic_*` and `sine_*` flux functions, `best_cubic_in_box`, and the scripts that use them (Q1). They stay on the rebuttal branch.
- `AbstractGeneratedSolverOperator` (`generate_solver`, `step`). It is recorded in ADR-0006 as a requirement on Phase 3 and not brought across as code (Q2).
- The rebuttal's data generators (`generate_unseen_flux_1d.py`, `generate_shallow_water_step_1d.py`, `godunov_1d.py`) and the corrected shallow-water coefficient range (Q18b). They belong with roadmap Phase 4.4.
- Exact resume of the data stream (roadmap 5.6), typed configs (3b), and dataset regeneration (4.4).
- Experiment reports and paper records. `docs/document-conventions.md` defines both types; no document of either type is written in this round.
- The eight merged change documents under `docs/changes/` dated before 2026-11-25. Their text does not change.
- Any run that needs a GPU, Julia, the ICLR checkpoints or the datasets. The session has none of them.

### Assumptions of the writer

None of these was put to Joon as a question. Each is marked `[unconfirmed]` where the implementing session records it.

- **W1. The workflow files come from `workflow-bootstrap`, not from `main`.** Joon could not merge that branch before the run, so `refactor-foundation` starts from it (revised 2026-11-25). An agent definition is read when a session starts, from the checkout the session starts in. If the session started on `main`, the `implementer` type it offers is absent, or is the one of `UniversalForceReconstruction`, whose boundary names that repository's archive. Section 4, step 3 says what the orchestrator does in each case.
- **W2. The split into units 1 to 18** and each unit's model.
- **W3. The config census becomes a test with declared pairings.** The 14 failures of `scripts/check_configs.py` at `450f31d` are not broken configs: 6 are the two base configs (`hyperfluxno_base`, `hyperno_base`), which are not selectable, and 8 are dimension-specific model configs paired with data of the other dimension (`dpot`, `fno_spatiotemporal` and `hyperfluxno_vit` are 1D; `dpot_2d_euler` and `dpot_2d_scalar` are 2D). Q9 said failing configs are deleted; none of these is deleted. The census skips `*_base` and pairs a dimension-specific config only with data of its dimension, from one table in the test.
- **W4. The garbage-collection policy lives in the training driver, not in the loader.** Q10 placed it in `make_loader`. A loader factory that changes interpreter state on import or on call would surprise the evaluation script, which also calls it. `data/loader.py` provides a context manager `gc_paused(every)`; `experiment.train` uses it.
- **W5. Existing ADRs are converted to the ADR template** of `document-conventions.md` by moving their text under the required headings (`## Rejected alternatives` becomes `## Alternatives rejected`; a `## History` section is added with one dated line per recorded amendment). No sentence is rewritten.
- **W6. The eight merged change documents are exempt from the required-headings check** through one named list in `check_docs.py`, `PRE_FORMAT_CHANGES`. The list never grows.
- **W7. `check_configs.py` and `import_all.py` become tests** under rule R10 (a check reads, a test executes), since both import the package.
- **W8. Evaluation reads a store's channels in the store's order and converts to the family's order through one function.** At `450f31d` the two orders differ for Euler: the family is `(rho, m_x, m_y, E)` (`simulations/pde/euler.py:36-40`) and a Well store packs rank-0 fields before rank-1 fields, giving `(rho, E, m_x, m_y)`. The orchestrator records this as finding F16.
- **W9. `evaluation/cost.py` keeps `count_flops` and `measure` only.** `parameter_breakdown` and `step_flop_breakdown` on the rebuttal branch name fields of `HyperFluxFNOLocal` and would make `evaluation/` import `legacy/`.
- **W10. The form of a clean list.** `[tool.context_flux_no] clean` in `pyproject.toml`, read with `tomllib` by the checks, the `justfile` and `tests/conftest.py`, as in UFR.

## 3. What the implementing session does

| # | Unit | Who |
|---|---|---|
| 0 | Access, environment, branch, baseline, save this document to disk, create the change document | Orchestrator, inline |
| 1 | `justfile`, `prek.toml`, checks under `scripts/checks/`, CI | `implementer`, `sonnet` |
| 2 | `docs/document-conventions.md`, `check_docs.py` and `check_md_math.py` ported, existing documents made to pass | `implementer`, `opus` |
| 3 | Change-document format and `check_changes.py`; `CLAUDE.md` | `implementer`, `opus` |
| 3b | Commit this document under `docs/plans/` | Orchestrator, inline |
| 4 | Remove the legacy loader, the regression tier and the ICLR scripts | `implementer`, `sonnet` |
| 5 | Remove the xarray path, the notebooks and the ICON submodule | `implementer`, `sonnet` |
| 6 | Two small fixes: the `operatornet.py` key, the step sampler's minimum | `implementer`, `sonnet` |
| 7 | `legacy/` and the clean list | `implementer`, `sonnet` |
| 8 | One training entry point; the census and the import sweep as tests | `implementer`, `opus` |
| 9 | The loader factory and `bench_loader.py` | `implementer`, `opus` |
| 10 | `evaluation/metrics.py` | `implementer`, `opus` |
| 11 | Family admissibility and the store-to-family channel order | `implementer`, `opus` |
| 12 | `evaluation/rollout.py` and `evaluation/summary.py` | `implementer`, `opus` |
| 13 | `evaluation/cost.py` and `scripts/evaluate.py` | `implementer`, `opus` |
| 14 | Docstrings and complexity on the clean list | `implementer`, `opus` |
| 15 | `ty` on the package's clean paths; the container, dependency and docstring checks | `implementer`, `opus` |
| 16 | `ty` on `tests/` and `scripts/`; the jaxtyping and beartype import hook | `implementer`, `opus` |
| 17 | One bibliography | `implementer`, `opus` |
| 18 | The documents | Orchestrator, inline |

## 4. Before anything

1. **Access.** Push access to `Jhko725/CONTEXT_FLUX_NO`. Read access to `Jhko725/UniversalForceReconstruction`: clone it at `b80a341` beside the working clone. Add the rebuttal branch as a remote: `git remote add upstream https://github.com/xx257xx/CONTEXT_FLUX_NO && git fetch --depth 200 upstream joon`, and confirm `git rev-parse upstream/joon` prints `4fe3323…`. If either reference is not at the named commit, read the named commit, not the tip.
2. **The tip.** `git fetch origin`, then confirm that `git rev-parse origin/main` prints `450f31d…` and `git rev-parse origin/workflow-bootstrap` prints `d60a19a…`. If either moved, list what changed and stop if any file this document names changed.
3. **The agent.** After step 5, `test -f .claude/agents/implementer.md` must succeed. Every brief carries the *Boundary*, *Stop rule*, *Before and after* and *Report* sections of that file verbatim, whichever agent type runs it, so that this repository's boundary applies even when the session's `implementer` type was loaded from another repository. If the session offers no `implementer` type, use a general-purpose subagent with the unit's model and the same brief, and add to the brief that it must not run `git push`, since the hook of the agent file does not run for that type. Record which case applied in the change document's *Open issues*.
4. **The environment.** A cloud session has 2 cores, about 7 GB of memory, no GPU and no Julia. `clawpack` builds from source and needs a Fortran compiler: `apt-get update -q && apt-get install -y -q gfortran`, then `uv sync --group dev`. Set `JAX_PLATFORMS=cpu` for every command.
5. **The branch.** `git checkout -b refactor-foundation origin/workflow-bootstrap`.
6. **This document.** Save the Project copy to `docs/plans/2026-11-25-refactor-foundation.md` on disk without committing it. Unit 3b commits it, once the `plan` type exists.
7. **The change document.** Create `docs/changes/2026-11-25-refactor-foundation.md` in the format of section 5 (rule T1), `status: in-progress`, with a `### Round 1: from` table whose subheading names the merge base with `origin/main` (`450f31d`). Give it one row for each file of the bootstrap commit (`git diff --name-only 450f31d d60a19a`), move the three *Design* points of `docs/changes/2026-11-25-workflow-bootstrap.md` into its *Design*, then delete that document and its row in `docs/index.md`: the bootstrap is part of this branch, and one branch has one change document. Commit this as the first commit of the branch. Every implementer adds its rows from unit 1 on.
8. **What to read first:** `docs/index.md`, `docs/architecture.md`, `docs/roadmap.md`, `docs/coding-standards.md`, and in UFR `CLAUDE.md`, `docs/coding-standards.md`, `docs/document-conventions.md`.
9. **Skills.** Subagents need `code-conventions` and `clear-prose`. A missing skill is skipped without an error, so each report states which were present.

## 5. Rule text

These rules are not yet in the repository. Units 2, 3 and 18 write them into `docs/coding-standards.md`, `docs/document-conventions.md` and `CLAUDE.md`. Until then the briefs carry them verbatim.

- **T1. Change documents.** The sections are *Summary*, *Design*, *Bugs fixed*, *Tests*, *Open issues*, *Review notes*, in that order, and there is no *Files* section. *Review notes* is a log of rounds, each under ``### Round <n>: from `<commit>` `` with a table `Change | Thoughts | Modifications`. A *Change* cell is one backticked path, a dash, and what changed, on one line; a path that ends in `/` covers every changed file under it. Every file a commit touches gets a row in the open round in that commit. A round is closed once one of its *Thoughts* cells is filled, and its rows are never edited again. Round 1 starts from the merge base with `origin/main`.
- **T2. Unconfirmed decisions.** A decision that Claude takes without Joon is marked `[unconfirmed]` where it is stated. `just pending` lists every marker. A document whose status is `agreed`, `accepted`, `amended` or `merged` carries none.
- **R7. The strict checks apply to listed clean paths.** The list is `clean` under `[tool.context_flux_no]` of `pyproject.toml`. Everything else in the package is under `src/context_flux_no/legacy/`, which stays importable and otherwise untouched until the branch that rewrites it. A module joins the clean list by leaving `legacy/` and never returns.
- **R10. A check reads; a test executes.** A check script reads text or syntax and never imports the package. A test executes the package.
- **R11. A test pins a property of our code that could be wrong.** It never pins the behavior of JAX, Equinox, grain or NumPy. A formula is not tested against a second copy of the same formula; it is tested by a property or against a value computed by hand.
- **R12. `evaluation/` imports nothing from `legacy/`.** A model reaches it through the `Protocol` of `evaluation/rollout.py`.
- **History.** A pushed commit is never amended or rebased, and a branch is never force-pushed. A change to a pushed commit is a new commit on top.
- **Review.** The review and the merge are Joon's. Claude does not propose skipping a review round or merging ahead of one.
- **Outcomes.** A test's outcome is stated only after that test has run on the code the document describes.

## 6. Units

Every unit ends with `uv run just check` clean (from unit 1 on) and its named test files green. Each implementer adds the rows of its files to the open round of the change document.

### Unit 1 — Tooling

**Goal.** Every command is a recipe of one `justfile`, the hooks run through `prek`, the checks live under `scripts/checks/`, and CI calls the recipes.

**Model.** `sonnet`. **May change:** `justfile`, `prek.toml`, `.pre-commit-config.yaml` (deleted), `.github/workflows/test.yml`, `scripts/check_docs.py` and `scripts/check_spelling.py` (moved), `pyproject.toml` (dev group, hook configuration only), `uv.lock`. **Must not change:** the content of the two moved checks beyond their paths; `scripts/check_configs.py` and `scripts/import_all.py` (unit 8).

Items:

1. `justfile` — adapted from UFR `justfile`, with its first-line source comment. Recipes: `default`, `setup`, `check`, `test profile="dev" *paths="tests"`, `test-slow`, `pending`. `check` runs `uv run prek run --all-files` only; the `ty` call and the `ty: ignore` grep are added by unit 15. `test` runs `uv run pytest "${@:2}" -m "not slow" --hypothesis-profile="$1"` without `-n auto`: the session has 2 cores and `pytest-xdist` is not a dependency. Evidence: `uv run just --list` prints the six recipes.
2. `prek.toml` — adapted from UFR `prek.toml`: the four builtin hooks, `ruff-check --fix`, `ruff-format`, `codespell` 2.4.3, and the local hooks `check-spelling` and `check-docs` pointing at `scripts/checks/`. `exclude` covers `notebooks/`, `third_party/` and `reference/trixi/Manifest.toml`. `nbstripout` is dropped: unit 5 deletes the notebooks. Add `prek` and `rust-just` to the dev group, run `uv lock`. Delete `.pre-commit-config.yaml`. Evidence: `uv run just check` exits 0.
3. `git mv scripts/check_docs.py scripts/checks/check_docs.py` and the same for `check_spelling.py`; update every reference (`grep -rn "scripts/check_docs\|scripts/check_spelling" --include="*.md" --include="*.toml" --include="*.yml" --include="*.py" . | grep -v "^./docs/changes/"`). The `--files` mode of `check_docs.py` and its *Files* rule are removed here: rule T1 replaces them in unit 3. Evidence: the grep prints nothing.
4. `.github/workflows/test.yml` — the `fast` job becomes `uv sync`, `uv run just check`, `uv run just test ci`, on push to `main` and on pull requests. The `julia` job is kept as it is.

Tests: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_run_record.py` (a smoke test of the recipe).

### Unit 2 — Document conventions

**Goal.** `docs/document-conventions.md` states the form of every document type, and `check_docs.py` enforces the closed vocabularies and the required headings.

**Model.** `opus`. **May change:** `docs/document-conventions.md` (new), `scripts/checks/{check_docs.py,check_md_math.py,markdown.py}`, `prek.toml`, `pyproject.toml` (dev group), the frontmatter and headings of documents under `docs/` other than `docs/changes/`, `docs/index.md`. **Must not change:** any sentence of an ADR or design document (W5); `docs/changes/`.

Items:

1. `docs/document-conventions.md` — adapted from UFR `docs/document-conventions.md`. Keep: *What goes where*, *Frontmatter and status*, *Required sections*, *Change documents* (rule T1, with `origin/main`), *Plans*, *Decision records*, *Design documents*, *Paper records*, *Experiment reports*, *Glossary and notation* without the *Dimension* column, *Findings and verification*, *Provenance labels*, *Unconfirmed decisions*, *Markdown that renders in both previews*, *Checks*. Drop: *Dataset records* (the `DatasetRecord` JSON is the record here), *Format notes*, the unit boundary, the dissertation archive. *References* is written by unit 17; leave a one-line section that says so.
2. `scripts/checks/markdown.py`, `check_md_math.py`, `check_docs.py` — ported from UFR `scripts/checks/` with source comments. `STATUSES` and `REQUIRED_HEADINGS` follow item 1. Add `PRE_FORMAT_CHANGES`, the eight change documents dated before 2026-11-25, exempt from the heading rule (W6). The Done-ledger, index and architecture rules that the old `check_docs.py` had are kept. Add `markdown-it-py` to the dev group if UFR's `markdown.py` needs it.
3. Existing documents: map every `status` to the closed vocabulary (list each mapping in the report), add `kind` to the five design documents (`fv-conventions.md` derivation; the other four `note`), and convert the eleven ADRs per W5.

Evidence: `uv run python scripts/checks/check_docs.py` prints `0 problem(s)`; `git diff 450f31d --stat -- docs/decisions | tail -1` shows eleven files; for each ADR, `git diff 450f31d --word-diff=porcelain -- <file> | grep "^[-+]" | grep -v "^[-+]#\|^+++\|^---"` shows only the moved heading lines and the added *History* lines.

### Unit 3 — Change-document format and `CLAUDE.md`

**Goal.** `check_changes.py` enforces rule T1, and `CLAUDE.md` describes the session protocol of this workflow.

**Model.** `opus`. **May change:** `scripts/checks/check_changes.py` (new), `prek.toml`, `CLAUDE.md`, no test file: UFR has none for this check, and its evidence is item 1's two runs.

Items:

1. `scripts/checks/check_changes.py` — ported from UFR, with `origin/main` as the base and `PRE_FORMAT_CHANGES` skipped. Evidence: `uv run python scripts/checks/check_changes.py` exits 0 on the branch; staging a file with no row makes it exit 1 (show both tails, then unstage).
2. `CLAUDE.md` — rewritten after UFR's `CLAUDE.md`: the pointer to `.claude/rules/`, the session reading list scoped to a planning or orchestrating session, *During and at the end of a session* with rules T1 and T2, and *Repository facts*. Keep these facts from the current file: Python 3.13 and `copy.replace`; the Julia facts; tags cannot be pushed from Claude's sessions; the numerics conventions pointer; the provenance labels. Add: `gfortran` is needed for `uv sync`; `JAX_PLATFORMS=cpu` in a cloud session; `just check` sees only tracked files; the session that changes a dependency runs `uv lock`; the history rule; `legacy/` is importable and otherwise untouched. Remove: the *Files* section rule, the notebook filter, the submodule, the regression tier.

Tests: `uv run just check`.

### Unit 3b — Commit the plan (orchestrator)

Add the frontmatter block (`type: plan`, `status: planned`, `updated: 2026-11-25`, `sources` naming the Project path `claude/refactor-foundation-handoff-2026-11-25.md`) above the verbatim body of `docs/plans/2026-11-25-refactor-foundation.md`, add its row to `docs/index.md`, link it from the change document's `sources`, and commit.

### Unit 4 — Remove the ICLR compatibility code

**Goal.** Nothing in the repository reads an ICLR checkpoint.

**Model.** `sonnet`. **May change:** the files listed; `pyproject.toml` (pytest section); `src/context_flux_no/runs/{index,record}.py`; `docs/coding-standards.md` is the orchestrator's.

Items, all at `450f31d`:

1. Delete `src/context_flux_no/training/io.py`, `tests/test_training_io.py`, `tests/regression/` (whole directory), `scripts/diff_checkpoint_leaves.py`, `scripts/audit_bias_hyperinit.py`, `scripts/backfill_run_records.py`, `tests/unit/test_backfill_run_records.py`.
2. `src/context_flux_no/runs/index.py:90-94` and `:107-113` — remove the two `status == "legacy"` branches and the docstring sentences about legacy checkpoints (`:7-8`, `:104-106`). `load_run` then always goes through `load_state`.
3. `src/context_flux_no/runs/record.py:28` — `Status = Literal["running", "finished", "failed"]`; remove the docstring lines about legacy records (`:10`, `:59`). `RunRecord.read` raises on an unknown status; add that test to `tests/unit/test_run_record.py` if absent.
4. `pyproject.toml:95` — remove the `regression` marker; `:98` — `addopts = "-m 'not slow'"`.
5. `src/context_flux_no/training/__init__.py` — remove any export of `io` names.

Evidence: `grep -rn "load_model\|training\.io\|LEGACY_CHECKPOINTABLE\|translate_legacy_config\|checkpoints.root\|tests/regression\|not regression\|regression:" --include="*.py" --include="*.toml" --include="*.yml" src scripts tests pyproject.toml .github` prints nothing (it prints 50 lines at `450f31d`); `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_run.py tests/unit/test_run_record.py tests/unit/test_checkpoint.py` passes.

### Unit 5 — Remove the xarray path, the notebooks and the submodule

**Model.** `sonnet`. **May change:** the files listed, `pyproject.toml`, `uv.lock`, `.gitmodules`, `.gitignore`, `prek.toml`.

Items:

1. Delete `src/context_flux_no/training/loader.py`; remove its export at `src/context_flux_no/training/__init__.py:7`.
2. `pyproject.toml:13,20,30,41` — remove `dask[distributed]`, `h5netcdf`, `netcdf4`, `xarray`; then `uv lock`. Before removing, confirm with `grep -rn "import dask\|import h5netcdf\|import netCDF4\|import xarray\|from xarray\|from dask" src scripts tests` that only `training/loader.py` used them. `src/context_flux_no/data/sources.py:29` mentions xarray in a comment about `driver_kwds`; if `TheWellDataSource` needs `h5netcdf` at runtime, stop and report instead of removing it.
3. Delete `notebooks/` (five directories). Remove `"*.ipynb"` from `extend-include` and `notebooks` from `extend-exclude` at `pyproject.toml:53-54`, and the notebook entries of `[tool.codespell] skip` at `:107`.
4. Remove the submodule: `git submodule deinit -f third_party/icon && git rm -f third_party/icon`, delete `.gitmodules` if it is then empty, and remove `icon` and `third_party` from the ruff and codespell excludes and from `SKIP_PARTS` of `scripts/checks/check_spelling.py`.

Evidence: `grep -rn "xarray\|h5netcdf\|netcdf4\|dask\|SegmentLoader\|make_segment_axis" --include="*.py" --include="*.toml" src scripts tests pyproject.toml` prints at most the `sources.py:29` comment, reworded without the library names; `test ! -e notebooks && test ! -e third_party`; `uv sync --group dev` succeeds; `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_experiment.py tests/unit/test_trainer.py` passes.

### Unit 6 — Two small fixes

**Model.** `sonnet`. Two commits.

1. `src/context_flux_no/models/multiphysics/disco/operatornet.py:294-298` — the second bottleneck convolution takes `keys_bot[1]`; delete the four comment lines above it. `tests/unit/test_key_hygiene.py:28-36` — the DISCO case becomes a plain tuple again. Reason: the parked fix existed for the ICLR DISCO checkpoints (F15), which are no longer loaded. Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_key_hygiene.py tests/test_models_disco.py` passes with no xfail.
2. `src/context_flux_no/simulations/initial_conditions/step.py:57,63-64` — the default `num_jumps_min` is 2 and the constructor raises `ValueError` when `num_jumps_min < 2`. Reason (Q18a): a periodic step with one jump is a constant state, which makes relative errors undefined. Update the configs under `scripts/data/configs/initial_condition/` that set `num_jumps_min: 1` and the one test that passes 1 (`tests/unit/test_initial_conditions.py:39`). Evidence: a new test in `tests/unit/test_initial_conditions.py` that `PeriodicRandomStepFunction1D(1, 3)` raises; the file passes.

### Unit 7 — `legacy/` and the clean list

**Goal.** The code that awaits its rewrite is under `src/context_flux_no/legacy/`, and `pyproject.toml` names the clean paths.

**Model.** `sonnet`. **May change:** paths under `src/context_flux_no/{models,nn,plotting}` (moved only), `src/context_flux_no/metrics.py` (deleted), imports anywhere, `_target_` strings under `scripts/`, `pyproject.toml`. **Must not change:** any line inside a moved file other than an import path.

Items:

1. `git mv` `src/context_flux_no/models`, `nn` and `plotting` to `src/context_flux_no/legacy/`, with a `legacy/__init__.py` whose docstring states rule R7.
2. Rewrite imports: inside the moved files (25 absolute imports of `context_flux_no.models` or `context_flux_no.nn`), in `tests/test_models_*.py`, `tests/test_nn*.py`, `tests/unit/test_run_record.py:70`, `tests/unit/test_key_hygiene.py`, `tests/unit/test_experiment.py`, and `src/context_flux_no/utils` if it imports either.
3. Rewrite every `_target_: context_flux_no.models.multiphysics.<Name>` under `scripts/training/configs/model/` (seven distinct targets, 22 files) to `context_flux_no.legacy.models.multiphysics.<Name>`.
4. Delete `src/context_flux_no/metrics.py`: nothing imports it at `450f31d`.
5. `pyproject.toml` — the tests of legacy code (`tests/test_models_*.py`, `tests/test_nn*.py`, `tests/unit/test_key_hygiene.py`) move to `tests/legacy/`, so that the strict checks of units 14 to 16 do not reach them (Q13: they run unchanged in the fast tier). Then add `[tool.context_flux_no] clean = ["src/context_flux_no/simulations", "src/context_flux_no/runs", "src/context_flux_no/utils", "src/context_flux_no/data", "src/context_flux_no/training", "src/context_flux_no/evaluation", "src/context_flux_no/experiment.py", "src/context_flux_no/custom_types.py", "tests/unit", "tests/verification", "tests/helpers.py", "tests/conftest.py", "scripts"]` (a path that does not exist yet is skipped by its readers). Q15 named `tests/` as one entry; the split is the writer's and follows from Q13. Add `src/context_flux_no/legacy` and `tests/legacy` to ruff's `extend-exclude` and to codespell's `skip`.

Evidence: `grep -rn "context_flux_no\.\(models\|nn\|plotting\|metrics\)\b" --include="*.py" --include="*.yaml" src scripts tests` prints nothing (52 lines at `450f31d`); `JAX_PLATFORMS=cpu uv run python scripts/check_configs.py | tail -1` prints `14 failing combinations`, as before the move; `JAX_PLATFORMS=cpu uv run just test dev` passes with the baseline counts of section 7 minus the tests deleted in unit 4.

### Unit 8 — One training entry point

**Goal.** `scripts/train.py` trains every model on every dataset, and the config census and the import sweep are tests.

**Model.** `opus`. **May change:** `scripts/train.py` (new), `scripts/training/` (the two scripts deleted; configs moved to `scripts/configs/train/` is *not* done: keep `scripts/training/configs/`), `scripts/training/configs/{config.yaml,config_the_well.yaml,data/*.yaml}`, `scripts/check_configs.py` and `scripts/import_all.py` (deleted), `tests/unit/test_configs.py` and `tests/unit/test_imports.py` (new), `src/context_flux_no/data/sources.py` (one property, item 2).

Decisions: one script (Q9); the source class is chosen by the data config; W3; W7.

Items:

1. `scripts/train.py` — the body of `scripts/training/train_multiphysics.py:44-76`, with the source built by `hydra.utils.instantiate(cfg.data.source, well_split_name=split, window_size=window)`. Each data config gains a `source:` node with `_target_` `context_flux_no.data.ZarrWellDatasetSource` or `TheWellDataSource` and that class's keyword arguments, taken from the two scripts (`train_multiphysics.py:52-60`, `train_the_well.py:33-41`). `config_the_well.yaml` is deleted; `training=the_well` stays selectable. Delete both old scripts and update `scripts/training/README.md` and `README.md`.
2. The loss arguments come from the source for both classes: `train_the_well.py:54` hard-codes `(0.015, 1 / 64, 1 / 64)`. Add `loss_args` as a property of `WellDatasetSourceBase` returning `(dt, *dxs)` from `metadata_common` (`train_multiphysics.py:20-23` does this in the script). If `TheWellDataSource`'s metadata does not hold these values at the downsampled resolution, stop and report: the constant is then a fact about the dataset that needs Joon.
3. `tests/unit/test_configs.py` — the census of `scripts/check_configs.py` as a parametrized test, with `_base` configs skipped and `DIMENSIONS = {"dpot": 1, "fno_spatiotemporal": 1, "hyperfluxno_vit": 1, "dpot_2d_euler": 2, "dpot_2d_scalar": 2}` restricting those five to data of their dimension (W3). A model config absent from the table pairs with every data config.
4. `tests/unit/test_imports.py` — every module of the package imports on CPU, `legacy/` included (from `scripts/import_all.py`).
5. The loader is still built in the script in this unit; unit 9 moves it.

Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_configs.py tests/unit/test_imports.py` passes; `JAX_PLATFORMS=cpu uv run python scripts/train.py --cfg job | head -5` prints a composed config; `tests/unit/test_experiment.py` passes (it trains for a few steps on a synthetic store).

### Unit 9 — The loader factory

**Goal.** One function builds the training and evaluation loaders.

**Model.** `opus`. **May change:** `src/context_flux_no/data/{loader.py,__init__.py,sources.py}`, `src/context_flux_no/experiment.py`, `scripts/train.py`, `scripts/bench_loader.py` (new), `tests/unit/test_loader.py` (new).

Decisions: Q10; W4.

Items:

1. `data/loader.py::make_loader(source, *, batch_size: int, seed: int, shuffle: bool = True, repeat: bool = True, host_prefetch: int = 20, device_prefetch: int = 10)` — the pipeline of `scripts/training/train_multiphysics.py:26-41` (shuffle, repeat, batch with `drop_remainder=True`, host prefetch, `jax.device_put`, `ThreadPrefetchIterDataset`). `shuffle=False, repeat=False` gives the evaluation order. The `grain.DataLoader` path of `train_the_well.py:16-22` is not kept: one way to build a loader.
2. `data/loader.py::gc_paused(every: int)` — a context manager that calls `gc.disable()`, yields a callable to invoke once per step that runs `gc.collect()` every `every` steps, and restores the collector on exit (W4). `experiment.train` uses it around the loop. Property test: the collector is enabled again after an exception.
3. `tests/unit/test_loader.py` — with a small in-memory source: every index appears once per epoch; two loaders with the same seed give the same batches; `shuffle=False, repeat=False` yields `len(source) // batch_size` batches in order.
4. `scripts/bench_loader.py` — times `n` batches of `make_loader` on a named dataset and prints batches per second, host and device. It is not run in this session beyond `--help`; the roadmap item 4.6 keeps "numbers pending (cluster)".
5. `data/sources.py:323` — remove the `ty: ignore` by giving `__getitem__` the signature of its base; if that needs a design choice, stop.

Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_loader.py tests/unit/test_experiment.py tests/unit/test_trainer.py` passes; `grep -rn "ThreadPrefetchIterDataset\|grain.DataLoader" scripts src | grep -v "data/loader.py"` prints nothing.

### Unit 10 — `evaluation/metrics.py`

**Goal.** The metric functions of the rebuttal branch, as a clean module with their tests.

**Model.** `opus`. **May change:** `src/context_flux_no/evaluation/{__init__.py,metrics.py}`, `tests/unit/test_metrics.py`.

Items: port from the rebuttal branch's `src/context_flux_no/metrics.py` every function and container except `ideal_gas_pressure` (unit 11): the norms and relative errors, `spatial_mean_field`, `error_area_fraction`, `total_variation`, `radial_power_spectrum`, `error_power_spectrum`, `conservation_drift`, `RolloutReferences`, `rollout_references`, `RolloutDiagnostics`, `rollout_diagnostics`, `first_true_step`, `valid_horizon`, `SegmentErrors`, `segment_errors`, `largest_jump`, `count_jumps`. Names follow the repository's convention: `l2_norm`, `relative_l2_error` (lower case), not `L2_norm`. Docstrings lose every reference to a rebuttal action or a diagnostic number (D1 to D6); they say what the function computes. Port `tests/test_metrics.py` of the rebuttal branch without `test_ideal_gas_pressure_inverts_the_energy`, and delete any test that rule R11 excludes (report each).

Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_metrics.py` passes; `grep -n "rebuttal\|action [0-9]\|D[1-6]\b" src/context_flux_no/evaluation/metrics.py` prints nothing.

### Unit 11 — Admissibility and channel order

**Goal.** A family says whether a state is physically admissible, and one function converts a store's channel order to the family's.

**Model.** `opus`. **May change:** `src/context_flux_no/simulations/pde/{base,shallow_water,euler}.py`, `src/context_flux_no/simulations/writer.py`, `tests/unit/test_admissible.py` (new), `tests/unit/test_generate_dataset.py`.

Decisions: Q7; W8.

Items:

1. `AbstractPDEFamily.positive_quantities(u: [n_eqn, *shape]) -> [n_quantities, *shape]` — the quantities that must stay positive, in the family's conserved order. Default: an array with zero quantities. `ParametrizedShallowWater1D`: the depth. `Euler2D`: density and pressure, with `pressure(u)` as a method using the family's `gamma` (the formula of the rebuttal branch's `ideal_gas_pressure`). `admissible(u) -> Bool[""]` is `all(positive_quantities(u) > 0)` on the base class.
2. `simulations/writer.py::store_channel_order(field_rank_names, n_spatial_dims) -> tuple[int, ...]` — the permutation `p` with `u_store[i] == u_family[p[i]]`, derived from the rank groups the writer writes at `writer.py:166-186` and the order in which `WellDatasetSourceBase` packs them (rank 0 first). Test: write a trajectory of distinct constants per channel with `WellWriter` for `Euler2D`, read it back through `ZarrWellDatasetSource`, and assert the permutation maps it to the family's order; the same for a scalar family (identity).
3. Tests by property (R11): a state with one negative depth is not admissible; `Euler2D.pressure(to_conserved(rho, u, v, p))` returns `p`.

Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_admissible.py tests/unit/test_generate_dataset.py` passes.

### Unit 12 — `evaluation/rollout.py` and `evaluation/summary.py`

**Goal.** A model is rolled out over a test set on the accelerator, and the per-step arrays are reduced on the host.

**Model.** `opus`. **May change:** `src/context_flux_no/evaluation/{rollout.py,summary.py,__init__.py}`, `tests/unit/test_rollout.py`, `tests/unit/test_summary.py`, `tests/helpers.py`.

Decisions: Q3, Q4, Q14; R12.

Items:

1. `rollout.py::Operator` — a `typing.Protocol` with `rollout(u, args, num_steps, *, key=None, inference=None)` returning `(predictions, aux)`, the signature of `AbstractMultiphysicsOperator.rollout` at `legacy/models/multiphysics/abstract.py`.
2. `rollout.py` — from the rebuttal branch's `evaluation.py:106-286` and `:485-628`: `RolloutEvaluation`, `evaluate_rollouts`, `evaluate_references`, `SegmentBaselines`, `evaluate_segments`, `evaluate_segment_baselines`, with their private helpers. `evaluate_rollouts` takes `positive_quantities: Callable | None` in place of `minima_fn` and `coefficients`: the caller passes the bound method of a family instance, composed with the channel order of unit 11. `append_grid_channels` is not imported from `legacy/`; if a ported function needs it, stop and report.
3. `summary.py` — from `evaluation.py:288-483` and `:630-710`: `summarize_rollouts`, `summarize_conservation_drift`, `match_trajectories`, `mean_by_bin`, with their helpers.
4. Tests: port the tests of the rebuttal branch's `tests/test_evaluation.py` at lines 83 to 300 (baselines, windows, matching, binning, summaries). They use a stub that satisfies `Operator`, defined in `tests/helpers.py`: one that holds the last frame, and one that shifts the state by one cell per step, for which the error against a translating wave is zero. No test imports `legacy/`.

Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_rollout.py tests/unit/test_summary.py` passes; `grep -rn "legacy" src/context_flux_no/evaluation tests/unit/test_rollout.py tests/unit/test_summary.py` prints nothing.

### Unit 13 — `evaluation/cost.py` and `scripts/evaluate.py`

**Goal.** One command evaluates named runs on a named test set and writes one result file with its record.

**Model.** `opus`. **May change:** `src/context_flux_no/evaluation/{cost.py,record.py,__init__.py}`, `scripts/evaluate.py`, `tests/unit/test_cost.py`, `tests/unit/test_evaluate.py`.

Decisions: Q5, Q6; W9.

Items:

1. `cost.py` — `Measurement`, `count_flops`, `measure` from the rebuttal branch's `cost.py:31-122`. Port the first two tests of its `tests/test_cost.py`.
2. `scripts/evaluate.py` — arguments: `--runs` (one or more run directories or run ids resolved through `runs.index`), `--data` (a data config name under `scripts/training/configs/data/`), `--split` (default `test`), `--context-length`, `--area-thresholds`, `--batch-size`, `--limit`, `--out`. It loads each model with `runs.load_run`, reads the split through the config's source in evaluation order (unit 9), calls `evaluate_rollouts` and `evaluate_references`, and writes one HDF5 file with one group per run plus `references`.
3. `evaluation/record.py::EvaluationRecord` — a frozen dataclass written as `<out>.json` beside the result: the run ids, the `dataset_id`, the split, the arguments, and the provenance fields of `utils/provenance.py`, with `write` and `read` as `RunRecord` has them.
4. `tests/unit/test_evaluate.py` — trains a run for a few steps on a synthetic store, as `tests/unit/test_experiment.py` does, runs the script's `main` on it with `--limit 2`, and asserts that the result file has the run's group with arrays of the expected shapes and that the record round-trips.

Evidence: `JAX_PLATFORMS=cpu uv run just test dev tests/unit/test_cost.py tests/unit/test_evaluate.py` passes; `JAX_PLATFORMS=cpu uv run python scripts/evaluate.py --help` prints the arguments.

### Unit 14 — Docstrings and complexity

**Goal.** Ruff's `D100` to `D103`, `W605` and `C901` at 8 hold on the clean list.

**Model.** `opus`. **May change:** `pyproject.toml` (ruff section), docstrings anywhere on the clean list, and the bodies of the functions `C901` flags.

At `450f31d` the rules report 227 findings on the paths that are then clean (105 `D102`, 89 `D103`, 18 `D101`, 11 `D100`, 4 `C901`; 81 under `simulations/`, 35 under `training/`).

Items: select the rules as UFR's `pyproject.toml` does, with the per-file-ignore pattern built from the clean list; write the missing docstrings in the equinox convention, each saying what the object does, not restating its name; split the four functions `C901` flags without changing behavior. A test function's docstring states the property it pins. If a docstring cannot be written without understanding a numerical choice that no document records, mark it `[unconfirmed]` in the change document's *Open issues* and write what the code does.

Evidence: `uv run ruff check --select D100,D101,D102,D103,C901,W605 $(clean list)` prints `All checks passed!`; `JAX_PLATFORMS=cpu uv run just test dev` passes.

### Unit 15 — Types on the package, and three checks

**Goal.** `ty` reports nothing on the package's clean paths, and the container, dependency and docstring checks run.

**Model.** `opus`. **May change:** files on the clean list under `src/`, `scripts/checks/{check_containers,check_dependencies,check_docstrings}.py` (new), `prek.toml`, `justfile`, `pyproject.toml`.

At `450f31d`, `ty` 0.0.84 reports 59 diagnostics on these paths, 43 of them under `simulations/`. Units 4 to 13 change the count; measure it again at the start of the unit.

Items:

1. Fix each diagnostic by correcting the annotation or the code. No `ty: ignore`, no `cast` that hides a real mismatch, no `Any` introduced to silence a diagnostic. Fourteen are `invalid-attribute-override` on the family classes: if the fix changes a public signature of `AbstractPDEFamily`, stop the unit (section 9).
2. Port `check_containers.py`, `check_dependencies.py` and `check_docstrings.py` from UFR with source comments, reading the clean list from `pyproject.toml`. Drop the unit-library rule of `check_dependencies.py`; add rule R12 (`evaluation/` imports nothing from `legacy/`) and "the clean paths import nothing from `legacy/`" except `experiment.py`, which instantiates models by config and imports none.
3. `justfile::check` gains the `ty` call on the clean list and the `ty: ignore` grep, as in UFR.

Evidence: `uv run ty check <package clean paths>` prints `All checks passed!`; `uv run just check` exits 0; `JAX_PLATFORMS=cpu uv run just test dev` passes.

### Unit 16 — Types on the tests and scripts; runtime shape checks

**Model.** `opus`. **May change:** `tests/unit/`, `tests/verification/`, `tests/helpers.py`, `tests/conftest.py` (new), `scripts/`.

At `450f31d`, `ty` reports 75 diagnostics on `tests/unit`, `tests/verification`, `tests/helpers.py` and `scripts`, most of them under `tests/unit`. Measure again at the start of the unit.

Items:

1. Fix the diagnostics under the same constraints as unit 15.
2. `tests/conftest.py` — adapted from UFR's: the Hypothesis profiles `dev` (10), `default` (50), `ci` (100, derandomized), and `jaxtyping.install_import_hook` with `beartype.beartype` on the package's clean paths, read from `pyproject.toml`. It does not enable x64: this package trains in float32, and `tests/verification/conftest.py` enables x64 for its tier. Add `beartype` to the dev group if absent.
3. A shape or type error that the import hook exposes is a real defect: fix the annotation when the code is right, fix the code when the annotation is right, and list each in the report.

Evidence: `uv run just check` exits 0; `JAX_PLATFORMS=cpu uv run just test dev` and `uv run just test-slow dev` pass (the Trixi tests skip).

### Unit 17 — One bibliography

**Model.** `opus`. **May change:** `docs/references.bib`, `docs/references.md` (generated), `scripts/docs/generate_references.py`, `scripts/checks/check_references.py`, `prek.toml`, `justfile` (recipe `references`), `pyproject.toml` (dev group: `bibtexparser`), citations in docstrings and documents outside `docs/changes/` and `docs/plans/`, `docs/document-conventions.md` § References.

Items: port the generator and the check from UFR; collect every cited work from `sources:` frontmatter, design documents and docstrings into `references.bib` (the `sources` of `docs/design/fv-core-plan.md` alone names eleven); rewrite each citation to the key form (`Author Year`, `Author & Author Year`, `Author et al. Year`) with its locator after a comma. A work whose bibliographic data cannot be completed from what the repository states is entered with the fields that are known and listed in the report; nothing is invented.

Evidence: `uv run python scripts/checks/check_references.py` exits 0; `uv run just references && git diff --exit-code docs/references.md` exits 0.

## 7. Baseline

At `450f31d`, in a cloud container with `gfortran` installed, `JAX_PLATFORMS=cpu`:

- `uv run pytest -m "not regression and not slow" -q`: **107 passed, 1 skipped, 14 deselected, 1 xfailed** in 274 s. The skip is `test_describe_records_the_julia_side_versions` (no Julia). The xfail is the DISCO case of `tests/unit/test_key_hygiene.py` (F15), which unit 6 turns into a pass.
- `uv run python scripts/check_configs.py`: **14 failing combinations** of 66 (W3 lists them).
- `ruff check --select D100,D101,D102,D103,C901` at complexity 8 on the future clean paths: **227 findings**.
- `ty check` (0.0.84): **59 diagnostics** on the package's future clean paths and **75** on `tests/unit`, `tests/verification`, `tests/helpers.py` and `scripts`.
- `uv run python scripts/check_docs.py`: `0 problem(s)`. `uvx codespell@2.4.3` and `scripts/check_spelling.py`: 0 findings.

The orchestrator reruns the first two commands in unit 0 and records its own numbers. A difference from these is reported before unit 1 starts.

## 8. The documents unit

Before it, the full tier runs once: `JAX_PLATFORMS=cpu uv run just test default`, then `uv run just test-slow dev`. The counts go into the change document.

The orchestrator then writes, in one or more commits:

- **`docs/changes/2026-11-25-refactor-foundation.md`**: *Summary*, *Design* (every decision of section 10 with its rejected alternative, and every `[unconfirmed]` marker for W1 to W10 and for anything decided during the run), *Bugs fixed*, *Tests* with exact counts, *Open issues*; `status: landed`.
- **`docs/roadmap.md`**, edited in place with the phase numbers kept (Q20):
  - Principle 1 becomes "Every phase ends with a repository that trains a model on a Well-schema dataset, with the fast tier green." Principle 2 becomes "Refactor behind tests: the unit and property tiers guard every change, and the verification tier guards the numerics." Principle 3 becomes "Nothing dead is kept; code that awaits its rewrite lives in `legacy/`, and the tag `iclr2027-submission` is the archive of what was submitted."
  - Items 0.7 and 5.4 are closed as dropped, with the reason (no ICLR compatibility, decided 2026-11-25). The Phase 1 exit line about `evaluate_cubic.ipynb` is struck.
  - Phase 2g and item 3c are removed. Phase 3 is re-scoped: "written on `fv/`, replacing `legacy/`; no shim."
  - A new **Phase 1b — Foundation** records this branch, and items 4.5, 6.1 and the entry-point half of 7.1 are marked done by it. Item 4.6 reads "loader factory done; benchmark numbers pending (cluster)".
  - The *Parked* entries on the orbax pin and on `operatornet_scale` are closed. A Done row is appended.
- **`docs/architecture.md`**: `legacy/`, `evaluation/`, `data/loader.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/checks/`.
- **`docs/coding-standards.md`**, rewritten after UFR's: *Tiers* (the `just` recipes; no regression tier), *Scopes* (the clean list), *Rules* (R7, R10, R11, R12, American English, containers, `copy.replace`), *Checks* (one row per hook), *Deliberate exceptions* (`legacy/`, `docs/changes/`, `docs/plans/`).
- **ADRs.** ADR-0006: a *History* line and a *Decision* paragraph stating that Phase 3's `InContextOperator` exposes `generate_solver(context)` and `step(solver, u, args)` with the law `step(generate_solver(u), u[-1], args) == __call__(u, args)[0]` as a property test, and that the ICLR shim is dropped. ADR-0008: the regression tier is removed (status `amended`). ADR-0010: legacy records and the backfill are removed (status `amended`).
- **`docs/findings.md`**: F16, the two channel orders of Euler (W8). F1 and F15 each gain a dated closing line.
- **`docs/verification.md`**: V22 is marked retired with the date and the reason; rows are added for the evaluation properties that units 10 to 12 pin.
- **`docs/index.md`**, **`docs/glossary.md`** (the terms *clean list*, *legacy*, *admissible*, *evaluation record*).
- **`docs/plans/2026-11-25-refactor-foundation.md`**: `status: executed`.

Then `uv run just check`, `uv run just pending` (its output goes at the end of the final message to Joon), and the push.

## 9. Gates

The run is unattended and does not stop for review between units.

- **After every unit:** the orchestrator reviews the diff against the unit's allowed paths, runs `uv run just check` and the unit's test files, and pushes the branch. A branch pushed after every unit is what Joon finds if the run stops.
- **Full fast tier** (`uv run just test dev`) after units 7, 13 and 16.
- **A unit is retried once.** If its implementer returns `BLOCKED` or `ISSUES` a second time, or a gate fails after one repair attempt, the orchestrator records the state in the change document's *Open issues*, skips the units that depend on the failed one, continues with the units that do not, and says so in its final message. The dependencies are: 2 → 3 → 3b; 4, 5 and 6 are independent of each other; 7 needs 4 to 6; 8 needs 7; 9 needs 8; 10 to 13 need 7 and run in order, 13 also needs 9; 14 to 16 need every unit before them that landed; 17 needs 2.
- **A design choice this document did not make** is taken, marked `[unconfirmed]`, and the run continues, unless the choice changes a public signature of a class on the clean list. That stops the unit, not the run.
- **The run stops entirely** only if unit 0 fails, or if `main` or `workflow-bootstrap` moved in a way that touches a file this document names.
- Nothing is merged. The review and the merge are Joon's.

## 10. Decisions and where they are recorded

| Decision | Reason | Recorded in |
|---|---|---|
| No compatibility with the ICLR checkpoints; the regression tier is retired with no replacement | The rebuttal is over, the tag and the original repository archive the submission, and no phase before Phase 3 changes a model | Roadmap principles; **ADR-0008 History** (amended) |
| The 0.7 audit and the 5.4 backfill are dropped | They served the ICLR checkpoints | Roadmap; ADR-0010 History |
| Code awaiting its rewrite lives in `legacy/`, frozen; a clean list scopes the strict checks (decision 2, Q8, Q12, Q15) | One standard for the whole tree kept the strict checks off everywhere | **New ADR-0012** (hard to reverse, changes every import path); coding standards R7 |
| Tests of legacy code run unchanged and leave with the code (Q13) | They prove `legacy/` still imports and trains | Change document *Design* |
| The workflow of `UniversalForceReconstruction`: rules in the repository, the implementer agent, plans, round tables, unconfirmed markers, `just` and `prek` (decision 1) | An unattended run needs its rules in the tree and its decisions listed | `CLAUDE.md`; `docs/document-conventions.md`; change document *Design* |
| The general half of the rebuttal's evaluation code only (Q1) | The specific half is tied to models that Phase 3 replaces | Change document *Design* |
| The two-stage contract is a requirement on Phase 3, not code now (Q2) | No adopter in this repository once Q1 is applied | **ADR-0006 History** |
| `evaluation/{metrics,rollout,summary,cost}.py`; a `Protocol` for models; no import of `legacy/` (Q3, Q4, R12) | Accelerator and host code separate; the package survives Phase 3 unchanged | Coding standards R12; architecture |
| `scripts/evaluate.py` selects through `runs.index()`; HDF5 results with a JSON record (Q5, Q6) | Presets encoded ICLR paths | Change document *Design* |
| Admissibility is a method of the family (Q7) | Family knowledge, like `to_conserved` | Change document *Design*; finding F16 for the channel order |
| One `scripts/train.py`; the census is a test with declared pairings (Q9, W3) | Two scripts differed in the source class and the loader only | Change document *Design* |
| One `make_loader`; two thin source subclasses (Q10); the GC policy in the driver (W4) | The subclasses differ in exactly their two abstract methods | Change document *Design* |
| Evaluation tests use a stub model (Q14) | A stub has an answer computable by hand | Coding standards R11 |
| One bibliography (Q16) | Phases 2b to 2e add many citations | `docs/document-conventions.md` § References |
| The ICON submodule and the notebooks are deleted (decision 4, Q17) | Nothing imports the submodule; the notebooks read ICLR-era data | Change document *Design* |
| Step initial conditions need at least 2 jumps (Q18a) | A one-jump periodic step is a constant state | Change document *Bugs fixed* |
| The `operatornet.py` key fix is applied; `operatornet_scale` as checkpointed leaves is dropped (Q12) | F15 existed for checkpoints no longer loaded; Phase 3 rewrites the baselines | Findings F15 closing line |
| One branch, unattended, with the stop conditions of section 9 (Q11, Q19) | The units form one dependency line | This document |
| The roadmap keeps its phase numbers (Q20) | Eight merged change documents cite them | Roadmap |
| The FV core is out of scope (decision 3) | Joon writes it himself | Roadmap Phase 2 |

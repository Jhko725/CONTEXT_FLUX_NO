---
type: change
status: in-progress
updated: 2026-11-25
branch: refactor-foundation
roadmap: Phase 1b (foundation); closes 4.5, 6.1 and the entry-point half of 7.1
sources:
  - "docs/plans/2026-11-25-refactor-foundation.md (the handoff)"
---

# 2026-11-25 — `refactor-foundation`

## Summary

The branch gives the repository the working conventions of
`UniversalForceReconstruction`, removes the code that kept the ICLR checkpoints
loadable, moves the code that awaits its rewrite under `legacy/`, and replaces the
training entry point, the data loader and the evaluation code. It carries the commit of
`workflow-bootstrap` (`d60a19a`), which is not merged separately. The orchestrating
session completes this section in the documents unit.

## Design

The first three points come from the change document of `workflow-bootstrap`, which
this document replaces: one branch has one change document.

- **The rules are copied into the repository.** A cloud session does not load the global `CLAUDE.md`, and until now the rules reached a session only through the Project. Rejected: a pointer to the Project, which a session started from the repository alone cannot follow.
- **The hooks are copied unchanged**, each with a first line naming its source and commit. The amend rule of `block_history_rewrite.py` asks the remote directly, which matters in the shallow clones of a cloud session.
- **`implementer.md` names `src/context_flux_no/legacy/` before that directory exists.** `refactor-foundation` creates it; the boundary is stated here so that the agent file does not change in the middle of the run that uses it.

## Bugs fixed

None yet.

## Tests

The baseline of unit 0 is recorded under *Open issues* until the full tier runs at the
end of the branch.

## Open issues

- The hooks have no automated test in this repository. `UniversalForceReconstruction` has none either.
- **The agent type.** The session started outside the repository and offers no `implementer` type. Each unit runs as a general-purpose subagent on the unit's model, briefed with the *Boundary*, *Stop rule*, *Before and after* and *Report* sections of `.claude/agents/implementer.md` and told not to run `git push`, since the hook of the agent file does not run for that type (handoff section 4, step 3).
- **Six change documents predate the format, where the handoff says eight.** `docs/changes/` at `450f31d` holds six documents dated before 2026-11-25. `PRE_FORMAT_CHANGES` lists those six. [unconfirmed]

- **The new hooks rewrote whitespace in 45 tracked files** (unit 1): trailing spaces and final newlines, among them files that unit 7 moves under `legacy/`. `git diff --ignore-space-at-eol` shows no other change. The alternative was a wider `exclude` in `prek.toml`. [unconfirmed]
- **`tests/conftest.py` arrives in unit 1 with the Hypothesis profiles only.** The handoff creates it in unit 16, but `just test dev` needs the `dev` profile from unit 1 on. Unit 16 adds the import hook. [unconfirmed]
- The item-3 grep of unit 1 filters `^./docs/changes/`, which GNU grep's output never matches; with `^docs/changes/` it prints nothing.

- **ADR conversion (unit 2, W5).** No ADR had *Forces* text, so each `### Forces` holds `*[not recorded]*`, and the template allows that marker. *History* opens with an "accepted" line dated from the ADR's status line. The extra sections of ADR-0001, 0004, 0010 and 0011 follow *Decision* in their original order. Compared as multisets of lines, no sentence of an ADR changed. [unconfirmed]
- **The document types `guide`, `format` and `dataset` and the status `historical` are not in the vocabulary** (unit 2): no document here uses them. [unconfirmed]
- **`check_docs.py` has no architecture rule.** The handoff asked to keep the old check's rule, and the old check had none; the rule of `UniversalForceReconstruction` reports 19 problems on the fenced module map of `docs/architecture.md`. [unconfirmed]
- **The glossary stays a bullet list.** `docs/document-conventions.md` describes the table form, which the glossary takes when it is next rewritten. [unconfirmed]

- **`TheWellDataSource` scales its metadata by `downsample_spatial` and delivers the native grid** (found in unit 8). `data/sources.py:500-504` multiplies `spatial_resolution` by the factor, and no `__getitem__` slices the data; commit `41ac847` dropped the slice that `534503f` had added. `well_euler_2d` sets the factor to 8, so a Well run sees the native grid with spacings of the grid 8 times coarser, which is also what the constant `(0.015, 1 / 64, 1 / 64)` of the old script described. The branch does not fix this: restoring the slice or the spacings changes what every Well run trains on, and that is Joon's decision. `scripts/train.py` takes the loss arguments from the source's metadata for both classes, as the handoff says; the first attempt at unit 8 stopped on this and the retry proceeds on that reading. The orchestrator records it as a finding in the documents unit. [unconfirmed]

- **`gc_paused` is built and tested, and no driver uses it** (unit 9, W4). No committed revision of the repository pauses the collector, so no period exists to keep, and `Trainer.train`, not `experiment.train`, owns the step loop; a per-step hook would change a signature in `training/`. The period and the place of the call are left to Joon. [unconfirmed]
- **`worker_count` is removed from the data configs** (unit 9): the one loader factory has no workers, and nothing read the key once the `grain.DataLoader` path of the Well script was gone. A Well run therefore reads on the host without worker processes. [unconfirmed]
- **The loss arguments of a Well run now come from the store's metadata** (unit 8): `dt` from the time grid, and each spacing as the native spacing times `downsample_spatial`. They equal the old constant `(0.015, 1 / 64, 1 / 64)` only if the store's time step is 0.015 and its native spacing is 1/512. No Well store is in the session, so this was not checked. [unconfirmed]

## Review notes

### Round 1: from `450f31d`

| Change | Thoughts | Modifications |
|---|---|---|
| `.claude/rules/working-rules.md` — copied: the working rules of Joon's global `CLAUDE.md` | | |
| `.claude/rules/python.md` — copied: the design, Python and JAX rules | | |
| `.claude/agents/implementer.md` — adapted: the subagent that implements one commit of a handoff document | | |
| `.claude/hooks/block_history_rewrite.py` — copied: blocks a rebase, a force-push and an amend of a pushed commit | | |
| `.claude/hooks/block_push.py` — copied: blocks `git push` while the implementer runs | | |
| `.claude/settings.json` — registers the history hook | | |
| `CLAUDE.md` — points at `.claude/rules/`, states the history rule and names the agent; check paths follow the move to `scripts/checks/` | | |
| `docs/index.md` — the row of `workflow-bootstrap` replaced by the row of this branch; check paths follow the move | | |
| `docs/changes/2026-11-25-refactor-foundation.md` — this document | | |
| `justfile` — new: recipes `default`, `setup`, `check`, `test`, `test-slow` and `pending`, adapted from `UniversalForceReconstruction` |  |  |
| `prek.toml` — new: the hooks (builtin, ruff, codespell, `check-spelling`, `check-docs`) run through `prek` |  |  |
| `.pre-commit-config.yaml` — deleted: replaced by `prek.toml` |  |  |
| `.github/workflows/test.yml` — the `fast` job runs `just check` and `just test ci`; pushes trigger on `main` only |  |  |
| `scripts/check_docs.py` — moved to `scripts/checks/` |  |  |
| `scripts/checks/check_docs.py` — moved from `scripts/`; the `--files` mode and its *Files* rule removed; root path follows the move |  |  |
| `scripts/checks/check_spelling.py` — moved from `scripts/check_spelling.py`; usage line follows the move |  |  |
| `pyproject.toml` — dev group: `prek` and `rust-just` added, `pre-commit` and `nbstripout` removed; codespell comment path |  |  |
| `uv.lock` — relocked for the dev group change |  |  |
| `README.md` — path of `check_docs.py` updated; the hook install command is `just setup` |  |  |
| `docs/coding-standards.md` — paths of `check_docs.py` and `check_spelling.py` updated, nothing else |  |  |
| `docs/roadmap.md` — paths of `check_docs.py` and `check_spelling.py` updated, nothing else |  |  |
| `scripts/data/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `scripts/training/configs/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `src/context_flux_no/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `tests/conftest.py` — new: the Hypothesis profiles `dev`, `default` and `ci` that `just test` selects |  |  |
| `docs/document-conventions.md` — new: the form of every document type, adapted from `UniversalForceReconstruction`, with rules T1 and T2 |  |  |
| `scripts/checks/check_docs.py` — replaced by `UniversalForceReconstruction`'s: closed vocabularies, required headings with `PRE_FORMAT_CHANGES`, index links, Done ledger, ADR agreement, unconfirmed markers, templates; keeps the `updated` and `branch` rules |  |  |
| `scripts/checks/check_md_math.py` — new, copied from `UniversalForceReconstruction`: both Markdown previews and the frontmatter's form |  |  |
| `scripts/checks/markdown.py` — new, copied from `UniversalForceReconstruction`: the section tree the two checks read |  |  |
| `prek.toml` — registers `check-md-math`; `check-docs` runs on the documents and its own two modules |  |  |
| `pyproject.toml` — dev group: `markdown-it-py` added |  |  |
| `uv.lock` — relocked for `markdown-it-py` |  |  |
| `docs/decisions/` — the eleven ADRs moved under the template's headings (*Motivation*, *Forces* marked *[not recorded]*, *Alternatives rejected*, *History*); no sentence changed; `updated` and `sources` in the frontmatter |  |  |
| `docs/design/` — `kind` added (`fv-conventions.md` derivation, the other four note); `sources` as quoted block lists; `verified_by` of `python-julia-repo.md` quoted so the block parses |  |  |
| `docs/architecture.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/findings.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/glossary.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/roadmap.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/verification.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/index.md` — links `document-conventions.md`; ADRs under *In force* and *Superseded*; ADR-0004 and ADR-0011 shown as amended; the status list replaced by a pointer to the conventions |  |  |
| `scripts/checks/check_changes.py` — new, adapted from `UniversalForceReconstruction`: each round's rows cover exactly the files changed in it (rule T1); base `origin/main`; skips `PRE_FORMAT_CHANGES` |  |  |
| `prek.toml` — registers `check-changes`, which runs on every commit |  |  |
| `CLAUDE.md` — rewritten after `UniversalForceReconstruction`'s: reading list for planning sessions, rules T1 and T2, repository facts; the *Files* rule, notebook filter, submodule and regression tier removed |  |  |
| `docs/plans/2026-11-25-refactor-foundation.md` — new: the handoff document of this branch, verbatim under a frontmatter block |  |  |
| `src/context_flux_no/training/io.py` — deleted with its test `tests/test_training_io.py`: it loaded the ICLR checkpoints (`load_model`, `translate_legacy_config`) |  |  |
| `tests/test_training_io.py` — deleted with `training/io.py` |  |  |
| `tests/regression/` — deleted: the regression tier compared the ICLR checkpoints with goldens and has no replacement |  |  |
| `scripts/diff_checkpoint_leaves.py` — deleted: it compared ICLR checkpoint leaves |  |  |
| `scripts/audit_bias_hyperinit.py` — deleted: it audited the ICLR checkpoints |  |  |
| `scripts/backfill_run_records.py` — deleted: it wrote records for ICLR checkpoint directories |  |  |
| `tests/unit/test_backfill_run_records.py` — deleted with the script it tested |  |  |
| `src/context_flux_no/runs/index.py` — the two `legacy` branches of `load_state` and `load_run` and their docstring sentences removed |  |  |
| `src/context_flux_no/runs/record.py` — `Status` loses `"legacy"`; `RunRecord.from_dict` (so `read`) raises `ValueError` on an unknown status; legacy docstring lines reworded |  |  |
| `tests/unit/test_run_record.py` — the round trip no longer draws `legacy`; new test that `read` raises on an unknown status |  |  |
| `pyproject.toml` — `regression` marker removed; `addopts = "-m 'not slow'"` |  |  |
| `src/context_flux_no/training/loader.py` — deleted: the deprecated xarray segment loader (`SegmentLoaderBackground`, `make_segment_axis`), which nothing imported |  |  |
| `src/context_flux_no/training/__init__.py` — the export of `SegmentLoaderBackground` removed |  |  |
| `src/context_flux_no/data/sources.py` — the comment in `IO_PARAMS` no longer names xarray and h5netcdf; `TheWellDataSource` opens files with `h5py.File` |  |  |
| `pyproject.toml` — `dask[distributed]`, `h5netcdf`, `netcdf4` and `xarray` removed from the dependencies; the notebook and submodule entries removed from ruff's `extend-include` and `extend-exclude` and from codespell's `skip` |  |  |
| `uv.lock` — relocked without the four libraries and their dependencies |  |  |
| `notebooks/` — deleted: five legacy notebook directories |  |  |
| `third_party/icon` — the ICON baseline submodule removed |  |  |
| `.gitmodules` — deleted: it held only the ICON submodule |  |  |
| `prek.toml` — `exclude` loses `notebooks/` and `third_party/` |  |  |
| `scripts/checks/check_spelling.py` — `SKIP_PARTS` loses `third_party`, `icon` and `notebooks` |  |  |
| `src/context_flux_no/legacy/models/multiphysics/disco/operatornet.py` — the second bottleneck convolution takes `keys_bot[1]` instead of reusing `keys_bot[0]`; the four comment lines about F15 removed | | |
| `tests/legacy/test_key_hygiene.py` — the DISCO case is a plain tuple again, without the strict `xfail` | | |
| `src/context_flux_no/simulations/initial_conditions/step.py` — `PeriodicRandomStepFunction1D` defaults `num_jumps_min` to 2 and raises `ValueError` below 2: one periodic jump is a constant state |  |  |
| `tests/unit/test_initial_conditions.py` — the multichannel case passes 2 instead of 1; new test that `PeriodicRandomStepFunction1D(1, 3)` raises |  |  |
| `src/context_flux_no/legacy/` — `models/`, `nn/` and `plotting/` moved here with `git mv`; the absolute imports of `context_flux_no.models` and `.nn` inside them name `legacy`; new `__init__.py` states rule R7 | | |
| `src/context_flux_no/metrics.py` — deleted: nothing imported it | | |
| `scripts/training/configs/model/` — every `_target_` names `context_flux_no.legacy.models.multiphysics.<Name>` | | |
| `tests/legacy/` — the tests of legacy code moved here from `tests/` and `tests/unit/test_key_hygiene.py`; imports name `legacy` | | |
| `tests/unit/test_run_record.py` — the `_target_` string names `legacy.models` | | |
| `pyproject.toml` — new `[tool.context_flux_no] clean` list; ruff `extend-exclude` and `force-exclude` and codespell `skip` cover `legacy/` and `tests/legacy/` | | |
| `scripts/train.py` — new: the one training entry point, built from `train_multiphysics.py`; the source comes from `hydra.utils.instantiate(cfg.data.source, ...)` and the loss arguments from `source.loss_args` | | |
| `scripts/training/train_the_well.py` — deleted: replaced by `scripts/train.py` with `data=well_euler_2d training=the_well` | | |
| `scripts/training/configs/config_the_well.yaml` — deleted: `data=well_euler_2d training=the_well` composes the same config | | |
| `scripts/training/configs/data/` — each data config gains a `source:` node naming the source class and holding its keyword arguments | | |
| `scripts/check_configs.py` — deleted: the census is `tests/unit/test_configs.py` (rule R10) | | |
| `scripts/import_all.py` — deleted: the import sweep is `tests/unit/test_imports.py` (rule R10) | | |
| `tests/unit/test_configs.py` — new: the config census, skipping `*_base` and pairing the five dimension-specific model configs only with data of their dimension | | |
| `tests/unit/test_imports.py` — new: every module of the package imports, `legacy/` included | | |
| `src/context_flux_no/data/sources.py` — new property `WellDatasetSourceBase.loss_args`, `(dt, *dxs)` from `metadata_common` | | |
| `scripts/training/README.md` — names `scripts/train.py`, the `source:` node and the census test | | |
| `README.md` — names `scripts/train.py` and the two tests that replace the census and import scripts | | |
| `src/context_flux_no/data/loader.py` — new: `make_loader`, the one loader pipeline, and `gc_paused`, which no driver uses yet | | |
| `src/context_flux_no/data/__init__.py` — exports `make_loader` and `gc_paused` | | |
| `src/context_flux_no/data/sources.py` — `WellDatasetSourceBase.__getitem__` names its argument `index` as the base protocol does; its `ty: ignore` removed | | |
| `src/context_flux_no/experiment.py` — module docstring names `scripts/train.py` instead of the deleted scripts | | |
| `scripts/train.py` — its local `make_loader` removed; the loaders come from `data.make_loader` | | |
| `scripts/bench_loader.py` — new: prints the host and device batches per second of `make_loader` on a named data config | | |
| `scripts/training/configs/data/` — `worker_count` removed: nothing read it | | |
| `scripts/training/configs/training/the_well.yaml` — the opening comment no longer names `train_the_well.py` | | |
| `tests/unit/test_loader.py` — new: the laws of `make_loader` and `gc_paused` on an in-memory source | | |
| `scripts/training/train_multiphysics.py` — deleted: replaced by `scripts/train.py`; git no longer reads the pair as a rename once the local `make_loader` is gone | | |
| `src/context_flux_no/evaluation/metrics.py` — new: the metric functions and containers of the rebuttal branch's `metrics.py` with lower-case names and equinox-style docstrings, without `ideal_gas_pressure` | | |
| `src/context_flux_no/evaluation/__init__.py` — new: exports the public names of `metrics` | | |
| `tests/unit/test_metrics.py` — new: the rebuttal branch's `tests/test_metrics.py` with the new names, without the `ideal_gas_pressure` test | | |
| `src/context_flux_no/simulations/pde/base.py` — new methods `AbstractPDEFamily.positive_quantities`, none by default, and `admissible`, true when every positive quantity is positive | | |
| `src/context_flux_no/simulations/pde/shallow_water.py` — `ParametrizedShallowWater1D.positive_quantities` returns the depth | | |
| `src/context_flux_no/simulations/pde/euler.py` — new method `Euler2D.pressure` with the rebuttal branch's ideal-gas formula; `positive_quantities` returns the density and the pressure | | |
| `src/context_flux_no/simulations/writer.py` — new `store_channel_order`, the permutation from the family's channel order to a store's | | |
| `tests/helpers.py` — new tolerance `PRESSURE_ROUND_TRIP_RTOL` | | |
| `tests/unit/test_admissible.py` — new: the laws of `admissible` and `Euler2D.pressure` | | |
| `tests/unit/test_generate_dataset.py` — `store_channel_order` checked against an `Euler2D` and a `CubicFlux1D` store written with `WellWriter` and read through `ZarrWellDatasetSource` | | |

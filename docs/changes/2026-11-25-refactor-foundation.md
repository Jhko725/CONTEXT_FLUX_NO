---
type: change
status: landed
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
training entry point, the data loader and the evaluation code. It is roadmap Phase 1b.
It closes items 4.5 and 6.1 and the entry-point half of 7.1, drops 0.7 and 5.4, and
removes 2g and 3c. It carries the commit of `workflow-bootstrap` (`d60a19a`), which is
not merged separately.

The work followed the plan `docs/plans/2026-11-25-refactor-foundation.md` in an
unattended session. Every unit of the plan landed. Units 8 and 9 each needed a second
attempt, and the choices that the plan did not make are listed under *Open issues*,
each marked as unconfirmed.

- **Workflow.** Every command is a recipe of the `justfile`, the hooks run through
  `prek`, and eight check scripts live under `scripts/checks/`.
  `docs/document-conventions.md` states the form of every document type, and `CLAUDE.md`
  describes the session protocol.
- **Removals.** The legacy checkpoint loader, the regression tier, the audit and
  backfill scripts, the xarray loader with its four dependencies, the notebooks and the
  ICON submodule are gone.
- **`legacy/` and the clean list.** `models/`, `nn/` and `plotting/` moved under
  `src/context_flux_no/legacy/`, and `pyproject.toml` lists the clean paths that the
  strict checks read (ADR-0012).
- **Training.** `scripts/train.py` trains every model on every dataset, and
  `data.make_loader` builds the training and evaluation loaders.
- **Evaluation.** `evaluation/` holds the metrics, the rollout evaluation, the
  summaries, the cost measurement and the record, and `scripts/evaluate.py` evaluates
  named runs on a named test set.
- **Strict checks.** Docstrings, complexity, `ty`, the container, dependency and
  docstring checks and runtime shape checks hold on the clean list, and every cited work
  is in `docs/references.bib`.

## Design

The first three points come from the change document of `workflow-bootstrap`, which
this document replaces: one branch has one change document.

- **The rules are copied into the repository.** A cloud session does not load the global `CLAUDE.md`, and until now the rules reached a session only through the Project. Rejected: a pointer to the Project, which a session started from the repository alone cannot follow.
- **The hooks are copied unchanged**, each with a first line naming its source and commit. The amend rule of `block_history_rewrite.py` asks the remote directly, which matters in the shallow clones of a cloud session.
- **`implementer.md` names `src/context_flux_no/legacy/` before that directory exists.** `refactor-foundation` creates it; the boundary is stated here so that the agent file does not change in the middle of the run that uses it.

The decisions below are Joon's, from the planning session of 2026-11-25; the plan's
section 10 gives the question behind each.

- **No compatibility with the ICLR checkpoints.** The regression tier is retired with no replacement, and the 0.7 audit and the 5.4 backfill are dropped. The rebuttal is over, the tag and the original repository archive the submission, and no phase before Phase 3 changes a model. Rejected: keeping the shim through Phase 3, which made every refactor wait on checkpoints that only the cluster holds. Recorded in ADR-0008 and ADR-0010.
- **Code that awaits its rewrite lives in `legacy/`, and a clean list scopes the strict checks.** One standard for the whole tree had kept the strict checks off everywhere. Rejected: per-tool ignore lists, which hold the scope once per tool. Recorded in ADR-0012.
- **The tests of legacy code run unchanged and leave with the code.** They moved to `tests/legacy`, outside the clean list, and they show that `legacy/` still imports and trains. Rejected: holding them to the strict checks, which would mean editing tests of code that is about to be replaced.
- **The workflow of `UniversalForceReconstruction`.** The rules, the implementer agent, plans, round tables, unconfirmed markers, `just` and `prek` are in the tree, because an unattended run needs its rules there and its decisions listed. Rejected: keeping `pre-commit` and the *Files* section, which would leave two workflows across Joon's repositories.
- **Only the general half of the rebuttal's evaluation code is ported.** The question-specific half (context refresh, effective flux, cubic wave-speed coverage) is tied to models that Phase 3 replaces, and it stays on the rebuttal branch.
- **The two-stage contract is a requirement on Phase 3, not code now.** `generate_solver` and `step` have no user in this repository once the question-specific half stays behind. Recorded in ADR-0006.
- **`evaluation/` reaches a model through a `Protocol` and imports nothing from `legacy/`.** The accelerator code (`rollout.py`) and the host code (`summary.py`) are separate modules, and the package survives Phase 3 unchanged. Rejected: importing `AbstractMultiphysicsOperator`, which would tie the package to a class that Phase 3 deletes.
- **`scripts/evaluate.py` selects runs through `runs.index()` and writes HDF5 with a JSON record.** The rebuttal's presets encoded the paths of ICLR checkpoints.
- **Admissibility is a method of the family.** `positive_quantities` and `admissible` are family knowledge, like the conversion to conserved variables. Rejected: the free functions of the rebuttal branch, which took the equation of state as an argument. The two channel orders of Euler are finding F16.
- **One `scripts/train.py`; the config census is a test.** The two old scripts differed in the source class and the loader only, so the data config's `source:` node now picks the class.
- **One `make_loader`.** The `grain.DataLoader` path of the Well script is gone, since two ways to build a loader is one too many.
- **The evaluation tests use stub models.** A stub has an answer that can be computed by hand (rule R11).
- **One bibliography.** `docs/references.bib` holds every cited work, and citations use the key form. Phases 2b to 2e add many citations.
- **The ICON submodule and the notebooks are deleted.** Nothing imports the submodule, and the notebooks read ICLR-era data.
- **The roadmap keeps its phase numbers**, because the merged change documents cite them. This branch is Phase 1b.

The plan's writer made ten assumptions without Joon, W1 to W10 in its section 2. The
implementation follows each of them, and each awaits confirmation.

- **W1. The workflow files come from `workflow-bootstrap`.** The branch starts from that commit. [unconfirmed]
- **W2. The split into units and each unit's model** were followed as planned. [unconfirmed]
- **W3. The config census pairs a dimension-specific model config only with data of its dimension** and skips the `_base` configs. No config was deleted; the census has 52 pairs. [unconfirmed]
- **W4. The garbage-collection policy is not in the loader factory.** `data/loader.py` provides `gc_paused`, and no driver calls it yet (see *Open issues*). [unconfirmed]
- **W5. Existing ADRs were converted to the template by moving text.** [unconfirmed]
- **W6. The change documents dated before 2026-11-25 are exempt from the heading rule** through `PRE_FORMAT_CHANGES`. [unconfirmed]
- **W7. The config census and the import sweep are tests**, since both import the package (rule R10). [unconfirmed]
- **W8. Evaluation reads a store's channels in the store's order** and converts through `store_channel_order`. [unconfirmed]
- **W9. `evaluation/cost.py` keeps `count_flops` and `measure` only.** [unconfirmed]
- **W10. The clean list is `[tool.context_flux_no] clean` in `pyproject.toml`.** [unconfirmed]

## Bugs fixed

- **A periodic step initial condition could have one jump.** A periodic step with one jump is a constant state, for which relative errors are undefined. `PeriodicRandomStepFunction1D` now defaults to at least 2 jumps and raises `ValueError` below 2. The two step configs used the default, so their sampled jump count moves from the range 1 to 5 to the range 2 to 5.
- **The second DISCO bottleneck convolution reused the first one's key.** `legacy/models/multiphysics/disco/operatornet.py` now passes `keys_bot[1]`. The fix was parked for the ICLR DISCO checkpoints (F15), which are no longer loaded.
- **`RunRecord.read` accepted any status.** It now raises on a status other than `running`, `finished` or `failed`.
- **Annotations that the runtime shape checks exposed.** With the import hook on, 41 tests of the fast tier failed, each on an annotation that did not describe what the code receives. `WellDatasetSourceBase.__getitem__` declared a JAX array with the axes in the wrong order and returns a NumPy array with axes `time channel *spatial`. `x_spans` and `t_span` were declared `float` and receive the ints of the YAML configs. `num_parameters` declared a module and receives a tuple of layers from DISCO. The Riemann-solver arguments `aux_l` and `aux_r` are `None` when PyClaw calls them.
- **A missing compiler analysis in `evaluation/cost.py`** gave an `AttributeError` on `None`. A missing cost analysis now gives NaN FLOPs, as the docstring said, and a missing memory analysis raises `RuntimeError`.

## Tests

The full tier ran once at `6b85ca9`, the last commit that changes code, in a cloud
container with 2 cores, no GPU and no Julia, under `JAX_PLATFORMS=cpu`.

- **`uv run just test default`: 300 passed, 1 skipped, 7 deselected.** The skip is `test_describe_records_the_julia_side_versions`, which needs Julia. The 7 deselected tests are the slow tier.
- **`uv run just test-slow dev`: 1 passed, 6 skipped.** The 6 skipped tests are the Trixi cross-code tests, which need Julia.
- **`uv run just check` exits 0** at the commit that adds this text.
- **The baseline at `450f31d`** was 107 passed, 1 skipped, 14 deselected and 1 xfailed. The xfail was the DISCO case of the key-hygiene test, which now passes. Unit 4 deleted the regression tests and the tests of the legacy loader and the backfill script.

What the new tests pin:

- **`tests/unit/test_configs.py`** (53 tests): every model config instantiates with every data config of its dimension, and the dimension table names existing configs.
- **`tests/unit/test_imports.py`**: every module of the package imports on CPU, `legacy/` included.
- **`tests/unit/test_loader.py`** (6): every index appears once per epoch, two loaders with one seed give the same batches, the evaluation order yields `len(source) // batch_size` batches in index order, and `gc_paused` enables the collector again after an exception.
- **`tests/unit/test_metrics.py`** (14), **`test_rollout.py`** (9) and **`test_summary.py`** (9): the evaluation code against hand-computed values and stub models (V26 to V28 in `docs/verification.md`).
- **`tests/unit/test_admissible.py`** (4) and two tests of **`test_generate_dataset.py`**: pressure, admissibility and the store's channel order (V29, V30).
- **`tests/unit/test_cost.py`** and **`test_evaluate.py`** (3 together): the FLOP count, and `scripts/evaluate.py` run end to end on a run trained for a few steps on a synthetic store.

A reviewer with a fresh context compared `evaluation/` with the rebuttal branch
(`upstream/joon` at `4fe3323`) function by function, on the same random inputs and with
injected NaNs. Every ported function gave bitwise equal results. The Euler pressure
minima are the exception, for the two reasons listed under *Open issues*.

Not run in this session: anything that needs a GPU, Julia, a Well store or the wandb
service. `scripts/bench_loader.py` ran with `--help` only, and the Well path of
`scripts/train.py` was not run.

## Open issues

Choices that the plan did not make are marked as unconfirmed. `uv run just pending`
lists them.

**The run**

- **The agent type.** The session started outside the repository and offered no `implementer` type. Each unit ran as a general-purpose subagent on the unit's model, briefed with the *Boundary*, *Stop rule*, *Before and after* and *Report* sections of `.claude/agents/implementer.md` and told not to run `git push`, since the hook of the agent file does not run for that type (plan section 4, step 3). The gates of the plan's section 9 applied, so the session did not stop for review between units.
- The hooks have no automated test in this repository. `UniversalForceReconstruction` has none either.
- **Six change documents predate the format, where the plan says eight.** `docs/changes/` at `450f31d` holds six documents dated before 2026-11-25, and `PRE_FORMAT_CHANGES` lists those six. [unconfirmed]
- The item-3 grep of unit 1 filters `^./docs/changes/`, which GNU grep's output never matches; with `^docs/changes/` it prints nothing.

**Workflow and documents (units 1 to 3, 17)**

- **The new hooks rewrote whitespace in 45 tracked files** (unit 1): trailing spaces and final newlines, among them files that unit 7 moved under `legacy/`. `git diff --ignore-space-at-eol` shows no other change. The alternative was a wider `exclude` in `prek.toml`. [unconfirmed]
- **`tests/conftest.py` arrived in unit 1 with the Hypothesis profiles only.** The plan creates it in unit 16, but `just test dev` needs the `dev` profile from unit 1 on. Unit 16 added the import hook. [unconfirmed]
- **ADR conversion (unit 2, W5).** No ADR had *Forces* text, so each `### Forces` holds `*[not recorded]*`, and the template allows that marker. *History* opens with an "accepted" line dated from the ADR's status line. The extra sections of ADR-0001, 0004, 0010 and 0011 follow *Decision* in their original order. Compared as multisets of lines, no sentence of an ADR changed in that unit. [unconfirmed]
- **The document types `guide`, `format` and `dataset` and the status `historical` are not in the vocabulary** (unit 2): no document here uses them. [unconfirmed]
- **`check_docs.py` has no architecture rule.** The plan asked to keep the old check's rule, and the old check had none; the rule of `UniversalForceReconstruction` reports 19 problems on the fenced module map of `docs/architecture.md`. [unconfirmed]
- **The glossary stays a bullet list.** `docs/document-conventions.md` describes the table form, which the glossary takes when it is next rewritten. [unconfirmed]
- **The bibliography holds only what the repository states** (unit 17). Its 19 entries have surnames without first names, 13 have no title, and two author lists end in `and others`. The generator was changed to cite such a list as `et al.` and to show a technical report's institution and number. [unconfirmed]
- **Two citations are not in key form.** ADR-0002 cites "Roe (1991)" and ADR-0004 cites `arXiv 2402.05193`; the repository gives no more about either work, and a key would need a reworded sentence or an invented author.
- **`docs/index.md` links `assessment-2026-09-26.md`, which was never committed here.** The row carries an ignore marker for the link check. Joon decides whether the file is added or the row removed.
- **Eight works are cited only under `legacy/`** and are not in the bibliography, since the check does not read that directory: Beck et al. 2022, Morel et al. 2025 (DISCO), Raffel et al. 2020, Guibas et al. 2021, Hao et al. 2024 (DPOT), De et al. 2024 (Griffin), Patraucean et al. 2025 (TRecViT) and Beyer et al. (arXiv:2205.01580).

**Training and data (units 8 and 9)**

- **`TheWellDataSource` scales its metadata by `downsample_spatial` and delivers the native grid** (finding F17, found in unit 8). `data/sources.py` multiplies `spatial_resolution` by the factor, and no `__getitem__` slices the data; commit `41ac847` dropped the slice that `534503f` had added. `well_euler_2d` sets the factor to 8, so a Well run sees the native grid with the spacings of a grid 8 times coarser. The branch does not fix this: restoring the slice or the spacings changes what every Well run trains on, and that is Joon's decision. The first attempt at unit 8 stopped on it. [unconfirmed]
- **The loss arguments of a Well run now come from the store's metadata** (unit 8): `dt` from the time grid, and each spacing as the native spacing times `downsample_spatial`. They equal the old constant `(0.015, 1 / 64, 1 / 64)` only if the store's time step is 0.015 and its native spacing is 1/512. No Well store was in the session, so this was not checked. [unconfirmed]
- **`gc_paused` is built and tested, and no driver uses it** (unit 9, W4). No committed revision of the repository pauses the collector, so no period exists to keep, and `Trainer.train`, not `experiment.train`, owns the step loop; a per-step hook would change a signature in `training/`. The period and the place of the call are left to Joon. The first attempt at unit 9 stopped on this. [unconfirmed]
- **`worker_count` is removed from the data configs** (unit 9): the one loader factory has no workers, and nothing read the key once the `grain.DataLoader` path of the Well script was gone. A Well run therefore reads on the host without worker processes. [unconfirmed]

**Evaluation (units 10 to 13)**

- **`scripts/evaluate.py` passes no positive quantities** (unit 13). A store's `DatasetRecord` names the family's class only, the family's parameters vary per trajectory, and a source with `use_normalization` returns z-scored frames. `evaluate_rollouts` accepts the callable, and `positive_quantities` and `store_channel_order` exist and are tested, but the command does not join them. [unconfirmed]
- **The pressure minima of a test set with several values of `gamma` cannot be computed in one call** (found by the reviewer). The rebuttal branch passed `gamma` per trajectory; here `positive_quantities` is the bound method of one `Euler2D(gamma)`. A caller must group the trajectories by `gamma`, or the callable needs a per-trajectory form, which is a design choice for Joon. [unconfirmed]
- **`Euler2D.pressure` differs from the rebuttal's `ideal_gas_pressure` by one unit in the last place.** `gamma` is a Python float here, so `gamma - 1` is rounded once from float64, where the rebuttal branch subtracted in float32. The relative difference is at most 1.3e-7.
- **Choices of `scripts/evaluate.py` that the plan did not make** (unit 13). A run id is looked up under `runs/` relative to the working directory. The source is built twice, first with a window of 1 to read the trajectory length, so that a record is one whole trajectory. The loader drops an incomplete last batch, and the script prints how many trajectories that leaves out. The record is written beside `eval.h5` as `eval.json`, and the script refuses to overwrite either file. `EvaluationRecord.read` raises on an unknown key. [unconfirmed]
- **`valid_horizon` checks `l2` for non-finite values and not `l2_identity`** (found in unit 10, ported unchanged). A NaN in the baseline makes the comparison false, so the horizon runs past that step. No test pins either behavior.
- **`ATOL` and `RTOL` of `tests/unit/test_metrics.py` are module constants**, as on the rebuttal branch, where the working rules put named tolerances in `tests/helpers.py`.

**Strict checks (units 14 to 16)**

- **`simulations/pde/burgers.py` is excluded from `ty`** (unit 15). `ViscousBurgers1D.solve` returns its cell centers as a tuple, and `AbstractPDEFamily.solve` declares a list. `CubicFlux2D.solve` returns a tuple too. Widening the base to `Sequence[np.ndarray]` or returning a list both change a public return type, which the plan's section 9 leaves to Joon.
- **`ty` counted more diagnostics than the plan recorded**: 70 on the package's clean paths at the start of unit 15 (the plan: 59 at `450f31d`) and 87 on the tests and scripts at the start of unit 16 (the plan: 75). The new modules of units 8 to 13 account for the difference.
- **Unit 15 changed code to satisfy `ty`, beyond annotations.** `AbstractPDEFamily` declares `n_spatial_dims` and `field_rank_names` with `eqx.AbstractClassVar`, since every family defines them as class variables. `WellWriter` reads store attributes through readers that raise `TypeError` on an unexpected type, `Run.create` and `Run.open` check their config container types, `OrbaxCheckpointer.best` checks that each stored metric is a number, and `build_wandb_run` raises `ValueError` on an unknown mode. The one-step losses bound their model type by a new `OneStepPredictor` protocol. [unconfirmed]
- **Casts and one dynamic lookup remain** (units 15 and 16). Six casts in `src/` and about twenty in the tests tell `ty` the class of an `eqx.Module` instance, which `ty` 0.0.84 types as `Module` because equinox's metaclass call is unannotated; two more cast gradients to `optax.Params`, and one sets clawpack's `output_format` to `None`. `backends/trixi.py` reads `juliacall.Main` with `getattr`, since PythonCall adds the attribute at load time. Each carries a comment. [unconfirmed]
- **Two tests of an unknown option accept a second error** (unit 16). With the import hook on, beartype rejects an invalid `Literal` before the `ValueError` guard runs, so `test_pyclaw_backend.py` and `test_trixi_backend.py` accept `ValueError` or `jaxtyping.TypeCheckError`, and no test reaches the two guards any more. [unconfirmed]
- **`custom_types.Real = float | int`** is a new public alias (unit 16), used where a config passes an int for a span. [unconfirmed]
- **`omegaconf` is imported and not declared** (unit 15). It arrives through `hydra-core`, and `check_dependencies.py` lists it as a known exception with that reason. [unconfirmed]
- **Docstrings that describe a numerical choice no document records** (unit 14). Each says what the code does: the `1e-14` cut-offs and the diffusive limit `0.5 dx**2 / b` of `ViscousBurgers1D._stable_time_step`, the `1e-12` speed cut-off of `CubicFlux2D.solve`, the normalization of `TruncatedFourier1D` as a mean over modes divided by `sqrt(2)`, and the noise of `DenoisingOneStepLoss` as `noise_scale` (default `5e-5`) times the frame's norm. [unconfirmed]
- **`ViscousBurgers1D.solve` was split for complexity and has no test.** The split version was compared with the old one on three `(a, b)` settings and gave bit-identical output.
- **The module docstring of `initial_conditions/fourier.py` says the series lives on `[x0, x0 + L)`**, and the code evaluates it in `x` with no shift by `x0`. Left as found.
- **`_jsonable` is defined twice**, in `simulations/generate.py` and `backends/pyclaw.py`. Left as found.
- **beartype prints 68 deprecation warnings** in the fast tier, about `typing.Iterator` and `typing.MutableMapping` hints in clean modules. They are not errors.

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
| `src/context_flux_no/evaluation/rollout.py` — new: the `Operator` protocol and the rebuttal branch's rollout and segment drivers, with `positive_quantities` in place of `minima_fn` and `coefficients` | | |
| `src/context_flux_no/evaluation/summary.py` — new: the rebuttal branch's `summarize_rollouts`, `summarize_conservation_drift`, `match_trajectories` and `mean_by_bin` | | |
| `src/context_flux_no/evaluation/__init__.py` — exports the public names of `rollout` and `summary` | | |
| `tests/helpers.py` — new stub operators `HoldLastFrame` and `ShiftByOneCell`, and the tolerance `ROLLOUT_ATOL` | | |
| `tests/unit/test_rollout.py` — new: the rebuttal branch's segment tests on the stubs, and closed-form checks of `evaluate_rollouts` and `evaluate_references` | | |
| `tests/unit/test_summary.py` — new: the rebuttal branch's matching, binning and summary tests, and a check of `fraction_nonpositive` | | |
| `src/context_flux_no/evaluation/cost.py` — new: `Measurement`, `count_flops` and `measure` from the rebuttal branch's `cost.py`, without the model-specific breakdowns | | |
| `src/context_flux_no/evaluation/record.py` — new: `EvaluationRecord`, the JSON record of an evaluation with its runs, dataset, split, arguments and provenance | | |
| `src/context_flux_no/evaluation/__init__.py` — exports the public names of `cost` and `record` | | |
| `scripts/evaluate.py` — new: evaluates named runs on a split of a named data config and writes one HDF5 file and its `EvaluationRecord` | | |
| `tests/helpers.py` — new helpers `point_hydra_output_dir_at` and `make_dataset_record`, moved from the two test files below | | |
| `tests/unit/test_experiment.py` — its fixture uses `point_hydra_output_dir_at` from `tests/helpers.py` | | |
| `tests/unit/test_generate_dataset.py` — uses `make_dataset_record` from `tests/helpers.py` in place of its local `_record` | | |
| `tests/unit/test_cost.py` — new: the rebuttal branch's FLOP-count and measurement tests on a matrix product | | |
| `tests/unit/test_evaluate.py` — new: trains a tiny FNO on a synthetic store and checks the result file and record of `scripts/evaluate.py` | | |
| `pyproject.toml` — ruff selects `W605`, `D100` to `D103` and `C901` at complexity 8, as UFR's does, and exempts every path outside the clean list from the `D` rules | | |
| `src/context_flux_no/` — a docstring on every public module, class, method and function of the clean paths; `runs/run.py` and `training/trainer.py` name `scripts/train.py` in place of the deleted training script | | |
| `src/context_flux_no/runs/record.py` — `index_from_config` reads the config through the new module-level `_config_value` and `_peak_learning_rate` | | |
| `src/context_flux_no/simulations/pde/burgers.py` — `ViscousBurgers1D.solve` steps through the new `_march_to_save_times`, which takes its step size from the new `_stable_time_step` | | |
| `src/context_flux_no/simulations/writer.py` — `WellWriter.create` writes its groups through the new `_write_dimensions`, `_write_boundary_conditions` and `_create_field_arrays` | | |
| `src/context_flux_no/training/trainer.py` — `Trainer.train` uses the new `_initial_state`, `_check_loop_lengths`, `_first_batches` and `_logged_validation_metrics` | | |
| `tests/` — a docstring on every test function and on the test modules and script that lacked one, each stating what it checks | | |
| `scripts/` — a docstring on every `main`; `training/configs/training/the_well.yaml` names `experiment.state_template` | | |
| `src/context_flux_no/simulations/pde/base.py` — `n_spatial_dims` and `field_rank_names` are `eqx.AbstractClassVar`, the class variables that every family defines | | |
| `src/context_flux_no/simulations/pde/burgers.py` — `ViscousBurgers1D` annotates its flux and step on JAX arrays, `bc` in the base's type and the returned centers as the tuple they are; its private step function has a docstring | | |
| `src/context_flux_no/simulations/pde/cubic.py` — `CubicFlux2D.solve` takes `x_spans`, `Nxs` and `bc` in the base's types; its private step functions have docstrings | | |
| `src/context_flux_no/simulations/backends/base.py` — `normalize_bcs` narrows each boundary-condition name with the new `_is_bc_name` | | |
| `src/context_flux_no/simulations/backends/pyclaw.py` — the output times go to PyClaw as an array and `output_format=None` through a type-level `cast`; private functions have docstrings | | |
| `src/context_flux_no/simulations/backends/trixi.py` — reads `juliacall.Main`, which PythonCall adds at load time, by name | | |
| `src/context_flux_no/simulations/writer.py` — `WellWriter` reads store attributes through the new `_int_attr`, `_sequence_attr`, `_names_attr` and `_field_rank_names_attr`, and `finalize` takes a `Status` | | |
| `src/context_flux_no/simulations/generate.py` — its private functions have docstrings | | |
| `src/context_flux_no/simulations/initial_conditions/base.py` — `_check_dims` has a docstring | | |
| `src/context_flux_no/data/loader.py` — imports `jax.extend`, which it uses | | |
| `src/context_flux_no/data/sources.py` — the storage protocol's `attrs` is read-only, `filter_paths` takes any path type, `__getitem__` declares the NumPy array it returns; private functions have docstrings | | |
| `src/context_flux_no/evaluation/cost.py` — a missing cost analysis gives NaN FLOPs and a missing memory analysis raises `RuntimeError` | | |
| `src/context_flux_no/evaluation/metrics.py` — `total_variation` sums without an integer start, and the three result constructors return through a type-level `cast` | | |
| `src/context_flux_no/evaluation/rollout.py` — the spectrum sums start from zero-dimensional arrays | | |
| `src/context_flux_no/evaluation/summary.py` — reads the reference errors as NumPy arrays | | |
| `src/context_flux_no/experiment.py` — `build_wandb_run` raises `ValueError` on an unknown wandb mode | | |
| `src/context_flux_no/runs/run.py` — `Run.create` and `Run.open` check the container types of the model config and the loaded config; `_flatten` has a docstring | | |
| `src/context_flux_no/runs/record.py` — `_tail` has a docstring | | |
| `src/context_flux_no/training/checkpoint.py` — `OrbaxCheckpointer.best` checks that each stored metric is a number | | |
| `src/context_flux_no/training/loss.py` — the one-step losses bound their model type by the new `OneStepPredictor` protocol | | |
| `src/context_flux_no/training/trainer.py` — no `ty: ignore`; the optimizer update takes the model through a type-level `cast` to optax's `Params`; private steps have docstrings | | |
| `src/context_flux_no/utils/serialization.py` — `_default` has a docstring | | |
| `scripts/checks/check_containers.py` — new, from UFR: no `jax.Array` in a frozen dataclass or a static field of the clean paths under `src` | | |
| `scripts/checks/check_dependencies.py` — new, from UFR without its unit-library rule: declared packages, R12, no clean path imports from `legacy/`, `src` imports no scripts or tests | | |
| `scripts/checks/check_docstrings.py` — new, from UFR: header form, private docstrings, module length and plan labels on the clean paths | | |
| `prek.toml` — registers `check-docstrings`, `check-containers` and `check-dependencies` | | |
| `justfile` — `check` runs `ty` on the clean paths under `src` except `burgers.py`, then greps every clean path for `ty: ignore` | | |
| `tests/conftest.py` — installs `jaxtyping.install_import_hook` with `beartype.beartype` on the clean modules under `src/`, read from `pyproject.toml`, before any package import | | |
| `pyproject.toml` — `beartype` in the dev group | | |
| `uv.lock` — locks `beartype` | | |
| `justfile` — `check` runs `ty` on every clean path except `burgers.py`; the unused `clean_src` is removed | | |
| `src/context_flux_no/custom_types.py` — new alias `Real`, a `float` or an `int`, since beartype does not let `float` admit an `int` | | |
| `src/context_flux_no/simulations/` — `x_spans` and `t_span` take `Real` ends, as the YAML configs write them; the Riemann solvers of `cubic.py` and `shallow_water.py` take the `None` auxiliary arrays that PyClaw passes; `TruncatedFourier1D` takes NumPy or JAX coefficients | | |
| `src/context_flux_no/data/sources.py` — `HierarchicalStorage` is runtime-checkable, and `loss_args` declares the NumPy scalars it returns | | |
| `src/context_flux_no/evaluation/rollout.py` — `Operator` is runtime-checkable | | |
| `src/context_flux_no/evaluation/metrics.py` — `valid_horizon` declares that its arrays broadcast against each other | | |
| `src/context_flux_no/training/loss.py` — `AbstractTrainLoss` and `OneStepPredictor` are runtime-checkable | | |
| `src/context_flux_no/training/trainer.py` — `TrainerState.take_step` returns `TrainerState[M]` in place of `Self`, which beartype checks only on a decorated class | | |
| `src/context_flux_no/utils/arrays.py` — `num_parameters` takes any pytree, as DISCO calls it on a tuple of layers | | |
| `scripts/data/generate_dataset.py` — checks that `bc` is a name or a list before passing it on | | |
| `scripts/evaluate.py` — a run id without an index is unresolved before `runs.select` is called | | |
| `tests/helpers.py` — `make_dataset_record` takes named arguments, and `SHARPCLAW_SSP33` is a `TypedDict` | | |
| `tests/unit/` — passes `ty`: equinox instances through a type-level `cast`, zarr members and optional record fields narrowed by assertion; `test_summary.py` stacks per-rollout references; the two unknown-option tests accept the hook's `TypeCheckError`; `test_experiment.py` patches Hydra through `point_hydra_output_dir_at` | | |
| `tests/verification/test_crosscode.py` — `CubicFlux2D` through a type-level `cast` | | |
| `docs/references.bib` — new: the one bibliography, nineteen works with only the fields the repository states | | |
| `docs/references.md` — new, generated from `references.bib` by `just references` | | |
| `scripts/docs/generate_references.py` — new, from UFR: an author list ending in `and others` gives the `et al.` key, and a technical report shows its institution and number | | |
| `scripts/checks/check_references.py` — new, from UFR: every citation key and relative link resolves and `references.md` is current; skips `legacy/` beside `changes/` and `plans/` | | |
| `prek.toml` — registers `check-references` | | |
| `justfile` — new recipe `references` | | |
| `pyproject.toml` — `bibtexparser` in the dev group | | |
| `uv.lock` — locks `bibtexparser` | | |
| `docs/document-conventions.md` — writes § References: the key form, the generated page, what the check reads | | |
| `docs/index.md` — lists `references.md`; the link to the never-committed assessment carries `references:ignore` | | |
| `docs/design/fv-conventions.md` — citations in the key form; the *Sources* list names keys | | |
| `docs/design/fv-core-plan.md` — citations in the key form, in the frontmatter, the reading path and the *Sources* list | | |
| `docs/decisions/` — ADR-0002, 0003, 0005 and 0008: each citation token in the key form, the sentences unchanged | | |
| `README.md` — `references:ignore` on the two lines whose year is not a citation (ICLR 2027, Oct 2026) | | |
| `docs/coding-standards.md` — rewritten after `UniversalForceReconstruction`: tiers, test contracts, scopes, rules R2 to R12, checks, deliberate exceptions |  |  |
| `docs/decisions/0012-legacy-directory-and-clean-list.md` — new: `legacy/` and the clean list |  |  |
| `docs/decisions/0006-operator-hierarchy-and-scale.md` — amended: the two-stage contract is part of the decision; the ICLR shim is dropped |  |  |
| `docs/decisions/0008-four-tier-testing.md` — amended: the regression tier is removed; citation tokens in key form (unit 17) |  |  |
| `docs/decisions/0010-run-identity.md` — amended: legacy records, the backfill and the legacy loader are removed |  |  |
| `docs/roadmap.md` — principles 1 to 3 reworded; 0.7 and 5.4 dropped; 2g and 3c removed; Phase 3 re-scoped; Phase 1b added; 4.5, 4.6, 6.1 and 7.1 updated; two parked entries closed; a Done row; check paths (unit 1) |  |  |
| `docs/architecture.md` — module map after Phase 1b; tests section; ADR-0012 in the decision table; frontmatter (unit 2) |  |  |
| `docs/findings.md` — F16 (the two channel orders of Euler) and F17 (Well downsampling); closing lines for F1 and F15; frontmatter (unit 2) |  |  |
| `docs/verification.md` — V22 retired; V24 at its new path; V26 to V30 for the evaluation, admissibility and channel-order properties; frontmatter (unit 2) |  |  |
| `docs/glossary.md` — the terms clean list, legacy, admissible and evaluation record; frontmatter (unit 2) |  |  |

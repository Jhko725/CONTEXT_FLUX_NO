---
type: reference
status: current
updated: 2026-11-25
sources:
  - "docs/plans/2026-11-25-refactor-foundation.md, sections 5 and 8"
  - "UniversalForceReconstruction docs/coding-standards.md at b80a341"
---

# Coding standards

This document says where this repository enforces the conventions for code, tests and
docstrings, and which checks stand in for prose. The conventions themselves, with the
reasoning behind them, are in the `code-conventions` skill, and the pervasive ones are in
`.claude/rules/working-rules.md` and `.claude/rules/python.md`. The conventions for the
documents under `docs/` are in [`document-conventions.md`](document-conventions.md). Read
this document when a task touches tiers, scopes, rules or checks.

## Tiers

The commands are the recipes of the `justfile` (`uv run just --list`), and CI runs the
same recipes. The Hypothesis profiles are `dev` (10 examples), `default` (50) and `ci`
(100, derandomized); `tests/conftest.py` registers them.

| Purpose | Recipe |
|---|---|
| Fast, while editing | `just test dev <test files of the touched modules>` (`slow` tests excluded) |
| Full, once at the end | `just test default`; CI runs `just test ci` |
| Slow tier | `just test-slow [profile]`: the tests marked `slow` |
| Every hook, then types | `just check`: the `prek` hooks on every tracked file, then `ty` on the clean scope, then a grep for `ty: ignore` there |
| What awaits Joon | `just pending`: every unconfirmed marker under `docs/` with its file and line |
| The bibliography page | `just references`: writes `docs/references.md` from `docs/references.bib` |

`prek run --all-files` sees only the files that git tracks, so a new file is staged
before `just check` runs. `pytest-xdist` is not a dependency, so the tests run in one
process. A cloud session sets `JAX_PLATFORMS=cpu`; PyClaw needs `gfortran` at `uv sync`.

There is no regression tier. It compared the ICLR checkpoints with goldens, and the
branch `refactor-foundation` removed it together with the code that loaded those
checkpoints (ADR-0008, amended 2026-11-25).

## Test contracts

Each tier promises one thing, so that a new test has one obvious home (ADR-0008).

| Tier | Directory | Contract |
|---|---|---|
| unit | `tests/unit` | fast, deterministic, no external data; each test is a law at a contract of a module, not at its internals |
| verification | `tests/verification` | every test carries a `docs/verification.md` row id in its docstring and compares against an independent oracle; it runs under `jax_enable_x64` (its conftest) and is marked `slow` only when it is slow |
| legacy | `tests/legacy` | the tests of the code under `legacy/`, unchanged; they run in the fast tier and leave with the code they test |

Tolerances and solver settings are named entries with a one-line reason in
`tests/helpers.py`, and a test file tests one module or one claim. The property tier of
ADR-0008 (`tests/property`) arrives with the finite-volume core.

Backend goldens (roadmap Phase 2a) are bit-identity: a reference solver's output is
pinned exactly, and `docs/design/simulation-backends.md` records the policy and the
measured agreement ratios. A golden is never regenerated without a change-document
entry that names the commit that changed behavior and why.

## Scopes

The strict checks apply to listed clean paths (rule R7). The list is held once, as
`clean` under `[tool.context_flux_no]` of `pyproject.toml`: the subpackages
`simulations/`, `runs/`, `utils/`, `data/`, `training/` and `evaluation/`, the modules
`experiment.py` and `custom_types.py`, `tests/unit`, `tests/verification`,
`tests/helpers.py`, `tests/conftest.py` and `scripts`. The check scripts, the `ty` call
of `just check` and the jaxtyping import hook of `tests/conftest.py` read the list with
`tomllib`. Everything else in the package is under `src/context_flux_no/legacy/`, which
ruff and `codespell` exclude by that one path, with `tests/legacy` beside it.

## Rules

The rule numbers are those of `UniversalForceReconstruction`, so a number means the same
thing in both repositories. The rules that exist only there (units, float64, the
dimensionless equation of motion, declared dimensions) have no entry here.

- **R2. Containers.** A value that a JAX transform sees is an `eqx.Module`. A value that
  stays out of tracing is a frozen `@dataclass` that holds NumPy or Python values, never
  a `jax.Array`. A result is a fixed-shape pytree that carries what downstream needs.
  `scripts/checks/check_containers.py` checks the rule on the clean scope.
- **R6. Copied files name their source.** A file copied or adapted from a reference
  repository opens with one comment line that gives the repository, the path and the
  commit.
- **R7. The strict checks apply to listed clean paths.** The list is the `clean` list of
  § Scopes. Everything else in the package is under `src/context_flux_no/legacy/`, which
  stays importable and otherwise untouched until the branch that rewrites it. A module
  joins the clean list by leaving `legacy/` and never returns (ADR-0012).
- **R8. American English** in code, identifiers, docstrings and documents.
- **R10. A check reads; a test executes.** A check script reads text or syntax and never
  imports the package. A test executes the package. The config census and the import
  sweep are therefore tests (`tests/unit/test_configs.py`, `tests/unit/test_imports.py`).
- **R11. A test pins a property of our code that could be wrong.** It never pins the
  behavior of JAX, Equinox, grain or NumPy. A formula is not tested against a second
  copy of the same formula; it is tested by a property or against a value computed by
  hand. The evaluation tests use stub models for this reason: a stub that shifts the
  state by one cell per step has zero error against a translating wave.
- **R12. `evaluation/` imports nothing from `legacy/`.** A model reaches it through the
  `Operator` protocol of `evaluation/rollout.py`. No other clean path imports `legacy/`
  either; `scripts/checks/check_dependencies.py` checks both.
- **`copy.replace`, never `dataclasses.replace`.** The Python floor is 3.13, and ruff's
  `TID251` bans the older call.

## Persistence verbs

One verb pair serves each kind of thing, so that a reader can guess the method name.

| Thing | Verbs | Examples |
|---|---|---|
| a record or document file (JSON, YAML) | `write(path)` and the classmethod `read(path)`; `from_dict` for an in-memory dict | `RunRecord`, `DatasetRecord`, `EvaluationRecord` |
| a directory-like object | the classmethod `open(path)`; `create` or `from_config` to make a new one | `Run.open`, `WellWriter.create` |
| checkpointed state through orbax | `save(step, state)` and `restore(template, step)` | `OrbaxCheckpointer`, `Run.restore` |
| JSON encoding | `utils.serialization.dumps`; decoding is `json.loads` and `from_dict` | the records, zarr attributes |

A dataclass has no `to_json`, `from_json` or `to_dict` wrapper: `dataclasses.asdict` and
`copy.replace` are called directly.

## Checks

The configuration lives in `prek.toml` and in the tool sections of `pyproject.toml`,
with a comment there on each setting that is not obvious. This table says what each
check is for; the files have the current values. Every hook of `prek.toml` runs in
`just check` and in CI.

| Hook or check | Rule it enforces | Scope |
|---|---|---|
| `trailing-whitespace` | no trailing whitespace | every tracked file outside the excluded paths |
| `end-of-file-fixer` | every file ends in one newline | as above |
| `check-added-large-files` | no large file is added by accident | as above |
| `check-merge-conflict --assume-in-merge` | no conflict marker in any file, in or out of a merge | as above |
| `ruff-check` | lint: `E`, `F`, import order, no invalid escape (`W605`), the banned `dataclasses.replace` (`TID251`), docstrings present (`D100` to `D103`), at most 8 decision points in one function (`C901`) | Python outside `legacy/` and `tests/legacy`; the `D` rules on the clean scope only |
| `ruff-format` | one formatting | as `ruff-check` |
| `codespell` 2.4.3 | American English, irregulars and typos | outside `docs/changes/`, the legacy scope and the Julia `Manifest.toml` |
| `check-spelling` (`scripts/checks/check_spelling.py`) | American English, the productive `-ise` families | Python, Markdown, TOML and YAML outside the excluded paths |
| `check-md-math` (`scripts/checks/check_md_math.py`) | math that renders in both Markdown previews; the frontmatter parses | `docs/` |
| `check-references` (`scripts/checks/check_references.py`) | every citation key has an entry in `docs/references.bib`; `docs/references.md` equals the generated page; every relative link resolves | the tracked Python and Markdown outside `docs/changes/`, `docs/plans/` and the legacy scope |
| `check-docs` (`scripts/checks/check_docs.py`) | the closed `type` and `status` vocabularies, the required headings and their templates, the index, the chronological Done ledger, superseded ADRs, no unconfirmed marker in a settled document | `docs/` |
| `check-changes` (`scripts/checks/check_changes.py`) | each round of a change document's *Review notes* covers exactly the files changed in its range | the change documents after 2026-11-25 |
| `check-containers` (`scripts/checks/check_containers.py`) | rule R2 | the clean scope |
| `check-dependencies` (`scripts/checks/check_dependencies.py`) | every imported package is declared in `pyproject.toml`; rule R12; the package imports nothing from `scripts/` or `tests/` | the package, `tests/`, `scripts/` |
| `check-docstrings` (`scripts/checks/check_docstrings.py`) | docstring headers on their own line; private functions documented; modules under 600 lines; no docstring opens with a numbered plan label | the clean scope |
| `ty`, run by `just check` | the types check, with no `ty: ignore` comment (a grep, because `ty` honors the comments) | the clean scope, with `simulations/pde/burgers.py` excluded until the return type of `solve` is settled |
| `tests/unit/test_imports.py` | every module of the package imports on CPU | the package, `legacy/` included |
| `tests/unit/test_configs.py` | every model config instantiates with every data config of its dimension | `scripts/training/configs/` |
| the jaxtyping and beartype import hook of `tests/conftest.py` | shape and type annotations hold at run time | the clean scope under `src/`, in every test run |

**Two spelling checks, because one tool cannot do both jobs.** `codespell` is a list of
British-to-American words, which is the right shape for irregulars and for typos.
`-ise` is productive morphology and no list ships `discretisation` or `parametrised`, <!-- codespell:ignore -->
so `scripts/checks/check_spelling.py` applies the rule (`-ise`, `-isation`, `-isable`
and `-yse` take a `z`) and keeps the closed list of exceptions. Neither check sees a
British spelling inside a hyphenated compound, so a sweep ends with a grep for the
project's compounds.

## Deliberate exceptions

- **`legacy/` stays importable and otherwise untouched.** The strict checks, ruff and
  `codespell` skip `src/context_flux_no/legacy/` and `tests/legacy`. Later branches
  rewrite that code into the planned packages of [`architecture.md`](architecture.md).
- **`docs/changes/` is skipped by the spelling checks and by `check-references`.** A
  change document holds Joon's words verbatim. The six change documents dated before
  2026-11-25 are also exempt from the heading rule and from `check-changes`
  (`PRE_FORMAT_CHANGES` in `check_docs.py`); that list never grows.
- **The body of a document under `docs/plans/` is never edited.** A committed plan is a
  frozen record of what was planned, and `check-references` skips it.
- **`simulations/pde/burgers.py` is excluded from `ty`.** `ViscousBurgers1D.solve`
  returns its cell centers as a tuple where the base class declares a list, and the fix
  changes a public return type.

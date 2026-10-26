---
type: reference
status: current
updated: 2026-10-25
---

# Coding standards

Where this repository enforces the conventions, and which checks stand in for prose. The
conventions themselves, with the reasoning behind them, are in the `code-conventions`
skill; the pervasive ones are in the global `CLAUDE.md`. This document is the place to
look when a task touches tiers, tolerances, goldens or checks.

## Tiers and commands

| Purpose | Command |
|---|---|
| Fast, while editing | `uv run pytest <files touched> -m "not regression and not slow"` |
| Full fast tier | `uv run pytest -m "not regression and not slow"` |
| Regression, CPU only | `JAX_PLATFORMS=cpu uv run pytest tests/regression --checkpoints-root ./checkpoints` |
| Verification | `-m slow`, with `jax_enable_x64` |
| Imports | `uv run python scripts/import_all.py` |
| Configs | `uv run python scripts/check_configs.py` |
| Documents | `uv run python scripts/check_docs.py` |

Hypothesis profiles: `--hypothesis-profile=dev` (10 examples), default (50), `ci` (200,
derandomized). The four tiers are ADR-0008: `tests/unit` and `tests/property` on every
commit, `tests/regression` against the ICLR checkpoints, `tests/verification` as the slow
tier.

Read the baseline before editing — the branch change document's *Tests* and *Open
issues*, and `docs/findings.md` — rather than running a full tier to discover what is
already red. GPU machines add `--extra cuda`; PyClaw needs `build-essential` and
`gfortran`.

## Test contracts

What each tier promises, so a new test has one obvious home (ADR-0008):

| Tier | Directory | Contract |
|---|---|---|
| unit | `tests/unit` | fast, deterministic, no external data; each test is a law at a contract (what the module promises), not at its internals; a test that must change when the implementation changes was testing internals |
| property | `tests/property` | Hypothesis laws over generated inputs; strategies in `tests/strategies.py`, parameterized by bounds rather than duplicated; `N = 1, 2, 3` spatial dimensions wherever a function takes `axis` (V23) |
| verification | `tests/verification` | every test carries a `docs/verification.md` row id in its docstring and compares against an **independent oracle** (closed form, another code, a published number); runs under `jax_enable_x64` (its conftest); marked `slow` only when it is slow — a fast verification test still lives here and runs on every commit. Placement is by the claim, not by speed |
| regression | `tests/regression` | ICLR checkpoints reproduce their goldens on CPU; skips (never fails) without the checkpoints |

Shared across tiers: tolerances and solver settings are named entries with a one-line reason
in `tests/helpers.py` (created with the first numerics test); a test file tests one module or
one claim — a file edited for unrelated reasons is split.

## Goldens and tolerances

- Regression goldens (`tests/regression/goldens/*.npz`) are generated **and** compared on
  CPU, because XLA's GPU reductions are not bitwise reproducible across versions and
  devices. On CPU, fp32 results are stable to about `1e-6` across runs, which makes
  `rtol=1e-5, atol=1e-6` a real test rather than a flaky one.
- A golden is never regenerated without a change-document entry naming the commit that
  changed behavior and why.
- The regression tier skips rather than fails in three cases — no `--checkpoints-root` or
  `CFNO_CHECKPOINTS`, a checkpoint path still holding the `<TIMESTAMP>` placeholder, a
  missing input or golden — so CI without checkpoints stays green.
- `make_goldens.py` refuses a non-CPU backend and never overwrites without `--force`.
- Backend goldens (Phase 2a) are bit-identity: a reference solver's output is pinned
  exactly, and `docs/design/simulation-backends.md` records the policy and the measured
  agreement ratios.
- `ty` must stay clean on `src/context_flux_no/fv` once it exists.

## Scope

- `icon/` and `third_party/` are vendored and excluded from ruff; `notebooks/` is excluded
  until Phase 6 thins it.
- `E501` is ignored because `ruff format` governs line length; `F821` because jaxtyping
  shape strings such as `Float[Array, "channels *grids"]` trip it.
- CUDA is an optional extra so that CI and the regression tier install the CPU wheel.

## Checks that replace prose

The configuration lives in `pyproject.toml` (`[tool.ruff.lint]` with `TID251` selected,
`[tool.ruff.lint.flake8-tidy-imports.banned-api]`, `[tool.codespell]`), with a comment
there on each non-obvious setting. This section says what each check is for; read the file
for the current values.

| Rule | Check | Runs in |
|---|---|---|
| American English, irregulars and typos | `codespell` 2.4.3 | pre-commit, CI |
| American English, the productive families | `scripts/check_spelling.py` | pre-commit, CI |
| `copy.replace`, never `dataclasses.replace` (Python floor 3.13) | ruff `TID251` banned-api | pre-commit, CI |
| change-document *Files* matches the diff; frontmatter; index coverage; Done-ledger order | `scripts/check_docs.py` | pre-commit, CI |
| every model config instantiates | `scripts/check_configs.py` | by hand |
| every module imports | `scripts/import_all.py` | by hand |

**Two spelling checks, because one tool cannot do both jobs.** `codespell` is a
535-entry British-to-American word list: the right shape for irregulars (*behaviour*, <!-- codespell:ignore -->
*centre*, *artefact*) and for typos — it is what caught `splitted`, `recommand`, <!-- codespell:ignore -->
`quantites` and `meybe`. It is the wrong shape for this repository's vocabulary, because <!-- codespell:ignore -->
`-ise` is productive morphology and no list ships `discretisation`, `parametrised`, <!-- codespell:ignore -->
`denormalise` or `amortisation` [code: measured against codespell 2.4 and `breame`, <!-- codespell:ignore -->
2026-10-25; both miss all four]. `scripts/check_spelling.py` inverts the problem: it
applies the rule (`-ise`/`-isation`/`-isable`/`-yse` take a `z`) and keeps the exception
list, which is the part that *is* closed — the sixty-odd words where `-ise` is also
correct in American English. `-isation` and `-isable` have no exceptions at all; `-yses`
is left out of the rule on purpose, since it is also the plural of `-ysis` (*the
analyses*, which is correct).

Both checks report 0 findings in scope. Out of scope, for both: `docs/changes/` (a change
document is the record of its branch, British spellings included), `notebooks/`, the
vendored trees, and the Julia `Manifest.toml` (package names such as `Missings` are not <!-- codespell:ignore -->
words). Neither check sees a British spelling inside a hyphenated compound (`cell-centred`), <!-- codespell:ignore -->
so a sweep ends with a grep for the project's compounds. The 2026-10-26 cleanup took two
commits rather than one, because the British spelling of *cell center* reached a test name
as well as docstrings, and renaming a test is a code change.

## Persistence verbs

One verb pair per kind of thing, so a reader can guess the method name:

| Thing | Verbs | Examples |
|---|---|---|
| a record or document file (JSON, YAML) | `write(path)` / classmethod `read(path)`; `from_dict` for an in-memory dict | `RunRecord`, `DatasetRecord` |
| a directory-like object | classmethod `open(path)`; `create`/`from_config` to make a new one | `Run.open`, `WellWriter.create` |
| checkpointed state through orbax | `save(step, state)` / `restore(template, step)` (orbax's words) | `OrbaxCheckpointer`, `Run.restore` |
| JSON encoding | `utils.serialization.dumps`; decoding is `json.loads` + `from_dict` | both records, zarr attrs |

No `to_json`/`from_json`/`to_dict` wrappers on dataclasses: `dataclasses.asdict` and
`copy.replace` are called directly. `training/io.py::load_model` is the legacy exception and
goes with the ICLR shim.

## Deliberate exceptions

- `SegmentLoaderBackground` survives behind a `DEPRECATED` header naming Phase 4 as its
  removal: seven evaluation notebooks still read the ICLR-era 1D datasets through it, and
  `xarray`, `h5netcdf`, `netcdf4` and `dask` stay in `pyproject` for the same reason until
  Phase 7. A shim without a header and a named removal phase is not allowed.
- Phase 1 may be worked before the Phase 0.8 goldens exist; the goldens gate the *merge*,
  not the work.
- `third_party/icon` carries upstream's `pyproject.toml`, `uv.lock` and `data_generation/`
  folder; whether they stay is a question for that repository, not this one.

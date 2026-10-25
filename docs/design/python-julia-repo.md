---
type: design
status: agreed
updated: 2026-10-25
verified_by: joon (interview 2026-10-25: juliacall in-process, Julia ~1.12, committed Manifest; §Decided)
sources: [juliapkg 0.1.28 source (deps.py, state.py, find_julia.py) and PyPI README; JuliaCall docs (stable; Multi-threading, Configuration, Managing dependencies); PythonCall.jl README and FAQ 0.9.19; PySR (julia_import.py, sr.py, __init__.py) and SymbolicRegression.jl (parallelism); Julia Pkg docs; Trixi.jl docs; ADR-0004]
---

# Maintaining Julia (Trixi.jl) inside a `uv`-managed Python repository

ADR-0004 (amended 2026-10-25) keeps Julia a pinned dependency behind `SolverBackend`;
`TrixiBackend` runs it in-process through juliacall. The question here is purely operational — how a clone of this
repository gets a working, *reproducible* Julia + Trixi.jl without a second toolchain to document
— and which of the three workable layouts to pick. Provenance: **[cited]** from the named source,
**[deduced]** Claude's inference.

## The three layouts

### A. `juliapkg` as installer, `reference/trixi/` as the Julia project — **recommended**

- `juliapkg` (PyPI; the dependency manager behind JuliaCall) goes in the `dev` dependency group,
  so `uv sync --group dev` brings it in. It installs or locates Julia: the `-X juliapkg-exe`
  option, then `PYTHON_JULIAPKG_EXE`, then a compatible `julia` on `PATH`, then `juliaup` on
  `PATH` (used to install a compatible version), else a download into the Julia project
  **[cited: juliapkg README]**. The required Julia version is the `"julia"` compat string in a
  `juliapkg.json`; `juliapkg.executable()` resolves (installing if needed) and returns the
  binary **[cited]**.
- The Julia **environment** is *not* `juliapkg.json`'s package list but a conventional Julia
  project committed at `reference/trixi/{Project.toml, Manifest.toml}`. `Manifest.toml` is
  Julia's lock file (exact versions of the whole dependency graph), and
  `julia --project=reference/trixi -e 'using Pkg; Pkg.instantiate()'` recreates it exactly
  **[cited: Pkg docs, "Project.toml and Manifest.toml"]**. `juliapkg.json` therefore pins only
  Julia itself (e.g. `{"julia": "=1.11.7"}` or `"~1.11"`), nothing else. `juliapkg.json` has no
  lock file, so putting Trixi in it would give compat ranges, not exact reproducibility; it does
  take a `dev`/`path` entry for the local package [code: juliapkg 0.1.28 `PkgSpec`] — see
  §Decided.
- Python side: `simulations/trixi.py` offers two transports over the same Julia function
  (see C below): in-process through `juliacall` (default) or a subprocess
  `subprocess.run([juliapkg.executable(), f"--project={REF_TRIXI}", "--startup-file=no",
  "run.jl", request_json])`. Either way `juliapkg` only locates/installs Julia; the Trixi
  environment is `reference/trixi/`'s `Manifest.toml`.
- Where things live: with an active virtualenv, juliapkg's project (and a downloaded Julia) go
  to `.venv/julia_env/` **[cited]**; Julia packages themselves go to the depot (`~/.julia`, or
  `JULIA_DEPOT_PATH`). On a cluster set `JULIA_DEPOT_PATH` to a shared/scratch path and run the
  instantiate step on a login node with network; compute nodes then need nothing (set
  `PYTHON_JULIAPKG_OFFLINE=yes` to make an accidental resolve fail loudly instead of trying to
  download **[cited]**).
- One `uv run python scripts/setup_julia.py` does: `juliapkg.resolve()` → `Pkg.instantiate()`
  → `Pkg.precompile()` → a smoke run of one elixir. CI (CPU) runs it with a cache on
  `~/.julia` and `.venv/julia_env` keyed by the two lock files.

### B. Pure `juliaup` + committed `Project.toml`/`Manifest.toml`

Same Julia project as A, but Julia is installed by the user with `juliaup` and pinned by a
`juliaup override set <channel>` in the repo directory, or by `JULIA_VERSION` in the README.
No Python dependency at all. Cost: the Julia version is pinned by a README instruction rather
than by a file `uv sync` reads; a fresh clone needs one manual step before `TrixiBackend`
works; CI needs a `julia-actions/setup-julia` step. Reasonable if `juliapkg` ever misbehaves,
and A degrades to B gracefully (`PYTHON_JULIAPKG_EXE=$(which julia)`).

### C. `juliacall` in-process — re-assessed 2026-10-25 (round 2)

**Correction.** The first assessment conflated the two directions of PythonCall.jl. The FAQ's
"Is PythonCall/JuliaCall thread safe? — No", the rule "only call Python from the first
thread", and the `PythonCall.GC.disable()/enable()` advice concern *Python objects touched
from Julia threads* (the PythonCall direction, and Julia code that calls back into Python).
They do **not** forbid running multithreaded *pure-Julia* code from Python through JuliaCall.
What the JuliaCall page actually says [cited: JuliaCall docs, "Multi-threading"]:

- `PYTHON_JULIACALL_THREADS=N|auto` (and `_PROCS`) launches Julia with N threads;
- Julia code may run while Python's GIL is released by calling it through
  `f._jl_call_nogil(...)`, so Python threads can run Julia in parallel (the docs' example: four
  `Libc.systemsleep` calls in ~5 s instead of ~20 s);
- the constraints: a `_jl_call_nogil` function must not touch Python unless it re-acquires
  the GIL (`PythonCall.GIL.@lock`), and functions that *yield to Julia's scheduler* (e.g.
  `sleep`, `wait`) can hang without a `jl.yield()` polling loop;
- `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` is required with multiple threads (GC safepoints are
  implemented with deliberate segfaults); the documented cost is that Ctrl-C no longer raises
  `KeyboardInterrupt`;
- the feature is labelled "experimental and can change without notice".

**PySR as precedent** [cited: `pysr/julia_import.py`, `pysr/sr.py`, `pysr/__init__.py`;
`SymbolicRegression.jl/src/SymbolicRegression.jl`]: SymbolicRegression.jl's default is
`parallelism=:multithreading` (multiprocessing via `addprocs` also supported, including
Slurm cluster managers), and PySR drives it from Python exactly through the mechanism above:
it sets `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, `PYTHON_JULIACALL_THREADS=auto` and
`JULIA_NUM_GC_THREADS` *before* `import juliacall`, imports JuliaCall as the very first thing
in `pysr/__init__.py` ("to prevent library linking issues caused by numpy/pytorch/etc.
importing old libraries"), and calls `equation_search._jl_call_nogil(...)` for the search. It
works around the lost Ctrl-C with its own "external stop signal" context. So in-process,
multithreaded Julia from Python is a production pattern, not a corner case.

*What remains true against C, for this repository.*
- The import-order constraint (`juliacall` before `jax`/`numpy`) and the signal setting are
  process-wide invariants that every entry point must honour; PySR solves it by owning the
  package `__init__`. Here they can be confined to one module (`simulations/trixi.py`, which
  sets the environment and imports `juliacall` on first use) **provided** that module is
  imported before `jax` in the process that uses it — true for `scripts/data/generate_dataset.py`
  if the import is placed first, false for a notebook that already imported JAX. **[deduced]**
- Process coupling: a Julia OOM or crash takes the JAX process with it; a subprocess returns
  an exit code that `DatasetRecord.n_failed` records. For a Slurm generation job this is the
  one practical difference.
- `_jl_call_nogil` + scheduler-yielding code: Trixi/OrdinaryDiffEq time stepping does not
  yield, but Trixi's `AnalysisCallback`/`AliveCallback` printing and any `ProgressMeter` use
  can; run the elixir with callbacks disabled or accept blocking under the GIL. **[deduced]**

*What C gains.* No file protocol and no temp directory; arrays cross with no copy; one warm
Julia session per Python process, so TTFX is paid once per *process*, not once per dataset —
which matters for tests, notebooks and small regenerations far more than for a 100 × 100
trajectory job; and a REPL-like loop for developing the elixirs from Python.

*Verdict (revised; superseded by §Decided — Joon chose juliacall alone, subprocess as the
unbuilt fallback, and the `PYTHON_JULIACALL_EXE`/`_PROJECT` mechanism below gave way to
juliapkg's shared-project mode).* Both transports are viable; the decision is about failure isolation and
ergonomics, not thread safety. Design the Julia side as a **local package**
`reference/trixi/ContextFluxTrixi.jl` exposing one function
`solve(family, params, u0::Array, grid, t_out, bcs; polydeg) -> Array`, so the transport is a
thin layer either way, and implement **both** behind `TrixiBackend(transport=…)`:

- `transport="juliacall"` — the default for development, tests and notebooks: `simulations/trixi.py`
  sets `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, `PYTHON_JULIACALL_THREADS=auto` (and
  `PYTHON_JULIACALL_EXE`/`_PROJECT` to the `reference/trixi` project so `juliapkg.json` is
  not needed for packages) before importing `juliacall`, then calls
  `ContextFluxTrixi.solve._jl_call_nogil(...)` with numpy arrays. Document the import-order
  rule in the module docstring and make the generate script import it first.
- `transport="subprocess"` — `run.jl` wrapping the same `solve` via `request.json`/`u.h5`:
  for Slurm generation jobs where a Julia failure must not kill the Python process, and as the
  fallback if the experimental multithreading path misbehaves.

This revises ADR-0004's "Julia subprocess only" wording: the *pinned, file-based* intent (Julia
as an oracle/generator behind `SolverBackend`, not as a dependency of `src/` at large) stands;
the transport is a per-call choice. Amend ADR-0004 with one line when 2a-trixi lands.

## Decided (2026-10-25) and what the code does

Layout A with juliacall as the one transport; the details below are **[code]** against
juliapkg 0.1.28 and juliacall 0.9.37 unless marked, and correct two earlier deductions.

```
pyproject.toml                            dev group: juliapkg, juliacall
src/context_flux_no/juliapkg.json         {"julia": "~1.12", "packages": {"ContextFluxTrixi": {uuid, dev: true,
                                           path: "../../reference/trixi/ContextFluxTrixi"}}}
reference/trixi/
  Project.toml, Manifest.toml             committed; the lock (ContextFluxTrixi dev-ed, PythonCall at juliacall's version)
  ContextFluxTrixi/                       the Julia package: equations, node_coordinates, solve, versions; test/runtests.jl
  pyjuliapkg/                             juliapkg's state (Julia install, meta.json, lock) — git-ignored
scripts/setup_julia.py                    resolve + instantiate + precompile + describe(); --test runs Pkg.test
src/context_flux_no/simulations/trixi.py  TrixiSpec, TrixiBackend; sets the environment, imports juliacall on first solve
```

- **`juliapkg.json` location.** juliapkg scans every `sys.path` entry *and its immediate
  subdirectories* for `juliapkg.json`, and editable finders through their `MAPPING`
  [code: `deps.py::deps_files`, `editable_deps_files`]; the file at `src/context_flux_no/`
  is found in this editable install (verified). The earlier note that package-level files are
  read only from installed packages was wrong.
- **`dev`/`path` entries exist** [code: `deps.py::PkgSpec(dev, path)`; `path` is resolved
  relative to the json file]; the earlier "schema is `uuid`, `version` only" was wrong. So the
  local package is `Pkg.develop`-ed by juliapkg itself; no hand `] dev`.
- **Shared-project mode.** `PYTHON_JULIAPKG_PROJECT=<abs path>` makes juliapkg use that
  directory's `Project.toml` *as it is* and keep its `Manifest.toml` (only a non-shared project
  has its Manifest deleted on resolve) [code: `state.py:63-69`, `deps.py:560-607`]. That is how
  the committed `reference/trixi/Manifest.toml` is honored while juliapkg still installs Julia
  and pins PythonCall to the `juliacall` version — one mechanism instead of the two
  (`PYTHON_JULIACALL_EXE`/`_PROJECT` plus a hand-run `Pkg.instantiate`) first proposed.
- **Environment, set once** by `simulations.trixi.configure_environment()` before the first
  `import juliapkg`/`juliacall` (`setdefault`, so a shell setting wins):
  `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, `PYTHON_JULIACALL_THREADS=auto`,
  `PYTHON_JULIAPKG_PROJECT=<repo>/reference/trixi`.
- **Availability gate.** `trixi.available()` is "juliapkg has resolved this project here"
  (`reference/trixi/pyjuliapkg/meta.json` exists) — cheap, no Julia start, and `False` until
  `scripts/setup_julia.py` has run, so tests skip instead of triggering a Julia download.
- **Transport.** juliacall only; Python passes numpy arrays and plain scalars, Julia receives
  `PyArray`/`PyDict` and copies; no `_jl_call_nogil` (nothing runs in parallel on the Python
  side) and no Python called from Julia. The subprocess transport is the fallback (ADR-0004
  §Amendment), not built.
- **Pinning policy.** `Manifest.toml` is updated deliberately (`Pkg.update()` in a branch,
  V18 re-run, change-document entry), like `uv.lock`.

## What happened on Joon's laptop (ledger 4–5 of the change document, 2026-10-26) [code]

`uv sync --group dev --extra cuda` (the CUDA extra is installed there; a sync without it
removes it); write `reference/trixi/ContextFluxTrixi/` with a complete `Project.toml` (UUIDs
from the General registry; compat `Trixi = "0.17"`, `OrdinaryDiffEqSSPRK = "2"`,
`SciMLBase = "2.154, 3"`, `StaticArrays = "1"`, `julia = "1.10"`; `[extras]`/`[targets]` for
`Test`, `Random`); `touch reference/trixi/Project.toml`; `uv run python scripts/setup_julia.py
--test`. One run: juliapkg downloaded Julia 1.12.7 (1.1 GB) into `reference/trixi/pyjuliapkg/
install`, wrote `Project.toml` (`ContextFluxTrixi` dev-ed, `PythonCall = "=0.9.37"`,
`OpenSSL_jll`), resolved the `Manifest.toml` (Trixi 0.17.16, OrdinaryDiffEqSSPRK 2.3.2,
SciMLBase 3.58.0, StaticArrays 1.9.24; 2042 lines), precompiled (~15 min), ran `Pkg.test`,
started Julia through juliacall and printed `describe()` (34 s cold). Both TOMLs committed.

Two Trixi 0.17 facts the design documents had not recorded: `SemidiscretizationHyperbolic`
requires `boundary_conditions` explicitly (`boundary_condition_periodic`), and the shallow-water
equations moved to TrixiShallowWater.jl (the runtests oracle for the parametrized shallow water
is therefore the textbook flux and eigenvalues, LeVeque 2002 §13.1). Earlier a bare
`import juliacall` probe had installed Julia into `.venv/julia_env/` (juliapkg's per-venv
default); deleted, and `CLAUDE.md` now says to probe with `trixi.available()`.

## Sources

[cited] juliapkg README on PyPI (<https://pypi.org/project/juliapkg>): search order for the
Julia executable, `juliapkg.json` schema and locations, `PYTHON_JULIAPKG_{EXE,PROJECT,OFFLINE}`,
`.venv/julia_env` default, API (`require_julia`, `add`, `resolve`, `executable`, `project`) ·
JuliaCall documentation, PythonCall.jl 0.9
(<https://docs.juliahub.com/PythonCall/WdXsa/0.9.5/juliacall>): configuration options, Julia
installed via JuliaPkg on demand, environment choice · PythonCall.jl README
(<https://github.com/JuliaPy/PythonCall.jl>): non-copying array conversion, Julia ≥ 1.10 / Python ≥ 3.10 ·
PythonCall FAQ 0.9.19 (<https://docs.juliahub.com/PythonCall/WdXsa/0.9.19/faq>): thread safety
*of Python objects from Julia threads*, `PYTHON_JULIACALL_HANDLE_SIGNALS`, libstdc++ and PyTorch
heap-corruption notes · PySR (<https://github.com/MilesCranmer/PySR>: `pysr/julia_import.py`,
`pysr/sr.py` `_jl_call_nogil`, `pysr/__init__.py` import order) and SymbolicRegression.jl
(<https://github.com/MilesCranmer/SymbolicRegression.jl>: `parallelism=:multithreading` default)
as the production precedent for threaded Julia under juliacall · Julia `Pkg` documentation
(<https://pkgdocs.julialang.org>): Project/Manifest semantics and `instantiate` · Trixi.jl
documentation (<https://trixi-framework.github.io/Trixi.jl/stable/>) for the equation/elixir
structure · ADR-0004 for the subprocess decision.
[deduced] no-lock-file argument against listing Trixi in `juliapkg.json`; in-process risk
assessment; per-dataset process amortisation.

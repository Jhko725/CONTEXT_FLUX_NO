---
type: design
status: tentative
updated: 2026-10-25
verified_by: — (Claude's proposal for Joon; becomes `agreed` once Joon has chosen)
sources: [juliapkg PyPI README; JuliaCall docs (stable; Multi-threading, Configuration, Managing dependencies); PythonCall.jl README and FAQ 0.9.19; PySR (julia_import.py, sr.py, __init__.py) and SymbolicRegression.jl (parallelism); Julia Pkg docs; Trixi.jl docs; ADR-0004]
---

# Maintaining Julia (Trixi.jl) inside a `uv`-managed Python repository

ADR-0004 makes Julia "a pinned subprocess dependency only": `TrixiBackend` writes a request,
runs `julia`, reads the result. The question here is purely operational — how a clone of this
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
  lock file, so putting Trixi in it would give compat ranges, not exact reproducibility
  **[deduced from the README's schema: `julia`, `packages{uuid, version}` only]**.
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

*Verdict (revised).* Both transports are viable; the decision is about failure isolation and
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

## Recommended layout (A), concretely

```
pyproject.toml                   dev group: juliapkg>=0.1.17
juliapkg.json                    {"julia": "~1.11"}          ← project-level: package-level files are only
                                                              read from *installed* packages; keep it at the
                                                              root and point PYTHON_JULIAPKG_PROJECT at
                                                              .venv/julia_env in scripts/setup_julia.py
reference/trixi/
  Project.toml                   Trixi, OrdinaryDiffEqSSPRK (or OrdinaryDiffEq), HDF5, JSON3, [compat]
  Manifest.toml                  committed; the lock
  run.jl                         reads request.json → includes the elixir → writes u.h5
  elixirs/{cubic_1d,sine_1d,burgers_1d,shallow_water_1d,cubic_2d,euler_2d}.jl
  ContextFluxTrixi/              local package (`dev`-ed into Project.toml): custom scalar equations
                                 + `solve(...)`; run.jl (subprocess transport) and juliacall (in-process
                                 transport, default) are thin callers of the same function
scripts/setup_julia.py           resolve + instantiate + precompile + smoke test
src/context_flux_no/simulations/trixi.py   TrixiBackend (subprocess; temp dir; JSON + HDF5)
```

Pinning policy: `Manifest.toml` is updated deliberately (`Pkg.update()` in a branch, agreement
test re-run, change document entry) exactly like `uv.lock`. `.gitignore`: `.venv/julia_env/` is
already under `.venv/`; nothing else to ignore. Pre-commit: a `juliaformatter` hook is optional;
skip until the Julia surface grows.

## What this means for 2a-Trixi on Joon's machine

Steps 1–2 of the ledger's hand-off become: `uv add --group dev juliapkg juliacall`, write
`juliapkg.json` (`{"julia": "~1.11"}`), `uv run python scripts/setup_julia.py` (first run
downloads Julia if `juliaup`/`julia` are absent), then `julia --project=reference/trixi` →
`] add Trixi OrdinaryDiffEqSSPRK HDF5 JSON3 PythonCall` (PythonCall.jl must be in the project
for the juliacall transport, at the version matching the Python `juliacall` [cited: JuliaCall
docs, "Managing Julia dependencies"]), `] dev ./ContextFluxTrixi`, commit both TOMLs. Then
`ContextFluxTrixi.solve`, the elixirs, `simulations/trixi.py` with both transports, and V18.

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

---
type: design
status: tentative
updated: 2026-10-25
verified_by: — (Claude's proposal for Joon; becomes `agreed` once Joon has chosen)
sources: [juliapkg PyPI README; JuliaCall docs (PythonCall.jl 0.9); PythonCall.jl README and FAQ 0.9.19; PySR julia_import.py; Julia Pkg docs; Trixi.jl docs; ADR-0004]
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
- Python side: `simulations/trixi.py` calls
  `subprocess.run([juliapkg.executable(), f"--project={REF_TRIXI}", "--startup-file=no",
  "run.jl", request_json])`. No `juliacall`, no libjulia in the Python process: JAX and an
  embedded Julia runtime would share one process's threads and signal handlers, which is a class
  of problems a subprocess simply does not have **[deduced]**; JuliaCall's own knobs for
  threads/sysimage exist (`PYTHON_JULIACALL_THREADS`, `_SYSIMAGE`, … **[cited: JuliaCall docs]**)
  but are only needed in-process.
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

### C. `juliacall` in-process (assessed at Joon's request, 2026-10-25)

`import juliacall` embeds libjulia in the Python process; Julia is installed on demand through
juliapkg, and a `Main.seval`/`Main.include` API runs Julia code. What the evidence says:

*For it.*
- **No file protocol.** PythonCall advertises "fast non-copying conversion of numeric arrays in
  either direction" (numpy ↔ Julia `Array`) **[cited: PythonCall README]**, so `u0` goes in and
  `u` comes out as arrays; `request.json`/`u.h5` and the temp directory disappear.
- **Interactive use.** Calling a Trixi elixir from a notebook, inspecting the DG solution,
  changing `polydeg` and re-running is immediate; with the subprocess route every experiment is
  a file round trip.
- **Same installer.** It needs `juliapkg` and a `juliapkg.json` exactly as layout A does, so A's
  infrastructure is a prerequisite, not an alternative **[cited: JuliaCall docs]**.
- **Precedent.** PySR drives a large Julia package (SymbolicRegression.jl) from Python through
  juliacall in production, including its own handling of the signal and thread options
  (`pysr/julia_import.py`) **[cited]** — so the route is mature when its rules are followed.

*Against it, for this repository's use.*
- **Not thread safe.** The FAQ answers "No": Python may only be called from Julia's first
  thread, and Julia finalizers can run on any thread, so `PythonCall.GC.disable()/enable()` is
  needed around threaded Julia code — and "problems may still occur" **[cited: PythonCall FAQ
  0.9.19]**. Trixi's performance on 2D Euler comes from `Threads.@threads` over elements, so the
  production generator would run exactly in the configuration the FAQ warns about.
- **Signals.** Julia uses deliberate segfaults as GC safepoints; without
  `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` set *before import* they kill the process **[cited:
  FAQ]**. Settable, but one more invariant for every entry point (scripts, tests, notebooks,
  Slurm jobs) to respect.
- **Shared-library conflicts.** The FAQ documents `libstdc++` clashes ("`ccall` requires the
  compiler" when another library is imported first) and heap corruption with PyTorch unless
  `juliacall` is imported *before* it **[cited: FAQ]**. `jaxlib` is a manylinux wheel bundling its
  own C++ runtime, the same hazard class as the documented cases — untested for JAX specifically,
  but the mitigation ("import juliacall first, everywhere") is fragile across Hydra scripts and
  pytest collection **[deduced]**.
- **Process coupling.** A Julia crash, OOM or precompile stall takes the JAX process with it; with
  a subprocess it is a non-zero exit code recorded as a failed trajectory batch in
  `DatasetRecord.n_failed`. On Slurm the subprocess also lets Julia use all node cores without
  negotiating with Python.
- **No performance need.** `TrixiBackend` runs per *dataset* (hundreds of trajectories per
  process); array transfer through HDF5 is milliseconds against seconds of DG time stepping,
  and Julia start-up/TTFX is paid once per dataset either way **[deduced]**.

*Verdict.* The transport, not the Julia code, is what the two routes differ in. Design the Julia
side as a **local package** `reference/trixi/ContextFluxTrixi.jl` exposing one function,
`solve(family, params, u0::Array, grid, t_out, bcs; polydeg) -> Array`, and keep the elixirs as
thin callers of it. Then:

- **production / generator / tests**: `run.jl` wraps `solve` for the subprocess `TrixiBackend`
  (ADR-0004 as written);
- **exploration**: a notebook may `import juliacall` (with `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`,
  imported before jax) and call `ContextFluxTrixi.solve` on numpy arrays directly — nothing in
  `src/` depends on it.

That keeps one Julia implementation, costs nothing now, and leaves an in-process
`TrixiBackend(transport="juliacall")` as a ten-line addition if the file protocol ever becomes a
bottleneck. It is *not* recommended to make juliacall the default transport, for the thread,
signal and library-order reasons above.

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
                                 + `solve(...)`; elixirs and run.jl are thin callers; juliacall can
                                 call the same `solve` in a notebook (layout C, exploration only)
scripts/setup_julia.py           resolve + instantiate + precompile + smoke test
src/context_flux_no/simulations/trixi.py   TrixiBackend (subprocess; temp dir; JSON + HDF5)
```

Pinning policy: `Manifest.toml` is updated deliberately (`Pkg.update()` in a branch, agreement
test re-run, change document entry) exactly like `uv.lock`. `.gitignore`: `.venv/julia_env/` is
already under `.venv/`; nothing else to ignore. Pre-commit: a `juliaformatter` hook is optional;
skip until the Julia surface grows.

## What this means for 2a-Trixi on Joon's machine

Steps 1–2 of the ledger's hand-off become: `uv add --group dev juliapkg`, write `juliapkg.json`,
`uv run python scripts/setup_julia.py` (first run downloads Julia if `juliaup`/`julia` are
absent), then `julia --project=reference/trixi` → `] add Trixi OrdinaryDiffEqSSPRK HDF5 JSON3`,
commit both TOMLs. Everything after that is elixir writing.

## Sources

[cited] juliapkg README on PyPI (<https://pypi.org/project/juliapkg>): search order for the
Julia executable, `juliapkg.json` schema and locations, `PYTHON_JULIAPKG_{EXE,PROJECT,OFFLINE}`,
`.venv/julia_env` default, API (`require_julia`, `add`, `resolve`, `executable`, `project`) ·
JuliaCall documentation, PythonCall.jl 0.9
(<https://docs.juliahub.com/PythonCall/WdXsa/0.9.5/juliacall>): configuration options, Julia
installed via JuliaPkg on demand, environment choice · PythonCall.jl README
(<https://github.com/JuliaPy/PythonCall.jl>): non-copying array conversion, Julia ≥ 1.10 / Python ≥ 3.10 ·
PythonCall FAQ 0.9.19 (<https://docs.juliahub.com/PythonCall/WdXsa/0.9.19/faq>): thread safety,
`PYTHON_JULIACALL_HANDLE_SIGNALS`, libstdc++ and PyTorch heap-corruption notes · PySR's
`pysr/julia_import.py` (<https://huggingface.co/spaces/MilesCranmer/PySR/blob/47ce6616a74a03c692a8fc52b3e751bb3b62dd4b/pysr/julia_import.py>) as a production juliacall precedent · Julia `Pkg` documentation
(<https://pkgdocs.julialang.org>): Project/Manifest semantics and `instantiate` · Trixi.jl
documentation (<https://trixi-framework.github.io/Trixi.jl/stable/>) for the equation/elixir
structure · ADR-0004 for the subprocess decision.
[deduced] no-lock-file argument against listing Trixi in `juliapkg.json`; in-process risk
assessment; per-dataset process amortisation.

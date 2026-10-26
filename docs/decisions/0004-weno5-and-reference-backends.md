---
type: decision
id: ADR-0004
status: amended
updated: 2026-10-25
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0004 — Own WENO5, time-boxed, verified against PyClaw/SharpClaw and Trixi.jl

**Status**: accepted (2026-09-27); amended 2026-10-25 (§Amendment: the Trixi transport).

## Context

No JAX WENO5 exists for the scalar families: exponax is spectral and explicitly excludes
discontinuities [cited: its design notes]; JAX-Fluids has WENO5-Z/TENO6 but is Euler/NS-only
[cited: arXiv 2402.05193]; Trixi.jl is Julia, DGSEM, no WENO [cited: Trixi API]. Joon wants to
learn WENO by implementing it, and wants a fallback if it takes too long.

## Decision

Write WENO5 in-house (FD-WENO, LF splitting, JS weights, Z optional) as a classical
`FluxPredictor`, time-boxed to one week, with `design/weno5.md` as a learning deliverable.
*Before* writing it, build two file-based interop backends — `PyClawBackend` (Classic +
SharpClaw) and `TrixiBackend` (`reference/trixi/`, Julia subprocess, DG projected to cell averages,
Well-schema HDF5) — as reference oracles for three-way cross-code verification and as fallback
data generators. The MVP data regeneration uses the reference backends, not the JAX WENO5.

## Rejected alternatives

- *Trixi as the only generator* (no own WENO): loses the learning goal and the shared-stepper
  property between simulator and model.
- *exponax*: cannot handle discontinuities.
- *JAX-Fluids as generator*: cannot express the scalar flux families.

## Consequences

Julia enters the toolchain as a pinned dependency behind `SolverBackend` only (§Amendment for
the transport). DG and FV solutions differ near
shocks at the level of each scheme's dissipation; cross-code tolerance comes from a mutual
convergence study, not a fixed number. If the time-box overruns, `TrixiBackend` generates the data
and WENO5 continues as a background item.

## Amendment (2026-10-25) — in-process Julia, not a subprocess

`TrixiBackend` calls Julia **in-process through juliacall** (change document
`2026-10-25-phase2a-trixi-backend`, D-TR; `docs/design/python-julia-repo.md` §C): arrays cross
without files, one warm Julia session per Python process, and the initial condition is
evaluated by Python on the DG nodes Julia reports (ADR-0011 §Initial conditions). The first
assessment's thread-safety objection concerned the other direction (Python called from Julia
threads) and was withdrawn after re-reading the JuliaCall documentation and PySR. What stands
from the original wording: Julia is pinned (`reference/trixi/Manifest.toml`, Julia `~1.12` via
juliapkg) and enters `src/` through `simulations/backends/trixi.py` alone. A file-based subprocess
transport is the fallback if the in-process path proves unworkable in practice (a Julia crash
takes the Python process with it; a Slurm generation job may prefer isolation) — built then,
not now.

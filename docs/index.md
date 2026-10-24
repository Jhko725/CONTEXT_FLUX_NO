---
type: index
status: current
updated: 2026-10-25
---

# `context_flux_no` documentation — index

Progressive disclosure: start here, read only what the task needs. Every document carries a
frontmatter block (`type`, `status`, `updated`, plus `verified_by` / `sources` / `id` / `branch`
where relevant) so its standing is machine-readable. `scripts/check_docs.py` enforces that every
document is listed here and has frontmatter.

`status` values — `current` (kept true at all times) · `planned` / `in-progress` / `landed` /
`merged` (change documents) · `agreed` (design settled with Joon) · `accepted` / `amended`
(decisions) · `tentative` (written by Claude, not yet checked by Joon) · `historical` (frozen).

## Read first

| Document | Type | What it answers |
|---|---|---|
| [`architecture.md`](architecture.md) | architecture | what the code is now, module by module; what it becomes; where each decision is recorded |
| [`roadmap.md`](roadmap.md) | roadmap | the single current plan: tracks, phase tables with *Done when*, Parked, the Done ledger |
| [`../CLAUDE.md`](../CLAUDE.md) | conventions | how Claude works here: session protocol, change documents, review notes, repo facts |
| [`glossary.md`](glossary.md) | glossary | the vocabulary the other documents use |

## Ledgers

| Document | What it holds |
|---|---|
| [`findings.md`](findings.md) | append-only: anything that bears on a *reported result* or constrains a later phase (F1–F10 so far) |
| [`verification.md`](verification.md) | claim → test → oracle → tolerance → status, for every numerics claim (V1–V24) |

## Design documents (`design/`)

| Document | Status | Covers |
|---|---|---|
| [`fv-conventions.md`](design/fv-conventions.md) | agreed | grid/face/stencil conventions, ghost cells and BCs, N-D handling, the three-part numerical flux, integrators, the `PDEFamily` tiers, analytic solutions, extensibility table |
| [`run-identity.md`](design/run-identity.md) | agreed | run directory layout, `RunRecord` fields and index columns, finding/loading runs, training/resume/warm-start commands, what the trainer sees |
| [`fv-core-plan.md`](design/fv-core-plan.md) | tentative | Phase 2b/2c for Joon: reading path (LeVeque, Shu, GST, Hesthaven, Clawpack, JAX/Hypothesis), module specs with invariants and tests for `stencil`/`boundary`/`divergence`/`split`/`integrate`/`analytic`, build order |
| `weno5.md` | — (Phase 2, [R]) | derivation of the WENO5 stencil coefficients, smoothness indicators, LF splitting, next to the code |

## Decision records (`decisions/`)

| ADR | Status | Decision |
|---|---|---|
| [0001](decisions/0001-tiered-pde-coupling.md) | accepted | PDE ↔ solver coupling is tiered; fluxes declare `requires` |
| [0002](decisions/0002-nd-stepping.md) | accepted | no per-dimension solver hierarchy; FD-WENO, unsplit dimension-by-dimension |
| [0003](decisions/0003-boundary-conditions-as-objects.md) | accepted | boundary conditions are objects applied by ghost-cell padding |
| [0004](decisions/0004-weno5-and-reference-backends.md) | accepted | own WENO5, time-boxed; PyClaw/SharpClaw and Trixi.jl as reference backends |
| [0005](decisions/0005-three-part-numerical-flux.md) | accepted | numerical flux = reconstruction ∘ (base − dissipation) |
| [0006](decisions/0006-operator-hierarchy-and-scale.md) | accepted | `InContextOperator` ⊃ `InContextFluxOperator`; scale wraps the conditioned module |
| [0007](decisions/0007-dataclass-configs.md) | accepted | dataclass component configs in Hydra `ConfigStore` |
| [0008](decisions/0008-four-tier-testing.md) | accepted | unit / property / regression / verification tiers |
| [0009](decisions/0009-tracks.md) | accepted | work organised as Common → MVP → Rest |
| [0010](decisions/0010-run-identity.md) | accepted | a run is a directory (`metadata.json` + whole-state checkpoints, Hydra output dir); trainer has no side effects; wandb is a mirror |

Smaller decisions without an ADR (D1 drop `DPOTEncoder`/LSK; D2 ICLR shim through Phase 3; D4
`icon/` submodule; D15 `bias-hyperinit` default; D16 extensibility interfaces reserved) are
recorded in the roadmap tables and the change documents that implement them.

## Change documents (`changes/`) — one per branch, the reviewer's map and the branch plan

| Branch | Status | Document |
|---|---|---|
| `phase0-freeze` | landed on `main` (0.7 run and 0.8 goldens pending) | [`2026-10-04-phase0-freeze.md`](changes/2026-10-04-phase0-freeze.md) |
| `phase1-prune` | landed (merge gated on the regression tier) | [`2026-10-04-phase1-prune.md`](changes/2026-10-04-phase1-prune.md) |
| `phase5-run-identity` | landed (based on `phase1-prune`; merge after it) | [`2026-10-24-phase5-run-identity.md`](changes/2026-10-24-phase5-run-identity.md) |
| `phase2a-pyclaw-backend` | planned (ledger awaiting Joon) | [`2026-10-25-phase2a-pyclaw-backend.md`](changes/2026-10-25-phase2a-pyclaw-backend.md) |

## Historical

| Document | What |
|---|---|
| [`assessment-2026-09-26.md`](assessment-2026-09-26.md) | survey of the repository at `iclr2027-submission`, with Joon's pain points P1–P10; frozen |

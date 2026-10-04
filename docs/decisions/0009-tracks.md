---
type: decision
id: ADR-0009
status: accepted
updated: 2026-09-27
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0009 — Work organised as Common → MVP → Rest

**Status**: accepted (2026-09-27).

## Context

The full plan is 6–7 focused weeks, realistically 10–12 calendar weeks alongside research. Joon
wants post-submission experiments (FiLM, reconstruction prototypes) to start on clean
infrastructure as early as possible, with a defensible minimum.

## Decision

Three tracks. **[C] Common**: P0, P1, P5, P2 (backends, core, integrators, `FluxPredictor`, cheap
classical fluxes, property tier, model migration), P3. **[M] MVP**: P4-lite (`PDEFamily` tier 0,
ICs, `DatasetRecord`, regen via reference backends, delete `SegmentLoaderBackground`), P6-lite
(rollout, metrics, run selection). **[R] Rest**: WENO5 + full verification, `JaxBackend`, figures,
P7. [R] is not started while [C]/[M] items are open, except WENO5 in its time-box alongside P3.

## Rejected alternatives

- *Strictly phase-ordered execution*: research waits ~10 weeks.
- *MVP first, infrastructure later*: rebuilds on the copy-pasted core; the point of the cleanup is lost.

## Consequences

MVP datasets are Trixi/Clawpack-generated; a later `JaxBackend` regen gets its own
`DatasetRecord` version, and `RunRecord` carries `dataset_id` so results across the two are comparable.

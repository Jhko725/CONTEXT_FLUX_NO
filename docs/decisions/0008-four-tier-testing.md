---
type: decision
id: ADR-0008
status: amended
updated: 2026-11-25
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0008 — Test tiers: unit, property, verification

**Status**: accepted (2026-09-27); amended 2026-11-25.

## Context

### Motivation

Tests covered only the baselines. The bugs found in the assessment were invariant violations
(key reuse, ignored flags, inconsistent face conventions, silent init fallthrough) that example
tests miss. Numerics has an oracle problem: correctness needs analytic solutions and cross-code
comparison [cited: Kanewala & Bieman 2014 on metamorphic testing of scientific software;
Roache 2002 on code verification by manufactured solutions; MacIver et al. 2019 for Hypothesis].

### Forces

*[not recorded]*

## Decision

`tests/unit` (examples, shapes, error paths); `tests/property` (Hypothesis + chex: what must hold
for all inputs — conservation, consistency `F̂(u,u)=f(u)`, equivariance, `vmap`==loop, monotone
fluxes, config/record round-trips);
`tests/verification` (`-m slow`, `x64`: EOC, MMS, exact discontinuous solutions, non-periodic cases,
three-way cross-code). CI runs the two fast tiers on CPU. The regression tier of the
original decision (ICLR goldens, CPU only) is removed with no replacement: nothing loads the
ICLR checkpoints any more, and no phase before Phase 3 changes a model. The tests of the
code under `legacy/` run in the fast tier from `tests/legacy` (ADR-0012). `docs/verification.md` maps each numerics
claim to its test and oracle.

## Consequences

JAX practicalities: shapes drawn from a small fixed set (retracing), `deadline=None`, bounded
finite floats, tolerance-based assertions, `derandomize=True` in CI, shrunk counterexamples pinned
with `@example`. Goldens are never regenerated without a change-document entry. A change to a
legacy model is no longer compared with the submitted checkpoints; the tag
`iclr2027-submission` is the record of what was submitted.

## Alternatives rejected

- *Unit tests only*: cannot express the invariants that failed.
- *Property tests for exact values*: no cheap property exists; that is what oracles are for.

## History

- 2026-09-27: accepted.
- 2026-11-25, `refactor-foundation`: amended. The regression tier is removed with the code
  that loaded the ICLR checkpoints.

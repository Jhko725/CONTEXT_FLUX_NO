---
type: decision
id: ADR-0008
status: accepted
updated: 2026-09-27
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0008 — Four test tiers: unit, property, regression, verification

**Status**: accepted (2026-09-27).

## Context

Tests covered only the baselines. The bugs found in the assessment were invariant violations
(key reuse, ignored flags, inconsistent face conventions, silent init fallthrough) that example
tests miss. Numerics has an oracle problem: correctness needs analytic solutions and cross-code
comparison [cited: Kanewala & Bieman, IST 56, 2014 on metamorphic testing of scientific software;
Roache 2002 on code verification by manufactured solutions; MacIver et al., JOSS 2019 for Hypothesis].

## Decision

`tests/unit` (examples, shapes, error paths); `tests/property` (Hypothesis + chex: what must hold
for all inputs — conservation, consistency `F̂(u,u)=f(u)`, equivariance, `vmap`==loop, monotone
fluxes, config/record round-trips); `tests/regression` (ICLR goldens, CPU only, `rtol 1e-5`);
`tests/verification` (`-m slow`, `x64`: EOC, MMS, exact discontinuous solutions, non-periodic cases,
three-way cross-code). CI runs the two fast tiers on CPU. `docs/verification.md` maps each numerics
claim to its test and oracle.

## Rejected alternatives

- *Unit tests only*: cannot express the invariants that failed.
- *Property tests for exact values*: no cheap property exists; that is what oracles are for.

## Consequences

JAX practicalities: shapes drawn from a small fixed set (retracing), `deadline=None`, bounded
finite floats, tolerance-based assertions, `derandomize=True` in CI, shrunk counterexamples pinned
with `@example`. Goldens are never regenerated without a change-document entry.

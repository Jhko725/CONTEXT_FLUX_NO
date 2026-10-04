---
type: decision
id: ADR-0001
status: accepted
updated: 2026-09-26
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0001 — PDE ↔ solver coupling is tiered; fluxes declare `requires`

**Status**: accepted (2026-09-26).

## Context

The finite-volume update was written eight times across model classes and target networks, each
embedding its own assumptions about the PDE (assessment §4d). A clean separation needs a statement
of *how much* a solver may depend on the PDE. In classical FV codes the stepping, component-wise
reconstruction and time integration are PDE-agnostic; only the numerical flux couples, and how
much it couples depends on the flux family [cited: Clawpack's design puts all physics in a per-PDE
Riemann solver; Shu's FD-WENO with Lax–Friedrichs splitting needs only `f(u)` and `max|f'|`].

## Decision

`PDEFamily` exposes capabilities in tiers. Tier 0 (required): `flux(u, params, axis, x)`,
`max_wave_speed(u, params, axis)`, `reflect(u, axis)`, `to_conserved`/`to_primitive`, layout and
parameter ranges, flags `has_source`, `has_nonconservative_terms`. Tier 1 (optional):
`wave_speed_estimates`, `eigensystem`. Tier 2: `riemann_flux`. Tier 3: `viscous_flux` (parabolic
terms). Every `FluxPredictor` declares `requires: frozenset[str]`; the pairing is checked at build
time against the family's capabilities. Learned fluxes have `requires = ∅`. Phase 2 implements
tier-0 fluxes only (Rusanov, Lax–Friedrichs, Godunov-scalar via Osher's formula, FD-WENO5-LF).

## Rejected alternatives

- *Per-PDE Riemann solver as the only interface* (Clawpack's model): maximal coupling; would make the
  learned flux a special case of something it is not, and would require tier-2 for every family.
- *Fully PDE-agnostic fluxes only*: loses HLLC/Roe quality for Euler where it matters; the tiers keep
  that door open without imposing it.

## Consequences

The "PDE-family-agnostic Flux NO" claim of the paper becomes structural (`requires = ∅`). Invalid
flux × PDE pairings fail at config composition. Nonconservative products and incompressible
Navier–Stokes are declared out of `fv/` scope (see `design/fv-conventions.md` §Extensibility).

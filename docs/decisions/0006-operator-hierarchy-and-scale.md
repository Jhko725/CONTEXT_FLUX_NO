---
type: decision
id: ADR-0006
status: accepted
updated: 2026-09-27
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0006 — `InContextOperator` ⊃ `InContextFluxOperator`; scale wraps the conditioned module

**Status**: accepted (2026-09-27).

## Context

Conditioning was baked into four top-level classes, each re-implementing the FV update; target
networks conflated flux prediction with time integration (P7). Plain neural-operator targets
(UNet, FNO) and finite-volume targets need to share encoder and conditioner but differ in what is
conditioned. RevIN-style scale normalisation was a sibling field; applied around the stepper it
would hand the integrator scaled states and break conservation of the physical `u`.

## Decision

`InContextOperator(encoder, conditioner, target)` with `target: u → u_next`; `InContextFluxOperator`
subclasses it with `target = FluxStepper(flux: NumericalFlux, integrator, source)` and exposes
`.flux()` for analysis. The conditioner always conditions exactly one module (`target`, or
`target.flux` in the subclass). The scale normaliser wraps that conditioned module: `Scaled(target)`
in the base class, `Scaled(target.flux)` in the subclass (normalise input, denormalise output); the
integrator always sees physical `u`.

## Rejected alternatives

- *One class with `flux=None` branch*: uniform ablation code, but conflates the two kinds of target
  — the conflation that caused P7.
- *Normaliser as a sibling slot around the whole operator*: breaks conservation; wrong by construction.

## Consequences

Hypernet/FiLM/ContextAppend are each written once. The ICLR checkpoints load through a shim
mapping the four old classes onto the hierarchy; review gate at the end of Phase 3.

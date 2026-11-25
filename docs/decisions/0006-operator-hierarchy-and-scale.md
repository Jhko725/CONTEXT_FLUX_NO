---
type: decision
id: ADR-0006
status: amended
updated: 2026-11-25
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0006 — `InContextOperator` ⊃ `InContextFluxOperator`; scale wraps the conditioned module

**Status**: accepted (2026-09-27); amended 2026-11-25.

## Context

### Motivation

Conditioning was baked into four top-level classes, each re-implementing the FV update; target
networks conflated flux prediction with time integration (P7). Plain neural-operator targets
(UNet, FNO) and finite-volume targets need to share encoder and conditioner but differ in what is
conditioned. RevIN-style scale normalization was a sibling field; applied around the stepper it
would hand the integrator scaled states and break conservation of the physical `u`.

### Forces

*[not recorded]*

## Decision

`InContextOperator(encoder, conditioner, target)` with `target: u → u_next`; `InContextFluxOperator`
subclasses it with `target = FluxStepper(flux: NumericalFlux, integrator, source)` and exposes
`.flux()` for analysis. The conditioner always conditions exactly one module (`target`, or
`target.flux` in the subclass). The scale normaliser wraps that conditioned module: `Scaled(target)`
in the base class, `Scaled(target.flux)` in the subclass (normalize input, denormalize output); the
integrator always sees physical `u`.

`InContextOperator` exposes its two stages. `generate_solver(context)` runs the encoder and
the conditioner once and returns the conditioned target, and `step(solver, u, args)` advances
a state with it. The law `step(generate_solver(u), u[-1], args) == __call__(u, args)[0]` is a
property test of Phase 3. The rebuttal branch has the two stages as
`AbstractGeneratedSolverOperator`; that class is a requirement on Phase 3 and is not brought
across as code, because its only users are the question-specific rebuttal diagnostics.

## Consequences

Hypernet/FiLM/ContextAppend are each written once. No shim loads the ICLR checkpoints into
the hierarchy: Phase 3 is written on `fv/` and replaces `legacy/`. A rollout that refreshes
its context, or that reuses one solver over many steps, calls the two stages directly.

## Alternatives rejected

- *One class with `flux=None` branch*: uniform ablation code, but conflates the two kinds of target
  — the conflation that caused P7.
- *Normaliser as a sibling slot around the whole operator*: breaks conservation; wrong by construction.

## History

- 2026-09-27: accepted.
- 2026-11-25, `refactor-foundation`: amended. The two-stage contract (`generate_solver`, `step`) is
  part of the decision, and the ICLR shim with its review gate is dropped.

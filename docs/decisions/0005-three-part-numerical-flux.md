---
type: decision
id: ADR-0005
status: accepted
updated: 2026-09-27
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0005 — Numerical flux = reconstruction ∘ (base − dissipation)

**Status**: accepted (2026-09-27).

## Context

Three stencil-flux MLPs, three dimensional-split drivers and eight FV updates were copies
(assessment §4d). Classical codes separate *reconstruction* (stencil → face states), the
*Riemann/base flux* and *dissipation* [cited: SharpClaw; Trixi's `FluxPlusDissipation`]. The
research agenda includes learned dissipation on a classical base, learned reconstruction with a
classical flux, and inspecting learned vs numerical flux — all of which are configurations of
such a decomposition. Well-balanced schemes for balance laws modify the reconstruction (Audusse et
al., SIAM J. Sci. Comput. 25, 2004), another reason to make it explicit.

## Decision

`NumericalFlux(reconstruction, base, dissipation)` with `F̂ = base(rec(u)) − dissipation(rec(u))`;
`requires` is the union of the parts'. WENO5 = `WENO5Reconstruction ∘ LaxFriedrichs`; the current
stencil MLP = `Fused` reconstruction + learned base; the v2 model = fixed `InterfaceReconstructor` +
learned 1×1 flux; the substencil-softmax prototype = learned reconstruction + classical base. Which
part the conditioner conditions is a config field. The effective-flux analysis reports parts separately.

## Rejected alternatives

- *Two parts (base, dissipation)*: hides reconstruction in `base`; cannot express well-balancing or
  the v2 split as configurations.
- *One opaque learned flux*: the status quo; every variant is a new class.

## Consequences

One more abstraction to carry; the fused learned case needs a `Fused` reconstruction returning
stencil features rather than face states.

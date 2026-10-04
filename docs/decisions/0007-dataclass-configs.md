---
type: decision
id: ADR-0007
status: accepted
updated: 2026-09-26
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0007 — Typed component configs as dataclasses registered in Hydra `ConfigStore`

**Status**: accepted (2026-09-26).

## Context

Models were built from `encoder_type: str` + `encoder_kwargs: dict` string dispatch; sub-model
signatures were invisible until `__init__`; a yaml typo (`bias_hyperinit`) silently selected the
wrong initialisation (assessment §3). Checkpoints re-instantiate via the stored Hydra dict, so a
renamed class makes them unloadable. Two data-config schemas coexisted.

## Decision

Frozen dataclasses per component with `build(key, data_spec) -> module`; `ModelConfig(encoder,
conditioner, target | flux + integrator)`; registered in `ConfigStore`, validated by
`OmegaConf.structured` (unknown key / wrong type → compose-time error). Shape-derived arguments and
`bcs` come from a `DataSpec` read off the dataset, not from yaml interpolation.
`ModelConfig.to_yaml()` is the model description stored next to the weights; `load_model =
ModelConfig.from_yaml(header).build(...)` + weights. `hydra.utils.instantiate` is no longer used
for models.

## Rejected alternatives

- *pydantic*: stronger validation and coercion, but an extra dependency and manual Hydra interop;
  Joon: only with a strong, irreplaceable reason.
- *Keep Hydra dicts + instantiate*: the status quo that produced the bugs.

## Consequences

Every (encoder × conditioner × target) combination is constructible by a property test. The
`bias_hyperinit` class of bug becomes impossible.

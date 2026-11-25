---
type: decision
id: ADR-0012
status: accepted
updated: 2026-11-25
verified_by: joon (planning session of 2026-11-25, decision 2 and questions Q8, Q12, Q13, Q15)
---

# ADR-0012 — Code awaiting its rewrite lives in `legacy/`; a clean list scopes the strict checks

**Status**: accepted (2026-11-25). Implemented on `refactor-foundation`.

## Context

### Motivation

- The strict checks were off everywhere. Docstring rules, `ty` and runtime shape checks
  could not be turned on for the whole tree, because the models and the `nn` utilities
  fail them in hundreds of places, and so they ran on no part of it. At `450f31d`, ruff's
  docstring and complexity rules reported 227 findings and `ty` 134 diagnostics on the
  paths that are clean today.
- The models are rewritten in roadmap Phase 3 on the finite-volume core. Cleaning them
  before that rewrite would be work on code with a known end.
- Roadmap principle 3 said "No `legacy/` directory" and relied on the tag
  `iclr2027-submission` as the archive. The tag archives what was submitted. It does not
  say which code in the working tree is held to the current standard.

### Forces

- The models must keep training until Phase 3 replaces them, so the code that awaits its
  rewrite stays importable and tested.
- A checker has to learn the scope from one place, or the scripts, the `justfile` and the
  test configuration drift apart.

## Decision

The package has two kinds of code, told apart by path.

`src/context_flux_no/legacy/` holds the code that awaits its rewrite: `models/`, `nn/` and
`plotting/`. It stays importable and otherwise untouched. A change there is an import
path, or a fix that Joon asked for. Its tests live in `tests/legacy/`, run unchanged in
the fast tier, and leave with the code they test.

Every other path of the package is clean. The clean paths are listed once, as `clean`
under `[tool.context_flux_no]` of `pyproject.toml`, with `tests/unit`,
`tests/verification`, `tests/helpers.py`, `tests/conftest.py` and `scripts` beside the
package paths. The strict checks read that list with `tomllib`: the docstring and
complexity rules of ruff, `ty`, the container, dependency and docstring checks, and the
jaxtyping import hook of the tests.

A module joins the clean list by leaving `legacy/`, in the branch that rewrites it, and
never returns. No clean path imports `legacy/`; a model reaches the evaluation code
through the `Operator` protocol of `evaluation/rollout.py`, and the training entry point
builds a model from a config's `_target_`.

## Consequences

- Every import path of a model changed once: `context_flux_no.models…` became
  `context_flux_no.legacy.models…`, in the code and in the `_target_` of every model
  config. A `config.yaml` saved by an earlier run names the old path.
- Phase 3 writes the operator hierarchy as new clean modules and deletes the legacy ones;
  there is no migration of a legacy module in place.
- The evaluation package survives Phase 3 unchanged, because it depends on the protocol
  and not on a model class.

## Alternatives rejected

- *One standard for the whole tree*: this was the state before the branch, and it kept
  every strict check off.
- *Per-file ignore lists in each tool's configuration*: the scope would then be held
  once per tool, and a file could be clean for one checker and exempt for another.
- *Delete the models now and restore them from the tag when needed*: the repository
  could not train between this branch and Phase 3.

## History

- 2026-11-25, `refactor-foundation`: accepted.

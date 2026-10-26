---
type: change
status: landed
updated: 2026-10-26
branch: conventions-docs
roadmap: none (repository hygiene)
---

# 2026-10-25 — `conventions-docs`

## Summary

Turns the American English rule into two checks that run in pre-commit and CI, and
applies them to the tree once. `codespell` catches the irregular spellings and the typos;
`scripts/check_spelling.py` catches the productive `-ise` families that no word list
ships. The sweep touched prose, comments and docstrings in 44 files and renamed one test.
No roadmap item.

The branch was first cut on 2026-10-25 from the `phase2a-pyclaw-backend` tip and carried
the split of the conventions into four layers (global `CLAUDE.md`, the `code-conventions`
skill, `docs/coding-standards.md`, this repository's `CLAUDE.md`). That split reached
`main` through the Phase 2a review, so on 2026-10-26 the branch was recreated from `main`
with only the remaining work: the checks, their wiring and the sweep. Its earlier commit
that removed the `copy.replace` ban was not carried over; `main` has had a Python 3.13
floor and the `TID251` ban since the Phase 2a review, and the standards document keeps
that row.

## Files

| File | Change |
|---|---|
| 44 files under `src/`, `tests/`, `scripts/`, `docs/` (not `docs/changes/`) and `README.md` | `style:` British spellings and typos in prose, comments and docstrings, as reported by the two checks; `cell-centred` → `cell-centered` in `simulations/grid.py`, `docs/design/fv-conventions.md`, `docs/design/fv-core-plan.md` and `docs/findings.md` (found by grep: neither check looks inside a hyphenated compound) |
| `tests/unit/test_pyclaw_backend.py` | `test_grid_matches_pyclaw_centres` → `test_grid_matches_pyclaw_centers`; run, green |
| `pyproject.toml` | `[tool.codespell]`: builtin `clear,en-GB_to_en-US`, the skip list with a comment on why each tree is out of scope, `Manifest.toml` among them |
| `scripts/check_spelling.py` | files named explicitly respect the skip list, so a hook and a root walk agree; `docs/changes/` skipped like `codespell` does |
| `.pre-commit-config.yaml` | `codespell` 2.4.3 and `check_spelling.py` as hooks |
| `.github/workflows/test.yml` | `uvx codespell@2.4.3` and `check_spelling.py` before pytest |
| `CLAUDE.md` | the hook list, which went stale as hooks were added, now points at `.pre-commit-config.yaml` |
| `docs/coding-standards.md` | the checks table gains a *Runs in* column; the scope of the spelling checks and the hyphenated-compound caveat recorded; the pre-sweep finding counts removed |
| `docs/index.md` | the branch row |
| `docs/changes/2026-10-25-conventions-docs.md` | this document |

## Design

- **Two spelling checks, because the rule has two halves.** `codespell`'s
  `en-GB_to_en-US` data is a 535-entry word list, and measured against this repository's
  vocabulary it misses `discretisation`, `parametrised`, `denormalise` and `amortisation`; <!-- codespell:ignore -->
  `breame`, the other candidate, misses the same four [code: 2026-10-25]. A list cannot
  close over productive morphology. So `codespell` keeps the half that a list is good at,
  the irregulars and the typos, and `scripts/check_spelling.py` takes the half that a rule
  is good at: `-ise` → `-ize` with an exception list of the words where `-ise` is also
  correct in American English, which is finite where the inclusion list is not.
  *Rejected*: a hunspell differential (`en_GB` minus `en_US`), which flags every domain
  term and would need a personal word list larger than the exception list it replaces.
- **`docs/changes/` is out of scope for both checks.** A change document is the record of
  its branch as it was reviewed; respelling it would change a historical record for no
  reader. The first sweep on this branch had edited four of them before the script's skip
  list was aligned with `codespell`'s; those edits were reverted.
- **The sweep is one `style:` commit, the test rename a second, the wiring a third**, so
  that `git log -p` on a test file shows a rename and nothing else, and the checks are
  turned on by a commit whose first CI run is green.
- **The Julia `Manifest.toml` is skipped by name.** `codespell` rewrote the package names
  `Missings` and `RCall` on the first run, which would have broken the Julia project; <!-- codespell:ignore -->
  the lock files (`*.lock`) were already skipped for the same reason.

## Bugs fixed

None.

## Findings

None that bear on a reported result.

## Tests

No test added; `uvx codespell@2.4.3` and `uv run python scripts/check_spelling.py` both
report 0 findings, and the renamed test runs green with the rest of
`tests/unit/test_pyclaw_backend.py`. Fast tier after the sweep: 107 passed, 1 skipped, 1 xfailed
(the baseline from the Phase 2a-trixi merge); `check_docs --files --base main` and
`import_all` clean.

## Open issues

- `scripts/check_spelling.py` has no test of its own; its behavior is pinned by the
  measurement in the standards document (13 of 13 on a line of British forms, 0 of 37 on
  a line of words where `-ise` is correct).
- The `code-conventions`, `clear-prose`, `diagnose`, `numerical-method`, `grilling` and
  `retro` skills live in Joon's account, not in this repository, so a clone gets the
  standards document but not the reasoning behind it.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `style:` sweep over 44 files (prose, comments, docstrings) | Confirmed. | None |
| `tests/unit/test_pyclaw_backend.py` — test renamed | Looks good. | None |
| `pyproject.toml` `[tool.codespell]`; `scripts/check_spelling.py` skip list; hooks; CI; `CLAUDE.md` hook pointer | Confirmed. | None |
| `docs/coding-standards.md` — checks table, scope, compound caveat | Looks good. | None |

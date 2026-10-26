---
type: change
status: in-progress
updated: 2026-10-25
branch: conventions-docs
roadmap: none (repository hygiene)
---

# 2026-10-25 — `conventions-docs`

## Summary

Splits this repository's conventions across the four layers they belong to, so that each
one is stated once. The universal rules — jit and vmap, docstrings, parameter leaves,
American English, annotations — now live in Joon's global `CLAUDE.md` and in a
`code-conventions` skill that loads on demand; this repository's instantiation of them —
tiers and commands, the goldens and tolerance policy, scope, and the checks that replace
prose — moves to a new `docs/coding-standards.md`; and `CLAUDE.md` keeps the session
protocol, the change-document contract and the facts Claude cannot infer from the code.
No roadmap item, no code change. Branched from the `phase2a-pyclaw-backend` tip, the
latest work in the repository on this date, so that the pruning applies to the current
`CLAUDE.md` rather than to a stale copy.

## Files

| File | Change |
|---|---|
| `CLAUDE.md` | pointer paragraph to the global rules, the skill and the new standards document; tiers and goldens delegated to it; the Markdown bullet compressed to the rule the checker enforces; a baseline-before-editing line added |
| `docs/coding-standards.md` | **new** — tiers and commands, goldens and tolerances, scope, the checks that replace prose, deliberate exceptions |
| `docs/index.md` | the new document listed under *Read first*; the branch row in the change-document table |
| `scripts/check_spelling.py` | **new** — the rule for the `-ise` / `-isation` / `-yse` families, with the closed exception list |
| `docs/changes/2026-10-25-conventions-docs.md` | this document |

## Design

- **Four layers, one home per rule.** A rule that applies to every edit in every
  repository is loaded every session, so it belongs in the global `CLAUDE.md` and has to
  earn its place there; a mechanically checkable rule belongs in a check, because prose
  is advisory and a check is not; a rule whose reasoning has to stay arguable belongs in
  a skill, loaded when design work starts; a fact particular to this repository belongs
  here. *Rejected*: keeping everything in `CLAUDE.md` — it duplicated most of
  `deep_isochron/CLAUDE.md` line for line, and the duplicate drifted (the jit rule and
  the docstring convention were added there and never here).
- **Why this branch and not `main`.** The convention files are cross-cutting and the
  repository's own rule is one change document per branch, so they get their own branch
  rather than riding along with Phase 2a. It is cut from the `phase2a-pyclaw-backend`
  tip because that is the newest state; it holds no Phase 2a-specific content, so it
  rebases onto `main` or onto `phase5-run-identity` cleanly if that order is preferred.
- **The spelling rule becomes two checks, because the rule has two halves.** `codespell`
  is not a dictionary: its `en-GB_to_en-US` data is a 535-entry word list, and measured
  against this repository's vocabulary it misses `discretisation`, `parametrised`, <!-- codespell:ignore -->
  `denormalise` and `amortisation`; `breame`, the other candidate, misses the same four <!-- codespell:ignore -->
  [code: 2026-10-25]. A list cannot close over productive morphology. So `codespell`
  keeps the half a list is good at — the irregulars and the typos, which is how
  `splitted`, `recommand`, `quantites` and `meybe` were found — and <!-- codespell:ignore -->
  `scripts/check_spelling.py` takes the half a rule is good at: `-ise` → `-ize` with an
  exception list of the words where `-ise` is also correct in American English, which is
  finite where the inclusion list is not. *Rejected*: a hunspell differential (`en_GB`
  minus `en_US`), genuinely morphological but flagging every domain term, which would
  need a personal word list larger than the exception list it replaces.
  The cleanup still needs two commits, because the British spelling of *cell center*
  reaches a test name as well as docstrings — see *Open issues*.
- **What `CLAUDE.md` keeps.** The session protocol, the change-document section list with
  its status lifecycle, the end-of-session checklist, and the facts: the submodule, the
  notebook filter, the tags policy, the frontmatter vocabulary, the provenance labels,
  and the pointer to `docs/design/fv-conventions.md`. Those are either navigation or
  things no check can supply.

## Bugs fixed

None; no code changed. One pre-existing defect is recorded rather than fixed: `CLAUDE.md`
used the British spelling of *behavior*, in the file that states the American-English
rule.

## Findings

None that bear on a reported result. Nothing appended to `docs/findings.md`.

## Tests

None added; no code changed, so the baselines recorded in the Phase 2a and Phase 5
change documents still hold. `uv run python scripts/check_spelling.py` is itself new.
`uv run python scripts/check_docs.py` must pass: the new
document carries frontmatter, is listed in `docs/index.md`, and the *Files* table above
matches `git diff --stat phase2a-pyclaw-backend...HEAD`.

## Open issues

- The checks are written down but not wired up. `codespell` reports 80 findings on this
  branch and `check_spelling.py` 38, overlapping: about 60 in prose and docstrings, which
  are one `style:` commit, and thirteen
  occurrences of the British spelling of *cell center*, which besides docstrings in
  `simulations/grid.py` and `simulations/pde/{base,burgers}.py` reaches a test function
  name in `tests/unit/test_pyclaw_backend.py`. Renaming a test is a code change and
  belongs in its own commit.
- The `code-conventions`, `diagnose`, `numerical-method`, `grilling` and `retro` skills
  live in Joon's account rather than in this repository, so a clone gets the standards
  document but not the reasoning behind it. Copies under `.claude/skills/` are the
  alternative if that matters.
- `docs/verification.md` is untouched: no numerics claim gained or lost a test.
- `scripts/check_spelling.py` has no test of its own. Its behavior is pinned only by the
  measurement in the Design section: 13 of 13 on a line of British forms, 0 of 37 on a
  line of words where `-ise` is correct.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `CLAUDE.md`: pointer paragraph; tiers and goldens delegated; Markdown bullet compressed; baseline line added | | |
| `docs/coding-standards.md` (new): tiers, goldens and tolerances, scope, checks, exceptions | | |
| `docs/index.md`: the new document under *Read first*, and the branch row | | |
| `scripts/check_spelling.py` (new): the `-ise` / `-isation` / `-yse` rule with its exception list; 38 findings here, none of them false | | |
| `docs/changes/2026-10-25-conventions-docs.md`: this document | | |

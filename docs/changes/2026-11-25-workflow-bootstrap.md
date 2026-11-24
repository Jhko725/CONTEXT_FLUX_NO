---
type: change
status: landed
updated: 2026-11-25
branch: workflow-bootstrap
roadmap: none (prerequisite of `refactor-foundation`)
---

# 2026-11-25 — `workflow-bootstrap`

## Summary

Adds the files that a Claude session reads when it starts: Joon's working rules, the
`implementer` subagent, and two hooks. They come from `UniversalForceReconstruction` at
`b80a341`. They are on their own branch because an agent definition is read at session
start, so the unattended run of `refactor-foundation` needs them on `main` before it
begins. The rest of that repository's workflow (the `justfile`, `prek`, the checks, the
document conventions) is the work of `refactor-foundation` itself.

## Files

- `.claude/rules/working-rules.md` — copied: the working rules of Joon's global `CLAUDE.md`.
- `.claude/rules/python.md` — copied: the design, Python and JAX rules, loaded when a Python file is read.
- `.claude/agents/implementer.md` — adapted: the subagent that implements one commit of a handoff document. The archive and notebook boundaries of the source are replaced by the `legacy/` boundary of this repository.
- `.claude/hooks/block_history_rewrite.py` — copied: blocks `git rebase`, a force-push, and an amend of a pushed commit.
- `.claude/hooks/block_push.py` — copied: blocks `git push` while the `implementer` runs.
- `.claude/settings.json` — registers the history hook.
- `CLAUDE.md` — points at `.claude/rules/`; states the history rule and names the agent.
- `docs/changes/2026-11-25-workflow-bootstrap.md` — this document; `docs/index.md` — its row.

## Design

- **The rules are copied into the repository.** A cloud session does not load the global `CLAUDE.md`, and until now the rules reached a session only through the Project. Rejected: a pointer to the Project, which a session started from the repository alone cannot follow.
- **The hooks are copied unchanged**, each with a first line naming its source and commit. The amend rule of `block_history_rewrite.py` asks the remote directly, which matters in the shallow clones of a cloud session.
- **`implementer.md` names `src/context_flux_no/legacy/` before that directory exists.** `refactor-foundation` creates it; the boundary is stated here so that the agent file does not change in the middle of the run that uses it.

## Bugs fixed

None.

## Findings

None.

## Tests

No test added. Both hooks were exercised by hand on 2026-11-25: `block_history_rewrite.py` exits 2 on `git push --force origin x` and 0 on `git status`; `block_push.py` exits 2 on `cd x && git push origin y`.

## Open issues

- The hooks have no automated test in this repository. `UniversalForceReconstruction` has none either.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `.claude/rules/working-rules.md`, `.claude/rules/python.md` — copied rules | | |
| `.claude/agents/implementer.md` — the implementer subagent, adapted | | |
| `.claude/hooks/block_history_rewrite.py`, `.claude/hooks/block_push.py`, `.claude/settings.json` — hooks | | |
| `CLAUDE.md` — pointer to the rules, the history rule, the agent | | |

---
type: reference
status: current
updated: 2026-11-25
sources:
  - "UniversalForceReconstruction docs/document-conventions.md at b80a341"
  - "Handoff of refactor-foundation, section 5 (rules T1 and T2)"
---

# Document conventions

This document says what each kind of document under `docs/` is for, the form each one takes, and which check enforces what. The prose rules are the `clear-prose` skill's. The conventions for code, tests, and docstrings are in [`coding-standards.md`](coding-standards.md).

## What goes where

| Place | Type | Holds | Read it when |
|---|---|---|---|
| `architecture.md` | architecture | the module map, the data flow, and where each decision is recorded | starting any task |
| `roadmap.md` | roadmap | the single current plan, phase by phase, and the append-only Done ledger | choosing what to do next |
| `glossary.md` | glossary | every term once: its name, its symbol in the papers, its code name | naming anything |
| `design/` | design | the reasoning behind the code, one document per method, convention or problem | implementing or checking a method |
| `decisions/` | decision | numbered ADRs: why the structure is what it is | touching the area an ADR covers |
| `changes/` | change | one document per branch, the reviewer's map of the change | reviewing or continuing a branch |
| `plans/` | plan | the handoff of each implementation round, frozen when committed | reading why a branch was built the way it was |
| `papers/` | paper | one record per paper: what produces each figure, which code carries each equation and parameter | editing a paper or the code it reports |
| `experiments/` | experiment | one report per question: the runs that answer it and what they show | planning or citing a result |
| `findings.md` | findings | the append-only ledger of anything that bears on a reported result | before trusting a result |
| `verification.md` | verification | one row per numerical claim: its test, its oracle, its tolerance | making or checking a claim |
| `coding-standards.md`, `document-conventions.md` | reference | where each convention is enforced, for code and for documents | writing code or a document |
| `index.md` | index | every document above, with its status | finding a document |

A dataset is described by the `DatasetRecord` JSON stored with it (ADR-0011), not by a document under `docs/`.

## Frontmatter and status

Every document opens with a frontmatter block: `type`, `status`, `updated` (the ISO date of the last edit, changed in the same commit as the text), and where relevant `verified_by`, `sources`, `id`, `branch` (every change document) and `kind` (every design document). The block is valid YAML, and `sources` is a block list with every item quoted.

`type` and `status` are closed vocabularies, and `scripts/checks/check_docs.py` fails on a value outside them. Each type allows these statuses:

| Type | Allowed status | Meaning |
|---|---|---|
| architecture, roadmap, glossary, findings, verification, reference, index, paper | `current` | kept true at all times |
| design | `draft for review`, `tentative`, `agreed` | written and awaiting Joon; written by Claude and unchecked; settled with Joon |
| decision | `accepted`, `amended`, `superseded` | in force; in force and changed in place; replaced by a later ADR, which it names |
| change | `planned`, `in-progress`, `landed`, `merged` | written before the work; work under way; work complete on the branch; branch merged |
| experiment | `planned`, `running`, `concluded` | question and setup written; runs under way; results and reading written |
| plan | `planned`, `executed` | committed at the start of its round; the round's documents are written |

A `concluded` experiment, a `merged` change document, and a `superseded` ADR are history. Their text is not rewritten; a correction is appended as a dated line.

## Required sections

Each type has required headings, in a fixed order, so that a heading means the same thing in every document of the type. A section with nothing to say is one sentence, never left out. The required headings live in one table in `scripts/checks/check_docs.py`, which fails on a missing or reordered heading. The same check asserts that each form of that table is quoted by exactly one template of this document, so the two cannot drift.

## Change documents

Every branch has a companion document `changes/<YYYY-MM-DD>-<branch>.md`, committed with the code and updated in the same commit as any later change on the branch. It is the reviewer's map of the change, not a changelog.

```markdown
## Summary        what the branch does and why
## Design         the decisions not obvious from the code, each with the rejected alternative, and one line on whether the change can be undone
## Bugs fixed     with the symptom
## Tests          what they pin down and how to run them, with exact counts
## Open issues    with the reason each is open
## Review notes   a log of review rounds, each a table Change | Thoughts | Modifications; the only list of changes
```

- The sections are *Summary*, *Design*, *Bugs fixed*, *Tests*, *Open issues*, *Review notes*, in that order, and there is no *Files* section. *Review notes* is a log of rounds, each under ``### Round <n>: from `<commit>` `` with a table `Change | Thoughts | Modifications`. A *Change* cell is one backticked path, a dash, and what changed, on one line; a path that ends in `/` covers every changed file under it. Every file a commit touches gets a row in the open round in that commit. A round is closed once one of its *Thoughts* cells is filled, and its rows are never edited again. Round 1 starts from the merge base with `origin/main`.
- *Tests* is the baseline the next session reads. A test's outcome is stated only after that test has run on the code the document describes.
- The *Thoughts* and *Modifications* cells are Joon's, and Claude edits them only when asked.
- When a handoff document planned the branch, the frontmatter's `sources` links its committed copy under `plans/` by relative path.
- A `merged` change document keeps Joon's words verbatim, and the spelling checks skip `changes/`.
- The six change documents dated before 2026-11-25 predate this form. `PRE_FORMAT_CHANGES` in `check_docs.py` lists them, and the heading rule skips them. The list never grows.

## Plans

`plans/<YYYY-MM-DD>-<branch>[-<suffix>].md` is the handoff of one implementation round. A handoff is committed once and then frozen. The planning session writes and revises a handoff in the Project. The implementing session commits it verbatim under `plans/` at the start of the round, and nobody edits its body afterward. Only its frontmatter `status` changes, from `planned` to `executed`, when the round's documents are written. The change document's `sources` links to it by relative path.

```markdown
## 1. Header                     goal, date, who settled the design, repository, branch and tip, earlier documents
## 2. Out of scope and assumptions   what the round leaves alone, and the writer's assumptions, each confirmed or not
## 3. What the implementing session does   a table of the units and who implements each
## 4. Before anything            access, the tip, the environment, and what to read first
## 5. Rule text                  the rules not yet in the repository, verbatim
## 6. Units                      one subsection per unit: goal, model, what it may change, items, evidence, decisions
## 7. Baseline                   the commands and their outputs before the first unit
## 8. The documents unit         the documents written last, after the code
## 9. Gates                      what is run between units, and the one condition that stops the run
## 10. Decisions and where they are recorded   each decision and the document that records it
```

- The frontmatter block stands above the verbatim body, and its `sources` names the handoff's Project path.
- `check_md_math.py` reads only the frontmatter of a plan, and `just pending` skips `plans/`: a plan is a record of what was said, and its line numbers refer to the commit it names.

## Decision records

An ADR describes the current design, and its history follows it. An ADR earns its place when the decision is hard to reverse, surprising without context, and came from a real trade-off. Otherwise the change document's *Design* section is enough.

```markdown
## Context
### Motivation    one bullet per point, each with its evidence, or *[not recorded]*
### Forces        what any solution had to respect, or *[not recorded]*
## Decision       the current design
## Consequences
## Alternatives rejected   superseded designs too, marked *Superseded* with date and reason
## History        one dated line per change, with its branch
```

A change to a decision rewrites *Decision* and *Consequences* in the same commit, and the status becomes `amended`. A change that reverses the decision is a new ADR.

Every ADR keeps one path for its whole life, `decisions/<NNNN>-<slug>.md`, and a number is never reused. A superseded ADR is not moved, because other ADRs, docstrings, and merged change documents link to it, and a merged change document is not rewritten. Its standing is shown in three places instead:

- **The index groups decisions.** `index.md` lists them in two tables, *In force* and *Superseded*.
- **The frontmatter links both ways.** The superseded ADR carries `superseded_by: ADR-<NNNN>`, and its successor carries `supersedes: ADR-<NNNN>`.
- **The first line redirects.** A superseded ADR opens, above *Context*, with one sentence naming its successor.

`scripts/checks/check_docs.py` fails when the status, the two frontmatter fields, the opening line, and the index table disagree.

## Design documents

A design document under `design/` is one of three kinds, named in its frontmatter as `kind`. A **method** document takes a numerical method from the literature to tested code and follows the `numerical-method` skill. A **derivation** fixes conventions, derives the quantities in one convention stated once, and ends with the API and the tests it implies. A **note** states a problem, what was measured, and the rule adopted.

A method document has six numbered sections:

```markdown
## 1. The problem        what is computed, from what, to what accuracy; the conventions fixed up front
## 2. What exists        the survey: the routes, what each costs and assumes, where each fails
## 3. The method         the derivation in one convention; one subheading per stage, in the order the code runs them
## 4. Exact and approximated   what is exact, what is approximated, and the bound on each approximation
## 5. Oracles            the independent routes to the same number, each with its tolerance, its entry in `tests/helpers.py`, and its test
## 6. Limits             what the method does not do, and which later item does
```

- The module docstring of the implementation names the stages of section 3 in the same order.
- A *Decisions* section exists while the document is a `draft for review`, each decision phrased so that "yes" accepts the recommendation. Once taken, a decision lives in the ADR or the change document, and the section is deleted.

## Paper records

`papers/<paper>.md` is the one place that ties a paper to the code. It exists so that a difference between the draft and the code is found by reading one table, not by an audit.

```markdown
## 1. Identity          title, where the draft lives, the tag and lock file frozen at submission
## 2. Figures           a table: Figure | Panel | Command | Runs and datasets | Output file | State
## 3. Equations         a table: Equation | What it states | Code symbol | Test
## 4. Parameters        a table: Quantity | Value in the draft | Configuration key or constant | Agrees
## 5. Numerical claims  every number in the running text: Claim | Value | Source
## 6. Open differences  each place where the draft and the code disagree, with who decides and by when
```

- A row of *Figures* is complete when its command regenerates the output file from the named runs. *State* is `planned`, `regenerates`, or `frozen`.
- Every *Source* in section 5 carries a `[measured]` or `[cited]` label.
- A difference in section 6 is closed by a commit or a draft revision, named in the row, and the row then moves to the table it belongs in.
- A paper record is `current`: an edit to the draft or to the reported code updates it in the same commit.

## Experiment reports

`experiments/<YYYY-MM-DD>-<slug>.md` answers one question with runs. It is written in two steps: sections 1 and 2 before the runs (`planned`), sections 3 to 5 after (`concluded`).

```markdown
## 1. Question       one sentence, and what answer would change what
## 2. Setup          commit, configurations, datasets, seeds, and the run identifiers
## 3. Results        tables and figures; every number carries `[measured: <run id>]`
## 4. Reading        what the results show, and what they cannot show
## 5. Consequence    what changes: a paper-record row, a roadmap item, a finding, or nothing
```

One report covers one question. A comparison of two dissipation forms is one report; adding a third form later is a new report that cites the first.

## Glossary and notation

`glossary.md` names every term once. A name used in code, a document, or a figure label is the glossary's name.

| Column | Holds |
|---|---|
| Term | the one name used in prose |
| Symbol | the symbol in the papers |
| Code name | the identifier in the package |
| Definition | one sentence |
| Avoid | the synonyms that are not used |

The glossary is still a list of terms with their definitions; it takes this table form when it is next rewritten. The *Avoid* column is what the prose pass removes.

## Findings and verification

`findings.md` is an append-only table: number, date, finding, evidence, what it affects, action and state. It holds anything discovered that bears on a reported result or constrains a later branch. Cleanup goes in the roadmap.

`verification.md` has one row per numerical claim: number, claim, test, oracle, tolerance, tier, state. A claim with no row is unverified. A row is added when the claim is made and marked green, with the commit, when the test passes.

## Provenance labels

A claim in a design document, a paper record, or an experiment report carries a label in square brackets:

| Label | Means |
|---|---|
| **[cited]** | from a named source |
| **[cited via X]** | read in X, not in the source it credits |
| **[code]** | verified in the tree at the moment of writing |
| **[deduced]** | Claude's inference |
| **[measured: <run id>]** | a number produced by the named run |
| **[data: <path>]** | observed in the named file |

A `[code]` claim is re-read when the text is edited, not recalled.

## Unconfirmed decisions

A decision that Claude takes without Joon is marked **[unconfirmed]** where it is stated. `just pending` lists every marker. A document whose status is `agreed`, `accepted`, `amended` or `merged` carries none.

`check_docs.py` fails on a marker inside a document with one of those statuses, so that nothing unconfirmed is recorded as settled. The marker written in code font is a mention of it and does not count.

## References

The repository has one bibliography, `references.bib`, and one in-text form. The key is `Author Year` for one author, `Author & Author Year` for two, `Author et al. Year` for three or more, with a letter suffix on a collision. A locator follows the key after a comma: `LeVeque 2002, Ch. 7`, `Gottlieb et al. 2001, §2`. Every key resolves to an entry in `references.bib`. Documents carry no reference list of their own; their frontmatter `sources` lists keys. A `sources` item that is not a work, such as `"code"`, `"joon"`, a Project document or a library's documentation, is not in that form and has no entry.

The page `references.md` is generated from `references.bib` by `scripts/docs/generate_references.py` (`just references`) and is not edited by hand. The generator derives each in-text key from the entry's `author` and `year` fields, and gives colliding keys their letter suffixes in the order of the entries' titles. An author list that ends in `and others` gives the `et al.` key whatever its length, for a work whose later authors the repository does not know. Names and titles are written in Unicode in the `.bib` file, since the generator removes braces and decodes no other LaTeX. An entry holds only fields that a document or a source in the repository states; a field nobody has checked is left out rather than filled from memory.

`check_references.py` fails on a key that has no entry, on a relative link whose target does not exist, and when the page differs from the generator's output. It reads the documents, the package, the tests and the scripts, and skips `changes/`, `plans/`, `src/context_flux_no/legacy/` and `tests/legacy/`: the first two are records of what was said, and rule R7 leaves the last two untouched. A line that carries `references:ignore`, in Markdown inside an HTML comment, is not read; it is for a year that is not a citation, such as a conference name or a month.

## Markdown that renders in both previews

GitHub runs Markdown before MathJax, and VS Code's preview parses inline math first. `scripts/checks/check_md_math.py` enforces these rules:

- An inline formula never crosses a line.
- Display math is a standalone block: `$$` alone on a line, the formula, `$$` alone on a line.
- Inside inline math, no backslash followed by ASCII punctuation.
- No `*` inside inline math; use `\ast`.
- Spaces around `<` and `>`.
- No `_` immediately after `}`, `)`, or `|`.

## Checks

| Rule | Check | Runs in |
|---|---|---|
| both Markdown previews; the frontmatter parses and carries `type`, `status` and `updated` | `check_md_math.py` | `prek`, CI |
| `type` and `status` are in the closed vocabularies; `updated` is an ISO date; a change document carries `branch` | `check_docs.py` | `prek`, CI |
| required headings present and in order | `check_docs.py` | `prek`, CI |
| `index.md` links every document | `check_docs.py` | `prek`, CI |
| the Done ledger is chronological | `check_docs.py` | `prek`, CI |
| a superseded ADR's status, frontmatter, opening line, and index table agree | `check_docs.py` | `prek`, CI |
| no `[unconfirmed]` in a settled document | `check_docs.py`; `just pending` lists all | `prek`, CI |
| the templates in this document match the check's table | `check_docs.py` | `prek`, CI |
| American English | `codespell`, `check_spelling.py` | `prek`, CI |

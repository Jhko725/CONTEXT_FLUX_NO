"""Enforce the documentation bookkeeping rules of CLAUDE.md.

Checks (all under docs/):
  1. every *.md has a frontmatter block with `type`, `status`, `updated` (ISO date);
  2. `status` is one of the allowed values; a change document also has `branch`;
  3. docs/index.md links every document (by relative path);
  4. the Done ledger in docs/roadmap.md is chronological (dates non-decreasing);

Usage:
  uv run python scripts/checks/check_docs.py
Exit status 1 on any problem; problems are printed one per line.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ALLOWED_STATUS = {
    "current",
    "planned",
    "in-progress",
    "landed",
    "merged",
    "agreed",
    "accepted",
    "amended",
    "tentative",
    "historical",
}
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def frontmatter(path: Path) -> dict[str, str] | None:
    m = FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not m:
        return None
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def check_frontmatter(problems: list[str]) -> None:
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(ROOT)
        fm = frontmatter(path)
        if fm is None:
            problems.append(f"{rel}: missing frontmatter block")
            continue
        for key in ("type", "status", "updated"):
            if key not in fm:
                problems.append(f"{rel}: frontmatter lacks `{key}`")
        status = fm.get("status", "").split(";")[0].strip()
        if status and status not in ALLOWED_STATUS:
            problems.append(f"{rel}: status `{status}` not in {sorted(ALLOWED_STATUS)}")
        try:
            date.fromisoformat(fm.get("updated", ""))
        except ValueError:
            problems.append(f"{rel}: `updated` is not an ISO date")
        if fm.get("type") == "change" and "branch" not in fm:
            problems.append(f"{rel}: change document lacks `branch`")


def check_index(problems: list[str]) -> None:
    index = DOCS / "index.md"
    if not index.exists():
        problems.append("docs/index.md missing")
        return
    text = index.read_text(encoding="utf-8")
    for path in sorted(DOCS.rglob("*.md")):
        if path == index:
            continue
        rel = path.relative_to(DOCS).as_posix()
        if rel not in text:
            problems.append(f"docs/index.md does not list {rel}")


def check_done_ledger(problems: list[str]) -> None:
    roadmap = DOCS / "roadmap.md"
    if not roadmap.exists():
        problems.append("docs/roadmap.md missing")
        return
    text = roadmap.read_text(encoding="utf-8")
    if "## Done" not in text:
        problems.append("docs/roadmap.md has no `## Done` ledger")
        return
    ledger = text.split("## Done", 1)[1]
    dates = []
    for line in ledger.splitlines():
        m = re.match(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", line)
        if m:
            dates.append(m.group(1))
    if dates != sorted(dates):
        problems.append(
            "docs/roadmap.md Done ledger is not chronological (append at the end)"
        )


def main() -> int:
    problems: list[str] = []
    check_frontmatter(problems)
    check_index(problems)
    check_done_ledger(problems)

    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

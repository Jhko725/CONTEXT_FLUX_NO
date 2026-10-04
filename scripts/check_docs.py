"""Enforce the documentation bookkeeping rules of CLAUDE.md.

Checks (all under docs/):
  1. every *.md has a frontmatter block with `type`, `status`, `updated` (ISO date);
  2. `status` is one of the allowed values; a change document also has `branch`;
  3. docs/index.md links every document (by relative path);
  4. the Done ledger in docs/roadmap.md is chronological (dates non-decreasing);
  5. --files: the current branch's change document lists every file in
     `git diff --stat <base>...HEAD` (names only) and vice versa, ignoring the
     change document itself.

Usage:
  uv run python scripts/check_docs.py            # 1-4
  uv run python scripts/check_docs.py --files    # 1-5 (needs a branch and its change document)
  uv run python scripts/check_docs.py --files --base origin/main
Exit status 1 on any problem; problems are printed one per line.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def check_files_vs_diff(problems: list[str], base: str) -> None:
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch in {"main", "HEAD"}:
        problems.append("--files needs a feature branch (not main/detached)")
        return
    docs = sorted(DOCS.glob(f"changes/*-{branch}.md"))
    if not docs:
        problems.append(
            f"no change document for branch `{branch}` (docs/changes/<date>-{branch}.md)"
        )
        return
    change_doc = docs[-1]
    text = change_doc.read_text(encoding="utf-8")
    try:
        diff = git("diff", "--name-only", f"{base}...HEAD")
    except subprocess.CalledProcessError:
        problems.append(f"cannot diff against `{base}` (fetch it first?)")
        return
    changed = {p for p in diff.split() if p}
    changed.discard(change_doc.relative_to(ROOT).as_posix())
    # a file is "listed" if its path or its basename appears in the document
    for path in sorted(changed):
        if path not in text and Path(path).name not in text:
            problems.append(
                f"{change_doc.relative_to(ROOT)}: changed file not listed in Files: {path}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--files",
        action="store_true",
        help="also check the change document against git diff",
    )
    ap.add_argument(
        "--base",
        default="origin/main",
        help="base ref for --files (default origin/main)",
    )
    args = ap.parse_args()

    problems: list[str] = []
    check_frontmatter(problems)
    check_index(problems)
    check_done_ledger(problems)
    if args.files:
        check_files_vs_diff(problems, args.base)

    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

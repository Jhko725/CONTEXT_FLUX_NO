# Adapted from Jhko725/UniversalForceReconstruction scripts/docs/generate_references.py at
# b80a341.
"""Generate the bibliography page ``docs/references.md`` from ``docs/references.bib``.

The BibTeX file is the source: an entry is added there, and ``just references`` rewrites
the page. Each entry is listed under its in-text key, which is derived from the entry's
``author`` and ``year`` fields by the rules of ``docs/document-conventions.md``
§ References: ``Author Year`` for one author, ``Author & Author Year`` for two,
``Author et al. Year`` for three or more. When two entries give the same key, each gets
a letter suffix (``2001a``, ``2001b``) in the order of their titles, so that the
suffixes do not depend on the order of the entries in the file.

An author list that ends in BibTeX's ``and others`` is cited and listed as ``et al.``,
whatever the number of names before it, since the repository knows only the first
authors of such a work.

A surname is the BibTeX ``von`` and ``last`` parts of a name, with the braces removed;
no other LaTeX is decoded, so names and titles are written in Unicode in the ``.bib``
file.

The page keeps the frontmatter that every document carries. Its ``updated`` date is
the one already on the page when the rest of the page is unchanged, and today's date
otherwise, so that running the generator twice changes nothing.
``scripts/checks/check_references.py`` imports `render` by file path and fails when the
page differs from it.

Usage: ``uv run python scripts/docs/generate_references.py`` (no arguments), or
``just references``. Exits 1 when an entry cannot be parsed or lacks an author or a
year.
"""

from __future__ import annotations

import datetime
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import bibtexparser
from bibtexparser.middlewares import NameParts, SeparateCoAuthors, SplitNameParts
from bibtexparser.model import Entry


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "references.bib"
PAGE = ROOT / "docs" / "references.md"
_UPDATED = re.compile(r"^updated: (?P<date>\S+)$", re.MULTILINE)
_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"
# The BibTeX name that ends an author list known only by its first authors.
_OTHERS = "others"
_HEADER = """\
---
type: reference
status: current
updated: {updated}
---

# References

The bibliography of `context_flux_no`: every work that the documents,
docstrings and tests cite, one entry per key, sorted by key. This page is generated
from [`references.bib`](references.bib) by `scripts/docs/generate_references.py`
(`just references`) and is not edited by hand. The key grammar and the in-text form
are in [`document-conventions.md`](document-conventions.md) § References, and
`scripts/checks/check_references.py` fails on a key that has no entry here, or when
this page differs from the generator's output. An entry is one line that opens with the
key in bold, followed by the reference and its DOI or URL.
"""
_EMPTY = "The bibliography has no entry yet.\n"


@dataclass(frozen=True)
class Reference:
    """One entry of the bibliography, under the in-text key that cites it.

    **Attributes:**

    - `key`: the in-text key that the documents cite.
    - `line`: the entry line of the page, without the leading ``- **key** — ``.
    """

    key: str
    line: str


def _text(entry: Entry, field: str) -> str:
    """The value of `field` of `entry` with braces removed, or ``""`` when absent."""
    value = entry.fields_dict.get(field)
    return "" if value is None else str(value.value).replace("{", "").replace("}", "")


def _surname(name: NameParts) -> str:
    """The surname of a split BibTeX name: its ``von`` and ``last`` parts."""
    parts = [*name.von, *name.last]
    return " ".join(parts).replace("{", "").replace("}", "")


def _full_name(name: NameParts) -> str:
    """A split BibTeX name in reading order: first, von, last, then jr."""
    text = " ".join([*name.first, *name.von, *name.last])
    if name.jr:
        text += ", " + " ".join(name.jr)
    return text.replace("{", "").replace("}", "")


def _base_key(entry: Entry) -> str:
    """The in-text key of `entry` before any collision suffix.

    **Raises:**

    `ValueError` when the entry has no author or no year.
    """
    authors = entry.fields_dict.get("author")
    year = _text(entry, "year")
    if authors is None or not authors.value or not year:
        raise ValueError(f"entry {entry.key!r} needs both an author and a year")
    surnames = [_surname(name) for name in authors.value]
    if surnames[-1] == _OTHERS or len(surnames) > 2:
        return f"{surnames[0]} et al. {year}"
    if len(surnames) == 1:
        return f"{surnames[0]} {year}"
    return f"{surnames[0]} & {surnames[1]} {year}"


def _line(entry: Entry) -> str:
    """The reference of `entry` as one line: authors, year, title, venue (with the number
    of a report), DOI or URL."""
    names = [_full_name(name) for name in entry.fields_dict["author"].value]
    if names[-1] == _OTHERS:
        authors = ", ".join(names[:-1]) + " et al."
    elif len(names) == 1:
        authors = names[0]
    else:
        authors = ", ".join(names[:-1]) + " and " + names[-1]
    parts = [f"{authors} ({_text(entry, 'year')})."]
    if title := _text(entry, "title"):
        parts.append(f"{title}.")
    venue = _text(entry, "journal") or _text(entry, "booktitle")
    venue = venue or _text(entry, "publisher") or _text(entry, "institution")
    if venue:
        details = [f"*{venue}*"]
        # A report's number identifies it; an article's issue number is left out.
        number = _text(entry, "number") if entry.entry_type == "techreport" else ""
        details += [
            d for d in (number, _text(entry, "volume"), _text(entry, "pages")) if d
        ]
        parts.append(", ".join(details) + ".")
    if doi := _text(entry, "doi"):
        parts.append(f"<https://doi.org/{doi}>")
    elif url := _text(entry, "url"):
        parts.append(f"<{url}>")
    if note := _text(entry, "note"):
        parts.append(f"{note}.")
    return " ".join(parts)


def references(text: str) -> list[Reference]:
    """The entries of the BibTeX `text`, each under its in-text key, sorted by key.

    **Raises:**

    `ValueError` when a block of `text` does not parse, a BibTeX key repeats, or an
    entry has no author or no year.
    """
    library = bibtexparser.parse_string(
        text, append_middleware=[SeparateCoAuthors(), SplitNameParts()]
    )
    if library.failed_blocks:
        # bibtexparser counts lines from 0 and may not know a block's line.
        lines = ", ".join(
            "?" if block.start_line is None else str(block.start_line + 1)
            for block in library.failed_blocks
        )
        raise ValueError(f"BibTeX blocks that do not parse, starting at lines {lines}")
    groups: defaultdict[str, list[Entry]] = defaultdict(list)
    for entry in library.entries:
        groups[_base_key(entry)].append(entry)
    found: list[Reference] = []
    for base, entries in groups.items():
        if len(entries) == 1:
            found.append(Reference(base, _line(entries[0])))
            continue
        entries.sort(key=lambda entry: (_text(entry, "title").lower(), entry.key))
        for suffix, entry in zip(_SUFFIXES, entries, strict=False):
            found.append(Reference(base + suffix, _line(entry)))
    return sorted(found, key=lambda reference: reference.key)


def render(text: str, updated: str) -> str:
    """The page ``docs/references.md`` for the BibTeX `text`, dated `updated`."""
    found = references(text)
    body = "".join(f"- **{r.key}** — {r.line}\n" for r in found) if found else _EMPTY
    return _HEADER.format(updated=updated) + "\n" + body


def page_date(page: str) -> str | None:
    """The ``updated`` date in the frontmatter of `page`, or ``None`` when absent."""
    match = _UPDATED.search(page)
    return match.group("date") if match else None


def main() -> int:
    """Rewrite ``docs/references.md`` from ``docs/references.bib``; return 1 on an
    entry that cannot be listed."""
    source = SOURCE.read_text(encoding="utf-8")
    current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
    try:
        kept = page_date(current)
        if kept is not None and render(source, kept) == current:
            return 0
        PAGE.write_text(render(source, datetime.date.today().isoformat()), "utf-8")
    except ValueError as error:
        print(f"{SOURCE.relative_to(ROOT).as_posix()}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

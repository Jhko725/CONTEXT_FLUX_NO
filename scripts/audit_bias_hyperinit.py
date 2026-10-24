"""Audit which wandb runs actually used Bias-HyperInit (docs/findings.md F1; Phase 0.7).

Ten of twelve model configs spelled `hypernet_init: bias_hyperinit`, which the code did not
recognise, so those runs trained with the default initialisation. The resolved config of every
run is stored on wandb, so the string is queryable. Produces docs/bias_hyperinit_audit.csv and a
summary table; append the conclusion to docs/findings.md (F1) and the post-submission plan.

    uv run python scripts/audit_bias_hyperinit.py --entity jhko725 --project hyperfluxfno_revised
    # repeat for every project that produced a reported number; --out appends when the file exists
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import wandb


def tail(name: str | None) -> str:
    return (name or "?").rsplit(".", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True, action="append", help="repeatable")
    ap.add_argument("--out", type=Path, default=Path("docs/bias_hyperinit_audit.csv"))
    args = ap.parse_args()

    api = wandb.Api()
    rows = []
    for project in args.project:
        for run in api.runs(f"{args.entity}/{project}"):
            cfg = run.config or {}
            model = cfg.get("model", {}) or {}
            rows.append(
                {
                    "project": project,
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created": run.created_at,
                    "model": tail(model.get("_target_")),
                    "target_network": model.get("target_network_type", ""),
                    "encoder": model.get("encoder_type", ""),
                    "data": (cfg.get("data", {}) or {}).get("name", ""),
                    "seed": (model.get("key", {}) or {}).get("seed", ""),
                    "hypernet_init": model.get("hypernet_init", "<absent>"),
                    "effective_init": "bias-hyperinit"
                    if model.get("hypernet_init") == "bias-hyperinit"
                    else "default",
                    "best_valid_loss": run.summary.get("valid_loss"),
                    "url": run.url,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["project"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # summary: counts per (data, model, effective_init)
    from collections import Counter

    counts = Counter(
        (r["data"], r["model"], r["hypernet_init"], r["effective_init"]) for r in rows
    )
    print(f"{len(rows)} runs from {args.project}; written to {args.out}\n")
    print(f"{'data':<18}{'model':<26}{'config string':<18}{'effective':<16}runs")
    for (data, model, s, eff), n in sorted(counts.items()):
        print(f"{data:<18}{model:<26}{s:<18}{eff:<16}{n}")
    print(
        "\nNext: match the runs behind each paper table (notebooks' CHECKPOINT_DICT timestamps ↔ run.created_at)"
        " and append the conclusion to docs/findings.md F1."
    )


if __name__ == "__main__":
    main()

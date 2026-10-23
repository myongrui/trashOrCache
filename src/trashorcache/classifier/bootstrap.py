"""Cold start: emit decisions.jsonl from rules, with no model involved.

The model needs labels to fit, and after a reset there are none. This walks the
library with the one signal that needs no training -- a WhatsApp filename or OCR
text means a forwarded screenshot -- and writes the same contract the model does.
Everything else goes to review rather than dispose.
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from ..contract.paths import DATA_DIR, DECISIONS
from .store import load_aligned


def rule_action(row: dict[str, object]) -> tuple[str, str]:
    if row["source_app"] == "whatsapp":
        return "dispose", "rule: whatsapp"
    if (row["ocr_chars"] or 0) > 0:
        return "dispose", "rule: ocr text"
    return "review", "rule: no signal"


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-classifier bootstrap")
    ap.add_argument("--out", default=str(DECISIONS))
    args = ap.parse_args()

    df, _ = load_aligned()
    rows = df.select(["id", "source_app", "ocr_chars"]).to_dicts()
    decided = {(row["id"]): rule_action(row) for row in rows}

    groups = []
    for name in ("groups.json", "clusters.json"):
        path = DATA_DIR / name
        if path.exists():
            groups += json.loads(path.read_text())

    in_group: set[str] = set()
    members_of = {g["group_id"]: [m["asset_id"] for m in g["members"]] for g in groups}
    for gid, members in members_of.items():
        in_group.update(members)

    out = []
    for gid, members in members_of.items():
        keeper = members[0]
        verdict = "dispose_all" if all(decided[m][0] == "dispose" for m in members) else "keep_one"
        for m in members:
            action = "dispose" if verdict == "dispose_all" else ("keep" if m == keeper else decided[m][0])
            why = "group keeper" if m == keeper else (
                "group disposed" if verdict == "dispose_all" else decided[m][1])
            out.append({"asset_id": m, "action": action, "score": 0.9 if action == "dispose" else 0.5,
                        "group_id": gid, "group": verdict, "keeper": m == keeper, "why": why})

    for asset_id, (action, why) in decided.items():
        if asset_id in in_group:
            continue
        out.append({"asset_id": asset_id, "action": action,
                    "score": 0.9 if action == "dispose" else 0.5,
                    "group_id": None, "group": None, "keeper": None, "why": why})

    with open(args.out, "w") as fh:
        for d in out:
            fh.write(json.dumps(d) + "\n")

    counts: dict[str, int] = {}
    for d in out:
        counts[d["action"]] = counts.get(d["action"], 0) + 1
    group_counts: dict[str, int] = {}
    for d in out:
        if d["group"]:
            group_counts[d["group"]] = group_counts.get(d["group"], 0) + 1
    print(f"{len(out)} decisions -> {args.out}")
    print(f"  actions   {counts}")
    print(f"  in groups {len(in_group)} assets across {len(members_of)} groups {group_counts}")


if __name__ == "__main__":
    main()

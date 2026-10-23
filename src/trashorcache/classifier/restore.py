"""Restore decisions from a labels JSONL export back into the database.

The archive written before a reset is the only copy of those decisions, so this
is the way back after `toc-classifier reset` or a manual clear.
"""

from __future__ import annotations

import argparse
import json

from ..contract import db
from ..contract.paths import LABELS


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-classifier restore")
    ap.add_argument("--from", dest="src", default=str(LABELS), help="labels jsonl to import")
    ap.add_argument("--apply", action="store_true", help="write to the database")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    con = db.connect()
    known = {r[0] for r in con.execute("SELECT asset_id FROM assets")}
    usable = [r for r in rows if r["asset_id"] in known]
    missing = len(rows) - len(usable)

    print(f"{len(rows)} rows in {args.src}")
    print(f"  {len(usable)} match an asset in the database, {missing} do not")
    counts: dict[str, int] = {}
    for r in usable:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"  labels: {counts}")

    if not args.apply:
        print("\ndry run. re-run with --apply to restore.")
        return

    con.executemany(
        "UPDATE assets SET decision=?, decision_source=?, decided_at=? WHERE asset_id=?",
        [(r["label"], r.get("source", "labeler"), r.get("at", db.now()), r["asset_id"]) for r in usable])
    con.commit()
    print(f"\nrestored; db {db.counts(con)}")

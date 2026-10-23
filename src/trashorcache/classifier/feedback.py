"""Fold decided ledger rows into labels, then prune them from the ledger.

labels.jsonl is the permanent record; the ledger is only the work queue.
"""

from __future__ import annotations

import json

from ..contract import ledger
from ..contract.paths import LABELS
from .store import read_labels

OUTCOMES = {"dispose", "keep"}


def ingest() -> dict:
    con = ledger.connect()
    rows = ledger.decided(con)
    if not rows:
        return {"added": 0, "skipped": 0, "pending": ledger.counts(con)["pending"]}

    known = {r["asset_id"] for r in read_labels()}
    added = skipped = 0
    at = ledger.now()

    with LABELS.open("a") as fh:
        for r in rows:
            if r["outcome"] not in OUTCOMES:
                continue
            if r["asset_id"] in known:
                skipped += 1
                continue
            fh.write(json.dumps({
                "asset_id": r["asset_id"],
                "label": r["outcome"],
                "tag": r["tag"],
                "source": "observe",
                "batch_id": r["batch_id"],
                "at": at,
            }) + "\n")
            known.add(r["asset_id"])
            added += 1
        fh.flush()

    pruned = ledger.prune_decided(con)
    return {"added": added, "skipped": skipped, "pruned": pruned,
            "pending": ledger.counts(con)["pending"]}

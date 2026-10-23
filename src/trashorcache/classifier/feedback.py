"""Fold observed Immich behaviour into labels. Deletions and survivals only.

An asset left untouched in a `Review` tag is inattention, not a decision, so it
produces no label.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..contract.paths import FEEDBACK, LABELS
from .store import read_labels

OUTCOME_LABEL = {"dispose": "dispose", "keep": "keep"}


def ingest() -> dict:
    if not FEEDBACK.exists():
        return {"added": 0, "skipped": 0, "inattention": 0}

    known = {r["asset_id"] for r in read_labels()}
    added = skipped = inattention = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with LABELS.open("a") as fh:
        for line in FEEDBACK.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            label = OUTCOME_LABEL.get(r["outcome"])
            if label is None:
                inattention += 1
                continue
            if r["asset_id"] in known:
                skipped += 1
                continue
            fh.write(json.dumps({
                "asset_id": r["asset_id"],
                "label": label,
                "source": "observe",
                "batch_id": r["batch_id"],
                "at": now,
            }) + "\n")
            known.add(r["asset_id"])
            added += 1
        fh.flush()

    return {"added": added, "skipped": skipped, "inattention": inattention}

"""Write the database back out as plain files: a readable backup, not a source.

The database is authoritative. These files exist so the irreplaceable part --
decisions -- can be inspected, diffed and copied without a running process.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..contract import db
from ..contract.paths import DATA_DIR, EMBEDDINGS, LABELS


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-classifier export")
    ap.add_argument("--vectors", default=None, help="also write vectors to this .npy path")
    args = ap.parse_args()

    con = db.connect()

    rows = con.execute(
        "SELECT asset_id, decision, decision_source, decided_at FROM assets "
        "WHERE decision IS NOT NULL ORDER BY decided_at, asset_id").fetchall()
    seen, lines = set(), []
    for asset_id, label, source, at in rows:
        if asset_id in seen:
            continue
        seen.add(asset_id)
        lines.append(json.dumps({"asset_id": asset_id, "label": label,
                                 "source": source, "at": at}))
    LABELS.write_text("\n".join(lines) + "\n")
    print(f"labels  {len(lines)} -> {LABELS}")

    if args.vectors:
        ids = [r[0] for r in con.execute("SELECT asset_id FROM vectors ORDER BY asset_id")]
        X = np.stack([np.frombuffer(r[0], dtype=np.float32)
                      for r in con.execute("SELECT vec FROM vectors ORDER BY asset_id")])
        np.save(args.vectors, X)
        (DATA_DIR / "vectors.index.json").write_text(json.dumps(ids) + "\n")
        print(f"vectors {X.shape} -> {args.vectors} (row order in vectors.index.json)")

    print(f"db {db.counts(con)}")

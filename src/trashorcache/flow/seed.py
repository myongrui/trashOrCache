"""One-off: materialise already-made decisions as Immich tags.

The labeler produced thousands of decisions before tagging existed, so those
assets are settled in the database but have no tag and never enter a proposal.
This backfills them so the tag grids reflect everything already judged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ..contract import db
from .immich import ACTION_TAGS, add_assets, ensure_tag

CHUNK = 500


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-flow seed")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--apply", action="store_true", help="actually write tags to Immich")
    args = ap.parse_args()

    con = db.connect()
    rows = [dict(r) for r in con.execute(
        "SELECT asset_id, decision FROM assets "
        "WHERE decision IS NOT NULL AND tag IS NULL AND purged_at IS NULL "
        "ORDER BY decision, asset_id")]

    by_tag: dict[str, list[str]] = {}
    for r in rows:
        by_tag.setdefault(ACTION_TAGS[r["decision"]], []).append(r["asset_id"])

    print(f"{len(rows)} already-decided assets without a tag")
    for tag, ids in sorted(by_tag.items()):
        print(f"  {tag:8s} {len(ids):5d}")
    calls = sum((len(v) + args.chunk - 1) // args.chunk for v in by_tag.values())
    print(f"  {calls} tag-add calls at {args.chunk}/call")

    if not args.apply:
        print("\ndry run. re-run with --apply to write tags.")
        return

    batch = f"seed-{datetime.now(timezone.utc).date().isoformat()}"
    written = 0
    for tag, ids in sorted(by_tag.items()):
        tag_obj = ensure_tag(tag)
        print(f"\n{tag}: tag id {tag_obj['id'][:8]}")
        for start in range(0, len(ids), args.chunk):
            chunk = ids[start : start + args.chunk]
            add_assets(tag_obj["id"], chunk)
            con.executemany(
                "UPDATE assets SET tag=?, tag_id=?, batch_id=COALESCE(batch_id, ?) WHERE asset_id=?",
                [(tag, tag_obj["id"], batch, a) for a in chunk])
            con.commit()
            written += len(chunk)
            print(f"  {written}/{len(rows)} tagged", flush=True)

    remaining = con.execute(
        "SELECT count(*) FROM assets WHERE decision IS NOT NULL AND tag IS NULL AND purged_at IS NULL"
    ).fetchone()[0]
    print(f"\ntagged {written}; still untagged: {remaining}")

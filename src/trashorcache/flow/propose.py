"""Apply decisions.jsonl to Immich tags, recording only what actually succeeded."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ..contract import db
from ..contract.paths import DECISIONS
from .immich import ACTION_TAGS, add_assets, ensure_tag

CHUNK = 500
PRIORITY = {"dispose": 0, "review": 1, "keep": 2}


def taken_ids(con) -> set[str]:
    """Assets already decided, or already waiting on a decision."""
    return {r[0] for r in con.execute(
        "SELECT asset_id FROM assets WHERE decision IS NOT NULL OR proposed_at IS NOT NULL")}


def load_decisions() -> list[dict]:
    if not DECISIONS.exists():
        raise SystemExit(f"{DECISIONS} not found - run `toc-classifier run` first")
    rows = [json.loads(l) for l in DECISIONS.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda d: (PRIORITY.get(d["action"], 9), -d.get("score", 0.0)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-flow propose")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of assets proposed")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--apply", action="store_true", help="actually write tags to Immich")
    args = ap.parse_args()

    con = db.connect()
    skip = taken_ids(con)
    rows = [d for d in load_decisions() if d["asset_id"] not in skip]
    if args.limit:
        rows = rows[: args.limit]

    by_tag: dict[str, list[dict]] = {}
    for d in rows:
        by_tag.setdefault(ACTION_TAGS[d["action"]], []).append(d)

    c = db.counts(con)
    print(f"{len(rows)} assets to propose "
          f"({len(skip)} skipped: {c['pending']} already pending, {c['decided']} decided)")
    for tag, items in sorted(by_tag.items()):
        print(f"  {tag:8s} {len(items):5d}")
    calls = sum((len(v) + args.chunk - 1) // args.chunk for v in by_tag.values())
    print(f"  {calls} tag-add calls at {args.chunk}/call")

    if not args.apply:
        print("\ndry run. re-run with --apply to write tags.")
        for tag, items in sorted(by_tag.items()):
            print(f"  {tag}: {[d['asset_id'][:8] for d in items[:3]]}")
        return

    batch = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written = 0
    for tag, items in sorted(by_tag.items()):
        tag_obj = ensure_tag(tag)
        print(f"\n{tag}: tag id {tag_obj['id'][:8]}")
        for start in range(0, len(items), args.chunk):
            chunk = items[start : start + args.chunk]
            ids = [d["asset_id"] for d in chunk]
            add_assets(tag_obj["id"], ids)
            con.executemany(
                "UPDATE assets SET batch_id=?, action=?, tag=?, tag_id=?, group_id=?, proposed_at=? "
                "WHERE asset_id=?",
                [(batch, d["action"], tag, tag_obj["id"], d.get("group_id"), db.now(), d["asset_id"])
                 for d in chunk])
            con.commit()
            written += len(chunk)
            print(f"  {written}/{len(rows)} tagged", flush=True)

    print(f"\ntagged {written} assets; db {db.counts(con)}")

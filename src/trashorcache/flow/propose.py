"""Apply decisions.jsonl to Immich tags, recording only what actually succeeded."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ..contract.paths import DECISIONS, LABELS, PROPOSALS
from .immich import ACTION_TAGS, add_assets, ensure_tag

CHUNK = 500
PRIORITY = {"dispose": 0, "review": 1, "keep": 2}


def settled_ids() -> set[str]:
    if not LABELS.exists():
        return set()
    return {json.loads(l)["asset_id"] for l in LABELS.read_text().splitlines() if l.strip()}


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

    skip = settled_ids()
    rows = [d for d in load_decisions() if d["asset_id"] not in skip]
    if args.limit:
        rows = rows[: args.limit]

    by_tag: dict[str, list[dict]] = {}
    for d in rows:
        by_tag.setdefault(ACTION_TAGS[d["action"]], []).append(d)

    print(f"{len(rows)} assets to propose ({len(skip)} labelled assets excluded)")
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
    with PROPOSALS.open("a") as log:
        for tag, items in sorted(by_tag.items()):
            tag_obj = ensure_tag(tag)
            print(f"\n{tag}: tag id {tag_obj['id'][:8]}")
            for start in range(0, len(items), args.chunk):
                chunk = items[start : start + args.chunk]
                ids = [d["asset_id"] for d in chunk]
                add_assets(tag_obj["id"], ids)
                for d in chunk:
                    log.write(json.dumps({
                        "asset_id": d["asset_id"],
                        "action": d["action"],
                        "tag": tag,
                        "tag_id": tag_obj["id"],
                        "group_id": d.get("group_id"),
                        "batch_id": batch,
                        "proposed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "observed_at": None,
                        "outcome": None,
                    }) + "\n")
                log.flush()
                written += len(chunk)
                print(f"  {written}/{len(rows)} tagged", flush=True)

    print(f"\ntagged {written} assets; proposals appended to {PROPOSALS.name}")

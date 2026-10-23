"""Read Immich and decide what the user did with each proposal."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..contract.paths import FEEDBACK, PROPOSALS
from .immich import get_asset

WORKERS = 16


def load_proposals() -> list[dict]:
    if not PROPOSALS.exists():
        raise SystemExit(f"{PROPOSALS} not found - run `toc-flow propose` first")
    return [json.loads(l) for l in PROPOSALS.read_text().splitlines() if l.strip()]


def observed_keys() -> set[tuple[str, str]]:
    if not FEEDBACK.exists():
        return set()
    return {(r["batch_id"], r["asset_id"])
            for r in (json.loads(l) for l in FEEDBACK.read_text().splitlines() if l.strip())}


def classify(asset: dict | None) -> str:
    if asset is None:
        return "dispose"
    if asset.get("isTrashed"):
        return "dispose"
    tags = {t.get("value") for t in asset.get("tags", [])}
    if "Review" in tags:
        return "inattention"
    return "keep"


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-flow observe")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="append findings to feedback.jsonl")
    args = ap.parse_args()

    done = observed_keys()
    todo = [p for p in load_proposals() if (p["batch_id"], p["asset_id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to observe")
        return

    print(f"observing {len(todo)} proposals from {len({p['batch_id'] for p in todo})} batch(es)")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        assets = list(pool.map(lambda p: get_asset(p["asset_id"]), todo))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for p, a in zip(todo, assets):
        outcome = classify(a)
        rows.append({
            "asset_id": p["asset_id"],
            "action": p["action"],
            "tag": p["tag"],
            "batch_id": p["batch_id"],
            "outcome": outcome,
            "observed_at": now,
        })

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print("  " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for outcome in ("dispose", "keep", "inattention"):
        sample = [r["asset_id"][:8] for r in rows if r["outcome"] == outcome][:3]
        if sample:
            print(f"  {outcome:12s} e.g. {sample}")

    if not args.apply:
        print("\ndry run. re-run with --apply to write feedback.jsonl.")
        return

    with FEEDBACK.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
    print(f"\nwrote {len(rows)} rows to {FEEDBACK.name}")

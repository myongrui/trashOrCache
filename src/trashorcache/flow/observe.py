"""Read Immich and decide what happened to each pending proposal.

Survival is keep, full stop. The user disposes by trashing; anything still
standing when observe runs is something they chose to keep. Survivors are moved
from their proposal tag to `Keep`, so the tag itself stays a readable record.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from ..contract import db
from .immich import add_assets, ensure_tag, get_asset, remove_assets

WORKERS = 16


def outcome_for(asset: dict | None) -> str:
    if asset is None or asset.get("isTrashed"):
        return "dispose"
    return "keep"


def _verify(rows: list[dict], keep_tag_id: str) -> list[str]:
    """Asset ids that do not actually carry the Keep tag yet."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        assets = list(pool.map(lambda r: get_asset(r["asset_id"]), rows))
    return [r["asset_id"] for r, a in zip(rows, assets)
            if keep_tag_id not in {t.get("id") for t in (a or {}).get("tags", [])}]


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-flow observe")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="move tags and record the decision")
    args = ap.parse_args()

    con = db.connect()
    pending = [dict(r) for r in con.execute(
        "SELECT asset_id, tag, tag_id, batch_id FROM assets "
        "WHERE decision IS NULL AND proposed_at IS NOT NULL ORDER BY proposed_at, asset_id")]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"nothing to observe ({db.counts(con)})")
        return

    print(f"observing {len(pending)} pending proposals from "
          f"{len({r['batch_id'] for r in pending})} batch(es)")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        assets = list(pool.map(lambda r: get_asset(r["asset_id"]), pending))

    outcomes = [outcome_for(a) for a in assets]
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    print("  " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for o in ("dispose", "keep"):
        sample = [r["asset_id"][:8] for r, x in zip(pending, outcomes) if x == o][:3]
        if sample:
            print(f"  {o:8s} e.g. {sample}")

    if not args.apply:
        print("\ndry run. re-run with --apply to move tags and record the outcome.")
        return

    keeps = [r for r, o in zip(pending, outcomes) if o == "keep"]
    if keeps:
        keep_tag = ensure_tag("Keep")
        by_tag: dict[tuple[str, str], list[str]] = {}
        for r in keeps:
            by_tag.setdefault((r.get("tag_id"), r["tag"]), []).append(r["asset_id"])
        for (tag_id, tag), ids in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
            if tag != "Keep" and tag_id:
                remove_assets(tag_id, ids)
            print(f"  {len(ids):5d} out of {tag}")
        add_assets(keep_tag["id"], [r["asset_id"] for r in keeps])
        print(f"  {len(keeps):5d} into Keep")

        # Immich's bulk tag calls fail silently for individual assets under
        # remove-then-add, so trust the read-back rather than the 2xx.
        unconfirmed = _verify(keeps, keep_tag["id"])
        for attempt in range(2):
            if not unconfirmed:
                break
            print(f"  retrying {len(unconfirmed)} unconfirmed ({attempt + 1})")
            add_assets(keep_tag["id"], unconfirmed)
            retry = [r for r in keeps if r["asset_id"] in set(unconfirmed)]
            unconfirmed = _verify(retry, keep_tag["id"])
        if unconfirmed:
            print(f"  WARNING: {len(unconfirmed)} still not tagged Keep; left pending")
            keeps = [r for r in keeps if r["asset_id"] not in set(unconfirmed)]

    decided = {r["asset_id"]: o for r, o in zip(pending, outcomes) if o == "dispose"}
    decided.update({r["asset_id"]: "keep" for r in keeps})
    at = db.now()
    con.executemany(
        "UPDATE assets SET decision=?, decision_source='observe', decided_at=? "
        "WHERE asset_id=? AND decision IS NULL",
        [(o, at, a) for a, o in decided.items()])
    con.commit()
    print(f"\ndb: {db.counts(con)}")

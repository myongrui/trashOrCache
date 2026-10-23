"""Apply decisions.jsonl to Immich: tags, one stack per group, and the album.

Write order is intent-first: mark the rows `applying` and commit, call the API,
then mark them `applied`. A crash therefore leaves visible rows to retry rather
than a divergence between Immich and the database.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..contract import db
from ..contract.paths import DECISIONS
from .immich import (
    ACTION_TAGS, add_album_assets, add_assets, create_stack, ensure_album, ensure_tag, get_asset,
    remove_assets,
)

CHUNK = 500
PRIORITY = {"dispose": 0, "review": 1, "keep": 2}
ALBUM = "Groups"


def load_decisions() -> list[dict]:
    if not DECISIONS.exists():
        raise SystemExit(f"{DECISIONS} not found - run `toc-classifier run` first")
    rows = [json.loads(l) for l in DECISIONS.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda d: (PRIORITY.get(d["action"], 9), -d.get("score", 0.0)))
    return rows


def taken_ids(con) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT asset_id FROM assets WHERE (decision IS NOT NULL AND recheck = 0) "
        "OR proposed_at IS NOT NULL")}


def live_map(asset_ids: list[str]) -> dict[str, bool]:
    def is_live(a: str) -> bool:
        x = get_asset(a)
        return x is not None and not x.get("isTrashed")

    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(zip(asset_ids, pool.map(is_live, asset_ids)))


def group_members(decisions: list[dict]) -> dict[str, list[dict]]:
    """All members of each group, keeper first, including already-decided ones."""
    out: dict[str, list[dict]] = {}
    for d in decisions:
        if d.get("group_id"):
            out.setdefault(d["group_id"], []).append(d)
    for members in out.values():
        members.sort(key=lambda d: (not d.get("keeper"), d["asset_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-flow propose")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of assets tagged")
    ap.add_argument("--max-groups", type=int, default=None, help="cap the number of new stacks")
    ap.add_argument("--no-groups", action="store_true", help="tags only, no stacks or album")
    ap.add_argument("--group", action="append", default=None, help="restrict to these group ids")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--refresh", action="store_true",
                    help="un-tag pending assets first, so a re-proposal can move them between tags")
    ap.add_argument("--apply", action="store_true", help="actually write to Immich")
    args = ap.parse_args()

    con = db.connect()
    decisions = load_decisions()
    skip = taken_ids(con)
    if args.group:
        decisions = [d for d in decisions if d.get("group_id") in set(args.group)]
        skip = {s for s in skip if not any(d["asset_id"] == s for d in decisions)}
    rows = [d for d in decisions if d["asset_id"] not in skip]
    if args.limit:
        rows = rows[: args.limit]

    by_tag: dict[str, list[dict]] = {}
    for d in rows:
        by_tag.setdefault(ACTION_TAGS[d["action"]], []).append(d)

    members = group_members(decisions)
    batch_groups = sorted({d["group_id"] for d in decisions if d.get("group_id")})
    have_stack = {r[0] for r in con.execute("SELECT group_id FROM groups")}
    new_groups = [g for g in batch_groups if g not in have_stack]
    if args.max_groups:
        new_groups = new_groups[: args.max_groups]
    album_assets = []

    c = db.counts(con)
    print(f"{len(rows)} assets to tag  ({len(skip)} skipped: {c['pending']} pending, {c['decided']} decided)")
    for tag, items in sorted(by_tag.items()):
        print(f"  {tag:8s} {len(items):5d}")
    if not args.no_groups:
        print(f"{len(new_groups)} new stacks -> album {ALBUM!r} (all keepers, refreshed at the end)")
        for g in new_groups:
            print(f"  {g}  {len(members[g])} members, keeper "
                  f"{next((d['asset_id'][:8] for d in members[g] if d.get('keeper')), '?')}")

    if not args.apply:
        print("\ndry run. re-run with --apply to write to Immich.")
        return

    if args.refresh:
        stale = [dict(r) for r in con.execute(
            "SELECT asset_id, tag, tag_id FROM assets "
            "WHERE decision IS NULL AND proposed_at IS NOT NULL")]
        by_old: dict[tuple[str, str], list[str]] = {}
        for r in stale:
            if r["tag_id"]:
                by_old.setdefault((r["tag_id"], r["tag"]), []).append(r["asset_id"])
        for (tag_id, tag), ids in by_old.items():
            remove_assets(tag_id, ids)
            print(f"  untagged {len(ids):5d} from {tag}")
        con.executemany(
            "UPDATE assets SET batch_id=NULL, action=NULL, tag=NULL, tag_id=NULL, "
            "proposed_at=NULL, apply_state=NULL WHERE asset_id=?", [(a,) for a, _, _ in
            [(r["asset_id"], r["tag"], r["tag_id"]) for r in stale]])
        con.commit()
        print(f"  cleared proposal state on {len(stale)} assets")

    batch = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tagged = 0
    for tag, items in sorted(by_tag.items()):
        tag_obj = ensure_tag(tag)
        print(f"\n{tag}: tag id {tag_obj['id'][:8]}")
        for start in range(0, len(items), args.chunk):
            chunk = items[start : start + args.chunk]
            ids = [d["asset_id"] for d in chunk]
            con.executemany(
                "UPDATE assets SET apply_state='applying', batch_id=?, action=?, tag=? WHERE asset_id=?",
                [(batch, d["action"], tag, d["asset_id"]) for d in chunk])
            con.commit()
            add_assets(tag_obj["id"], ids)
            con.executemany(
                "UPDATE assets SET apply_state='applied', tag_id=?, proposed_at=? WHERE asset_id=?",
                [(tag_obj["id"], db.now(), a) for a in ids])
            con.commit()
            tagged += len(ids)
            print(f"  {tagged}/{len(rows)}", flush=True)

    if args.no_groups or not new_groups:
        print(f"\ntagged {tagged}; db {db.counts(con)}")
        return

    album = ensure_album(ALBUM)
    db.set_meta(con, "groups_album_id", album["id"])
    print(f"\nalbum {ALBUM!r} id {album['id'][:8]}")

    # a stack needs two live assets; groups touched by deletions are skipped
    wanted = [a for g in new_groups for a in (d["asset_id"] for d in members[g])]
    alive = live_map(wanted)

    skipped = 0
    for gid in new_groups:
        ordered = [d["asset_id"] for d in members[gid] if alive.get(d["asset_id"])]
        if len(ordered) < 2:
            skipped += 1
            continue
        keeper = next((d["asset_id"] for d in members[gid]
                       if d.get("keeper") and alive.get(d["asset_id"])), ordered[0])
        if ordered[0] != keeper:
            ordered.remove(keeper)
            ordered.insert(0, keeper)
        stack = create_stack(ordered)
        con.execute("INSERT OR REPLACE INTO groups (group_id, stack_id, keeper_asset_id, created_at) "
                    "VALUES (?,?,?,?)", (gid, stack["id"], keeper, db.now()))
        con.commit()
        print(f"  {gid}: stack {stack['id'][:8]}, {len(ordered)} members, primary {keeper[:8]}")

    all_keepers = [r[0] for r in con.execute(
        "SELECT keeper_asset_id FROM groups WHERE keeper_asset_id IS NOT NULL")]
    live_keepers = [a for a, ok in live_map(all_keepers).items() if ok]
    add_album_assets(album["id"], live_keepers)
    print(f"  album: {len(live_keepers)} keepers ({skipped} groups skipped, too few live members)")
    print(f"\ntagged {tagged}; stacks {len(new_groups)}; db {db.counts(con)}")

"""Return tagged-but-unsorted assets to the pool: remove their tag, clear the proposal.

Used when a tag is being treated as a to-do list and only part of it has actually
been worked through. Removing the tag stops `observe` from reading those assets
as deliberate keeps.
"""

from __future__ import annotations

import argparse

from ..contract import db
from .immich import ensure_tag, remove_assets

CHUNK = 500


def main() -> None:
    ap = argparse.ArgumentParser(prog="toc-flow reset")
    ap.add_argument("--tag", default="Review")
    ap.add_argument("--before", required=True, help="only assets captured before this date")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    con = db.connect()
    rows = [dict(r) for r in con.execute(
        "SELECT asset_id FROM assets WHERE tag = ? AND purged_at IS NULL "
        "AND substr(file_created_at, 1, 10) < ? ORDER BY asset_id",
        (args.tag, args.before))]
    ids = [r["asset_id"] for r in rows]

    print(f"{len(ids)} assets tagged {args.tag} captured before {args.before}")
    if not args.apply:
        print("\ndry run. re-run with --apply to untag and clear their proposal state.")
        return

    tag = ensure_tag(args.tag)
    for start in range(0, len(ids), args.chunk):
        remove_assets(tag["id"], ids[start : start + args.chunk])
        print(f"  untagged {min(start + args.chunk, len(ids))}/{len(ids)}", flush=True)

    con.executemany(
        "UPDATE assets SET tag=NULL, tag_id=NULL, batch_id=NULL, action=NULL, "
        "proposed_at=NULL, apply_state=NULL WHERE asset_id=?", [(a,) for a in ids])
    con.commit()
    print(f"\ncleared proposal state on {len(ids)} assets; db {db.counts(con)}")

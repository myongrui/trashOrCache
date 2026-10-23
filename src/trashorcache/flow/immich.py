from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import polars as pl

from ..contract.paths import SCORES
from .config import KEY, URL


def call(method: str, path: str, body: dict | None = None) -> dict:
    if not KEY:
        raise SystemExit("IMMICH_API_KEY is not set. Create one in Immich: Settings > API Keys\n"
                         "  (scopes: tag.create, tag.read, tag.assign)")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{URL.rstrip('/')}{path}", data=data, method=method,
        headers={"x-api-key": KEY, "content-type": "application/json", "accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:300]}")


def ensure_tag(name: str) -> dict:
    for t in call("GET", "/api/tags"):
        if t["name"] == name:
            return t
    return call("POST", "/api/tags", {"name": name})


def add_assets(tag_id: str, ids: list[str], chunk: int = 500) -> int:
    for i in range(0, len(ids), chunk):
        call("PUT", f"/api/tags/{tag_id}/assets", {"ids": ids[i : i + chunk]})
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["auto_dispose", "redundant", "upgrade", "review"], default="auto_dispose")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="actually write the tag to Immich")
    ap.add_argument("--skip-labeled", action="store_true", default=True)
    args = ap.parse_args()

    args.tag = args.tag or f"toc:{args.band}"
    sc = pl.read_parquet(SCORES)
    if "action" not in sc.columns:
        raise SystemExit("scores.parquet has no action column - run confidence.py first")
    sel = sc.filter((pl.col("action") == args.band) & ~pl.col("labeled"))
    ids = sel["id"].to_list()
    if args.limit:
        ids = ids[: args.limit]

    print(f"band '{args.band}': {len(ids)} unlabelled assets")
    print(f"tag: {args.tag}")
    print(f"immich: {URL}  api key: {'set' if KEY else 'MISSING'}")

    if not args.apply:
        print("\ndry run. first 5 ids:", ids[:5])
        print("re-run with --apply to create the tag and assign the assets.")
        return

    tag = ensure_tag(args.tag)
    print(f"tag id {tag['id']}")
    n = add_assets(tag["id"], ids)
    print(f"tagged {n} assets")


if __name__ == "__main__":
    main()

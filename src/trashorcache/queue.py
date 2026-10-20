from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import DATA_DIR
from .store import LABELS, SCORES, clusters, load_aligned

QUEUE = DATA_DIR / "queue.json"

FIELDS = ("id", "originalFileName", "source_app", "file_created_at", "width", "height", "file_size", "ocr_chars", "face_count")


def cluster_path(C: np.ndarray, start: int) -> list[int]:
    order = [start]
    remaining = set(range(len(C))) - {start}
    while remaining:
        cur = C[order[-1]]
        nxt = max(remaining, key=lambda j: float(cur @ C[j]))
        order.append(nxt)
        remaining.discard(nxt)
    return order


def nn_tour(V: np.ndarray) -> list[int]:
    n = len(V)
    start = int(np.argmax(V @ V.mean(0)))
    order = [start]
    free = np.ones(n, dtype=bool)
    free[start] = False
    cur = start
    for _ in range(n - 1):
        sims = V @ V[cur]
        sims[~free] = -2.0
        cur = int(np.argmax(sims))
        free[cur] = False
        order.append(cur)
    return order


def labelled_ids() -> set[str]:
    if not LABELS.exists():
        return set()
    return {json.loads(line)["asset_id"] for line in LABELS.read_text().splitlines() if line.strip()}


def coverage(df: pl.DataFrame, X: np.ndarray, n: int, n_clusters: int, seed: int, done: set[str]) -> list[int]:
    assign = clusters(X, n_clusters, seed)
    cents = np.stack([X[assign == j].mean(0) for j in range(n_clusters)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)

    ids = df["id"].to_list()
    unused = np.array([i not in done for i in ids])
    per_cluster = max(1, int(np.ceil(n / n_clusters)))
    selected: list[int] = []
    seen: set[int] = set()

    for j in cluster_path(cents, int(np.argmax(cents.sum(1)))):
        members = np.flatnonzero((assign == j) & unused)
        if members.size == 0:
            continue
        picked = [int(members[np.argmax(X[members] @ cents[j])])]
        while len(picked) < per_cluster and len(picked) < members.size:
            pool = members[~np.isin(members, picked)]
            sims = (X[pool] @ X[picked].T).max(axis=1)
            fresh = pool[sims < 0.995]
            if fresh.size == 0:
                break
            picked.append(int(fresh[np.argmax(sims[sims < 0.995])]))
        for i in picked:
            if i not in seen:
                seen.add(i)
                selected.append(i)
        if len(selected) >= n:
            break

    selected = selected[:n]
    return [selected[k] for k in nn_tour(X[selected])]


def model_queue(df, X, n, done, hard, random_only=False) -> list[tuple[int, float, str]]:
    scores = pl.read_parquet(SCORES)
    if scores.height != df.height or scores["id"].to_list() != df["id"].to_list():
        raise SystemExit("scores.parquet is stale relative to the manifest - re-run probe")

    ids = df["id"].to_list()
    p = scores["score"].to_numpy()
    keep_mask = np.array([a not in done and np.isfinite(p[i]) for i, a in enumerate(ids)])
    if hard:
        keep_mask &= df["ocr_chars"].to_numpy() == 0
        keep_mask &= df["source_app"].fill_null("").to_numpy() != "whatsapp"
    pool = np.flatnonzero(keep_mask)
    if pool.size == 0:
        return []

    ranks = np.empty(pool.size)
    ranks[np.argsort(np.argsort(p[pool]))] = np.arange(pool.size)
    pct = ranks / max(pool.size - 1, 1)

    rank_of = {int(i): float(pct[k]) for k, i in enumerate(pool)}
    if random_only:
        pick = np.random.default_rng().choice(pool, size=min(n, pool.size), replace=False)
        return [(int(i), rank_of[int(i)], "random") for i in pick]

    n_unc = int(n * 0.5)
    n_rand = n - n_unc
    order = np.argsort(np.abs(p[pool] - np.median(p[pool])))
    unc = pool[order[:n_unc]]
    rest = np.setdiff1d(pool, unc)
    rand = np.random.default_rng().choice(rest, size=min(n_rand, rest.size), replace=False)

    strata = {int(i): "uncertain" for i in unc}
    strata.update({int(i): "random" for i in rand})
    chosen = list(dict.fromkeys([*unc, *rand]))
    ordered = [chosen[k] for k in nn_tour(X[chosen])]
    return [(i, rank_of[i], strata[i]) for i in ordered]


def to_items(df, idx, pct=None, assign=None, stratum=None) -> list[dict]:
    cols = {f: df[f].to_list() for f in FIELDS}
    out = []
    for i in idx:
        item = {
            "asset_id": cols["id"][i],
            "file_name": cols["originalFileName"][i],
            "source_app": cols["source_app"][i],
            "taken": cols["file_created_at"][i].strftime("%Y-%m-%d") if cols["file_created_at"][i] else None,
            "width": cols["width"][i],
            "height": cols["height"][i],
            "bytes": cols["file_size"][i],
            "ocr_chars": cols["ocr_chars"][i],
            "face_count": cols["face_count"][i],
            "cluster": int(assign[i]) if assign is not None else None,
        }
        if pct is not None:
            item["pct"] = pct[i]
            item["suggest"] = "dispose" if pct[i] > 0.5 else "keep"
            item["stratum"] = stratum[i]
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["coverage", "model", "eval", "agree", "dupes", "validate"], default="coverage")
    ap.add_argument("--band", choices=["auto_dispose", "review"], default="auto_dispose")
    ap.add_argument("--source", choices=["dupes", "variants"], default="dupes")
    ap.add_argument("--out", default=str(QUEUE))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--clusters", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", choices=["all", "hard"], default="all")
    args = ap.parse_args()

    df, X = load_aligned()
    done = labelled_ids()
    assign = clusters(X, args.clusters, args.seed)

    if args.mode == "validate":
        sc = pl.read_parquet(SCORES)
        ids = df["id"].to_list()
        want = np.array([a == args.band for a in sc["action"].to_list()])
        labelled = sc["labeled"].to_numpy()
        pool = np.flatnonzero(want & ~labelled & np.array([a not in done for a in ids]))
        pick = np.random.default_rng().choice(pool, size=min(args.n, pool.size), replace=False)
        items = to_items(df, [int(i) for i in pick], None, assign)
        for it in items:
            it["suggest"] = "dispose" if args.band == "auto_dispose" else "keep"
            it["pct"] = 0.0
            it["stratum"] = args.band
        Path(args.out).write_text(json.dumps(items, indent=1) + "\n")
        print(f"validate queue: {len(items)} random images from the '{args.band}' band -> {args.out}")
        print(f"  label them normally; /api/agreement hit rate = measured precision of the band")
        return

    if args.mode == "dupes":
        src = DATA_DIR / ("groups.json" if args.source == "dupes" else "clusters.json")
        groups = json.loads(src.read_text())
        todo = []
        for g in groups:
            fresh = [m for m in g["members"] if m["asset_id"] not in done]
            if len(fresh) < 2:
                continue
            settled_keep = [m for m in g["members"] if m["asset_id"] in done]
            g = dict(g)
            g["members"] = fresh + settled_keep
            todo.append(g)
        items = todo[: args.n]
        Path(args.out).write_text(json.dumps(items, indent=1) + "\n")
        n_img = sum(len(g["members"]) for g in items)
        print(f"dupes queue: {len(items)} groups, {n_img} images -> {args.out}")
        print(f"  {sum(1 for g in items if g['kind'] == 'duplicate')} duplicate groups, "
              f"{sum(1 for g in items if g['kind'] == 'similar')} similar groups")
        print(f"  up to {n_img - len(items)} avoidable images")
        return

    if args.mode == "agree":
        seen = sorted(done)
        if not seen:
            raise SystemExit("no labels to re-serve")
        pick = np.random.default_rng().choice(seen, size=min(args.n, len(seen)), replace=False)
        idx = [pos for pos in (df["id"].to_list().index(a) for a in pick)]
        items = to_items(df, idx, None, assign)
    elif args.mode == "coverage":
        idx = coverage(df, X, args.n, args.clusters, args.seed, done)
        items = to_items(df, idx, None, assign)
    else:
        pairs = model_queue(df, X, args.n, done, args.subset == "hard", random_only=args.mode == "eval")
        idx = [i for i, _, _ in pairs]
        items = to_items(df, idx, {i: q for i, q, _ in pairs}, assign, {i: s for i, _, s in pairs})

    Path(args.out).write_text(json.dumps(items, indent=1) + "\n")
    print(f"{args.mode} queue: {len(items)} images -> {args.out}")
    if items and "suggest" in items[0]:
        n_d = sum(1 for x in items if x["suggest"] == "dispose")
        print(f"  model suggests dispose {n_d}, keep {len(items) - n_d}")


if __name__ == "__main__":
    main()

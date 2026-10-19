from __future__ import annotations

import argparse
import json

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


def model_queue(df: pl.DataFrame, X: np.ndarray, n: int, done: set[str], hard: bool) -> list[tuple[int, float]]:
    scores = pl.read_parquet(SCORES)
    if scores.height != df.height or scores["id"].to_list() != df["id"].to_list():
        raise SystemExit("scores.parquet is stale relative to the manifest - re-run probe")

    ids = df["id"].to_list()
    p = scores["p_dispose"].to_numpy()
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

    n_unc = int(n * 0.5)
    n_hi = int(n * 0.25)
    n_lo = n - n_unc - n_hi

    order = np.argsort(np.abs(p[pool] - np.median(p[pool])))
    unc = pool[order[:n_unc]]
    hi = pool[np.argsort(-p[pool])[:n_hi]]
    lo = pool[np.argsort(p[pool])[:n_lo]]
    chosen = list(dict.fromkeys([*unc, *hi, *lo]))

    rank_of = {int(i): float(pct[k]) for k, i in enumerate(pool)}
    ordered = [chosen[k] for k in nn_tour(X[chosen])]
    return [(i, rank_of[i]) for i in ordered]


def to_items(df: pl.DataFrame, idx: list[int], pct: dict[int, float] | None, assign: np.ndarray | None) -> list[dict]:
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
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["coverage", "model"], default="coverage")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--clusters", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", choices=["all", "hard"], default="all")
    args = ap.parse_args()

    df, X = load_aligned()
    done = labelled_ids()
    assign = clusters(X, args.clusters, args.seed)

    if args.mode == "coverage":
        idx = coverage(df, X, args.n, args.clusters, args.seed, done)
        items = to_items(df, idx, None, assign)
    else:
        pairs = model_queue(df, X, args.n, done, args.subset == "hard")
        idx = [i for i, _ in pairs]
        items = to_items(df, idx, {i: q for i, q in pairs}, assign)

    QUEUE.write_text(json.dumps(items, indent=1) + "\n")
    print(f"{args.mode} queue: {len(items)} images, {len({x['cluster'] for x in items})} clusters -> {QUEUE}")
    if items and "suggest" in items[0]:
        n_d = sum(1 for x in items if x["suggest"] == "dispose")
        print(f"  model suggests dispose {n_d}, keep {len(items) - n_d}")


if __name__ == "__main__":
    main()

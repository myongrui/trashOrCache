from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .config import DATA_DIR
from .store import LABELS, clusters, load_aligned

QUEUE = DATA_DIR / "queue.json"


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


def build(df: pl.DataFrame, X: np.ndarray, n: int, n_clusters: int, seed: int) -> list[dict]:
    assign = clusters(X, n_clusters, seed)
    cents = np.stack([X[assign == j].mean(0) for j in range(n_clusters)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)

    done = set()
    if LABELS.exists():
        for line in LABELS.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["asset_id"])

    ids = df["id"].to_list()
    taken = df["file_created_at"].to_list()
    names = df["originalFileName"].to_list()
    apps = df["source_app"].to_list()
    widths = df["width"].to_list()
    heights = df["height"].to_list()
    sizes = df["file_size"].to_list()
    ocr = df["ocr_chars"].to_list()
    faces = df["face_count"].to_list()
    unused = np.array([ids[i] not in done for i in range(len(ids))])

    per_cluster = max(1, int(np.ceil(n / n_clusters)))
    selected: list[int] = []
    seen: set[int] = set()

    for j in cluster_path(cents, int(np.argmax(cents.sum(1)))):
        members = np.flatnonzero((assign == j) & unused)
        if members.size == 0:
            continue
        anchor = int(members[np.argmax(X[members] @ cents[j])])
        picked = [anchor]
        while len(picked) < per_cluster and len(picked) < members.size:
            pool = members[~np.isin(members, picked)]
            if pool.size == 0:
                break
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
    selected = [selected[k] for k in nn_tour(X[selected])]
    return [
        {
            "asset_id": ids[i],
            "file_name": names[i],
            "source_app": apps[i],
            "taken": taken[i].strftime("%Y-%m-%d") if taken[i] else None,
            "width": widths[i],
            "height": heights[i],
            "bytes": sizes[i],
            "ocr_chars": ocr[i],
            "face_count": faces[i],
            "cluster": int(assign[i]),
        }
        for i in selected
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--clusters", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, X = load_aligned()
    queue = build(df, X, args.n, args.clusters, args.seed)
    QUEUE.write_text(json.dumps(queue, indent=1) + "\n")
    print(f"queue {len(queue)} images, {len({q['cluster'] for q in queue})} clusters -> {QUEUE}")


if __name__ == "__main__":
    main()

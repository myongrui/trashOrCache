from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import polars as pl

from .config import DATA_DIR
from .store import SCORES, load_aligned, quality_raw

MICRO = 800


def micro_clusters(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), min(k, len(X)), replace=False)].copy()
    a = np.zeros(len(X), dtype=np.int32)
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        for j in range(len(C)):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return a


def complete_link(members, S, order, thr, cap, days=None) -> list[list[int]]:
    groups: list[list[int]] = []
    for i in members[np.argsort(-order[members])]:
        for g in groups:
            if days is not None and any(days[j] != days[i] for j in g):
                continue
            if len(g) < cap and (S[g] @ S[i] >= thr).all():
                g.append(int(i))
                break
        else:
            groups.append([int(i)])
    return [g for g in groups if len(g) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--k", type=int, default=MICRO)
    ap.add_argument("--out", default=str(DATA_DIR / "clusters.json"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--same-day", action="store_true", help="only group images taken on the same date")
    args = ap.parse_args()

    df, X = load_aligned()
    S_all = np.load(DATA_DIR / "signatures.npy").astype(np.float32)
    q = pl.read_parquet(DATA_DIR / "quality.parquet")
    ids = df["id"].to_list()
    n = len(ids)

    in_dupe = np.zeros(n, dtype=bool)
    for g in json.loads((DATA_DIR / "groups.json").read_text()):
        for m in g["members"]:
            in_dupe[ids.index(m["asset_id"])] = True

    keep_score = (
        np.log1p((df["width"].fill_null(0) * df["height"].fill_null(0)).to_numpy())
        + q["sharpness_center"].to_numpy() * 0.15
        + q["face_area_max"].fill_null(0).to_numpy() * 2.0
    )
    print(f"{n} assets, {int(in_dupe.sum())} already in a 0.98 duplicate group "
          f"({in_dupe.mean():.1%}), {int((~in_dupe).sum())} eligible for variant grouping")

    days = np.array([d.strftime("%Y-%m-%d") if d else "" for d in df["file_created_at"].to_list()])
    pool = np.flatnonzero(~in_dupe)
    print(f"k-means with k={args.k} over the eligible pool (pixel signatures)...")
    assign = micro_clusters(S_all[pool], args.k)
    sizes = np.bincount(assign)
    print(f"  micro-clusters: {len(sizes)}, median size {int(np.median(sizes))}, max {sizes.max()}")

    thresholds = (0.86, 0.88, 0.90, 0.92, 0.94) if args.sweep else (args.thr,)
    print(f"\n  {'thr':>5} {'groups':>7} {'images':>7} {'%lib':>6} {'median':>7} {'max':>4}")
    chosen = None
    for thr in thresholds:
        groups = []
        for j in range(len(sizes)):
            mem = pool[assign == j]
            if mem.size < 2:
                continue
            groups.extend(complete_link(mem, S_all, keep_score, thr, args.cap, days if args.same_day else None))
        tot = sum(len(g) for g in groups)
        med = int(np.median([len(g) for g in groups])) if groups else 0
        mx = max((len(g) for g in groups), default=0)
        print(f"  {thr:5.2f} {len(groups):7d} {tot:7d} {100*tot/n:5.1f}% {med:7d} {mx:4d}")
        if thr == args.thr:
            chosen = groups

    if not args.sweep and chosen is not None:
        scores = pl.read_parquet(SCORES) if SCORES.exists() else None
        action = dict(zip(scores["id"].to_list(), scores["action"].to_list())) if scores is not None else {}
        out = []
        for gi, g in enumerate(sorted(chosen, key=len, reverse=True)):
            members = sorted(g, key=lambda i: -keep_score[i])
            out.append({
                "group_id": f"c{gi:05d}",
                "kind": "variant",
                "members": [{
                    "asset_id": ids[i],
                    "file_name": df["originalFileName"][i],
                    "width": int(df["width"][i] or 0),
                    "height": int(df["height"][i] or 0),
                    "bytes": int(df["file_size"][i] or 0),
                    "taken": df["file_created_at"][i].strftime("%Y-%m-%d") if df["file_created_at"][i] else None,
                    "source_app": df["source_app"][i],
                    "score": action.get(ids[i]),
                } for i in members],
            })
        (DATA_DIR / "clusters.json").write_text(json.dumps(out, indent=1) + "\n")

        in_bands = Counter(a for gi in chosen for a in
                           [action.get(ids[i]) for i in gi] if a)
        covered = {ids[i] for gi in chosen for i in gi}
        lab = {json.loads(l)["asset_id"] for l in open(DATA_DIR / "labels.jsonl")}
        print(f"\n{len(out)} variant groups, {len(covered)} images ({len(covered)/n:.1%} of library)")
        print(f"  action mix inside them: {dict(in_bands)}")
        print(f"  already labelled inside them: {len(covered & lab)}")
        print(f"  untouched groups: {sum(1 for gi in chosen if not ({(ids[i]) for i in gi} & lab))}")
        print(f"  size distribution: {dict(sorted(Counter(len(g) for g in chosen).items()))}")
        print(f"-> {DATA_DIR / 'clusters.json'}")


if __name__ == "__main__":
    main()

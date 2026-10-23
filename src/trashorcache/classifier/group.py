from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .config import DATA_DIR
from .store import load_aligned, signatures

CHUNK = 1000
MICRO = 800
KIND = "dupes"  # set by cli


def components(S: np.ndarray, thr: float) -> list[np.ndarray]:
    """Connected components over pairwise correlation >= thr."""
    n = len(S)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in range(0, n, CHUNK):
        C = S[s : s + CHUNK] @ S.T
        ii, jj = np.where(C >= thr)
        for a, b in zip(ii + s, jj):
            if a < b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
    comp = np.array([find(i) for i in range(n)])
    return [np.flatnonzero(comp == c) for c in np.unique(comp)]


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


def day_components(pool: np.ndarray, S: np.ndarray, days: np.ndarray, thr: float) -> list[list[int]]:
    parent = {int(i): int(i) for i in pool}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_day: dict[str, list[int]] = {}
    for i in pool:
        by_day.setdefault(days[i], []).append(int(i))
    for members in by_day.values():
        if len(members) < 2:
            continue
        M = S[members] @ S[members].T
        ii, jj = np.where(M >= thr)
        for a, b in zip(ii, jj):
            if a < b:
                ra, rb = find(members[a]), find(members[b])
                if ra != rb:
                    parent[ra] = rb
    out: dict[int, list[int]] = {}
    for i in pool:
        out.setdefault(find(int(i)), []).append(int(i))
    return [g for g in out.values() if len(g) > 1]


def merge_groups(groups, S, days, order, thr, cap) -> list[list[int]]:
    if not groups:
        return []
    ids = [i for g in groups for i in g]
    gid = [k for k, g in enumerate(groups) for _ in g]
    C = S[ids] @ S[ids].T
    parent = list(range(len(groups)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if gid[a] != gid[b] and C[a, b] >= thr and days[ids[a]] == days[ids[b]]:
                ra, rb = find(gid[a]), find(gid[b])
                if ra != rb:
                    parent[ra] = rb

    merged: dict[int, list[int]] = {}
    for k, g in enumerate(groups):
        merged.setdefault(find(k), []).extend(g)

    out: list[list[int]] = []
    for members in merged.values():
        if len(members) > 1:
            out.extend(complete_link(np.array(members), S, order, thr, cap, days))
    return out


def _context():
    df, X = load_aligned()
    S = signatures(df)
    sharp = df["sharpness_center"].fill_null(0.0).to_numpy()
    face = df["face_area_max"].fill_null(0.0).to_numpy()
    pixels = (df["width"].fill_null(0) * df["height"].fill_null(0)).to_numpy()
    sizes = df["file_size"].fill_null(0).to_numpy()
    app = df["source_app"].fill_null("").to_list()
    is_wa = np.array([a == "whatsapp" for a in app])
    keep_score = np.log1p(pixels) + sharp * 0.15 + np.log1p(sizes) * 0.2 - is_wa * 1.5 + face * 2.0
    days = np.array([d.strftime("%Y-%m-%d") if d else "" for d in df["file_created_at"].to_list()])
    return df, X, S, sharp, sizes, app, is_wa, keep_score, days


def _emit(df, groups, score, S, days, kind, out_path, extra=None):
    names = df["originalFileName"].to_list()
    sizes = df["file_size"].fill_null(0).to_numpy()
    app = df["source_app"].fill_null("").to_list()
    taken = df["file_created_at"].to_list()
    prefix = "g" if kind == "duplicate" else "c"
    out = []
    for gi, g in enumerate(sorted(groups, key=len, reverse=True)):
        members = sorted(g, key=lambda i: -score[i])
        entry = {
            "group_id": f"{prefix}{gi:05d}",
            "kind": kind,
            "members": [
                {
                    "asset_id": df["id"][i],
                    "file_name": names[i],
                    "width": int(df["width"][i] or 0),
                    "height": int(df["height"][i] or 0),
                    "bytes": int(sizes[i]),
                    "taken": taken[i].strftime("%Y-%m-%d") if taken[i] else None,
                    "source_app": app[i] or None,
                    "sharpness": round(float(score[i]), 3),
                    "corr_to_best": round(float(S[i] @ S[members[0]]), 4),
                }
                for i in members
            ],
        }
        if extra:
            entry.update(extra)
        out.append(entry)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return out


def dupes_main() -> None:
    ap = argparse.ArgumentParser(prog="toc dupes")
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--dup-thr", type=float, default=0.99)
    ap.add_argument("--out", default=str(DATA_DIR / "groups.json"))
    args = ap.parse_args()

    df, _, S, _, _, _, _, score, days = _context()
    print(f"grouping at corr >= {args.thr} ...")
    groups = [b.tolist() for b in components(S, args.thr) if len(b) > 1]
    out = _emit(df, groups, score, S, days, "duplicate", args.out)

    for g in out:
        sims = [m["corr_to_best"] for m in g["members"][1:]]
        g["kind"] = "duplicate" if sims and min(sims) >= args.dup_thr else "similar"
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    total = sum(len(g["members"]) for g in out)
    dups = sum(1 for g in out if g["kind"] == "duplicate")
    print(f"{len(out)} groups, {total} images ({total / len(df):.1%} of library)")
    print(f"  {dups} duplicate, {len(out) - dups} similar")
    print(f"  at most {total - len(out)} discardable keeping one per group")
    print("  size distribution:", dict(sorted(_sizes(out).items())))
    print(f"-> {args.out}")


def variants_main() -> None:
    ap = argparse.ArgumentParser(prog="toc variants")
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--k", type=int, default=MICRO)
    ap.add_argument("--out", default=str(DATA_DIR / "clusters.json"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--method", choices=["components", "complete"], default="complete")
    args = ap.parse_args()

    df, X, S, sharp, sizes, app, is_wa, keep_score, days = _context()
    ids = df["id"].to_list()
    n = len(ids)

    dupe_path = DATA_DIR / "groups.json"
    in_dupe = np.zeros(n, dtype=bool)
    if dupe_path.exists():
        for g in json.loads(dupe_path.read_text()):
            for m in g["members"]:
                in_dupe[ids.index(m["asset_id"])] = True
    print(f"{n} assets, {int(in_dupe.sum())} already in a duplicate group "
          f"({in_dupe.mean():.1%}), {int((~in_dupe).sum())} eligible")

    pool = np.flatnonzero(~in_dupe)
    if not args.sweep:
        print(f"k-means k={args.k} over the eligible pool...")
    assign = micro_clusters(S[pool], args.k)
    sizes_mc = np.bincount(assign)
    if not args.sweep:
        print(f"  micro-clusters: {len(sizes_mc)}, median {int(np.median(sizes_mc))}, max {sizes_mc.max()}")

    thresholds = (0.86, 0.88, 0.90, 0.92, 0.94) if args.sweep else (args.thr,)
    print(f"\n  {'thr':>5} {'groups':>7} {'images':>7} {'%lib':>6} {'median':>7} {'max':>4}")
    chosen = None
    for thr in thresholds:
        if args.method == "components":
            groups = day_components(pool, S, days, thr)
        else:
            groups = []
            for j in range(len(sizes_mc)):
                mem = pool[assign == j]
                if mem.size > 1:
                    groups.extend(complete_link(mem, S, keep_score, thr, args.cap, days))
            groups = merge_groups(groups, S, days, keep_score, thr, args.cap)
        tot = sum(len(g) for g in groups)
        med = int(np.median([len(g) for g in groups])) if groups else 0
        print(f"  {thr:5.2f} {len(groups):7d} {tot:7d} {100 * tot / n:5.1f}% {med:7d} "
              f"{max((len(g) for g in groups), default=0):4d}")
        if thr == args.thr:
            chosen = groups

    if args.sweep or chosen is None:
        return

    seen: dict[int, int] = {}
    clean: list[list[int]] = []
    pruned = 0
    for g in chosen:
        kept = [i for i in g if i not in seen]
        pruned += len(g) - len(kept)
        for i in kept:
            seen[i] = 1
        if len(kept) > 1:
            clean.append(kept)
    print(f"  invariant: {len(seen)} assets in exactly one group, duplicates removed: {pruned}")

    out = _emit(df, clean, keep_score, S, days, "variant", args.out)
    total = sum(len(g["members"]) for g in out)
    lab = {json.loads(l)["asset_id"] for l in open(DATA_DIR / "labels.jsonl")}
    covered = {m["asset_id"] for g in out for m in g["members"]}
    print(f"\n{len(out)} variant groups, {total} images ({total / n:.1%} of library)")
    print(f"  already labelled: {len(covered & lab)},   untouched groups: "
          f"{sum(1 for g in out if not ({m['asset_id'] for m in g['members']} & lab))}")
    print("  size distribution:", dict(sorted(_sizes(out).items())))
    print(f"-> {args.out}")


def _sizes(out):
    d = {}
    for g in out:
        d[len(g["members"])] = d.get(len(g["members"]), 0) + 1
    return d


def main() -> None:
    (dupes_main if KIND == "dupes" else variants_main)()


if __name__ == "__main__":
    main()

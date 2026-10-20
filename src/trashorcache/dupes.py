from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from .config import DATA_DIR
from .store import load_aligned

CHUNK = 1000


def components(S: np.ndarray, thr: float) -> list[np.ndarray]:
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





def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.98)
    ap.add_argument("--dup-thr", type=float, default=0.99)
    ap.add_argument("--out", default=str(DATA_DIR / "groups.json"))
    args = ap.parse_args()

    df, _ = load_aligned()
    q = pl.read_parquet(DATA_DIR / "quality.parquet")
    if q["id"].to_list() != df["id"].to_list():
        raise SystemExit("quality.parquet misaligned with manifest - re-run quality.py")

    S = np.load(DATA_DIR / "signatures.npy").astype(np.float32)
    sharp = q["sharpness_center"].to_numpy()
    pixels = (df["width"].fill_null(0) * df["height"].fill_null(0)).to_numpy()
    sizes = df["file_size"].fill_null(0).to_numpy()
    app = df["source_app"].fill_null("").to_list()
    names = df["originalFileName"].to_list()
    taken = df["file_created_at"].to_list()
    is_wa = np.array([a == "whatsapp" for a in app])

    face = q["face_area_max"].fill_null(0.0).to_numpy()
    score = np.log1p(pixels) + sharp * 0.15 + np.log1p(sizes) * 0.2 - is_wa * 1.5 + face * 2.0

    print(f"grouping at corr >= {args.thr} ...")
    groups = [b.tolist() for b in components(S, args.thr) if len(b) > 1]
    groups.sort(key=len, reverse=True)

    out = []
    for gi, g in enumerate(groups):
        members = sorted(g, key=lambda i: -score[i])
        sims = [float(S[members[0]] @ S[i]) for i in members[1:]]
        out.append(
            {
                "group_id": f"g{gi:05d}",
                "kind": "duplicate" if sims and min(sims) >= args.dup_thr else "similar",
                "members": [
                    {
                        "asset_id": df["id"][i],
                        "file_name": names[i],
                        "width": int(df["width"][i] or 0),
                        "height": int(df["height"][i] or 0),
                        "bytes": int(sizes[i]),
                        "taken": taken[i].strftime("%Y-%m-%d") if taken[i] else None,
                        "source_app": app[i] or None,
                        "sharpness": round(float(sharp[i]), 2),
                        "corr_to_best": round(float(S[i] @ S[members[0]]), 4),
                        "keep_score": round(float(score[i]), 3),
                    }
                    for i in members
                ],
            }
        )

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    total = sum(len(g) for g in out)
    dups = sum(1 for g in out if g["kind"] == "duplicate")
    print(f"\n{len(out)} groups covering {total} images ({total / len(df):.1%} of library)")
    print(f"  {dups} duplicate groups, {len(out) - dups} similar groups")
    print(f"  at most {total - len(out)} images discardable if you keep one per group")
    sizes_dist = {}
    for g in out:
        sizes_dist[len(g["members"])] = sizes_dist.get(len(g["members"]), 0) + 1
    print("  size distribution:", dict(sorted(sizes_dist.items())))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

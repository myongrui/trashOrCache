from __future__ import annotations

import argparse
import io
import subprocess
from multiprocessing import Pool

import numpy as np
import polars as pl
from PIL import Image

from .config import DATA_DIR, IMMICH_CONTAINER, IMMICH_DB
from .store import load_aligned

SIZE = 512
FIELDS = [
    "sharpness", "sharpness_center", "sharpness_edge", "clip_high", "clip_low",
    "luma_mean", "luma_std", "noise", "colorfulness", "entropy", "flat_bright",
    "face_area_max", "face_area_sum", "big_faces",
]

FACE_SQL = """
SELECT a.id AS id,
       ((f."boundingBoxX2" - f."boundingBoxX1")::double precision
        * (f."boundingBoxY2" - f."boundingBoxY1")::double precision)
         / greatest((f."imageWidth" * f."imageHeight")::double precision, 1) AS frac
FROM asset_face f JOIN asset a ON a.id = f."assetId"
WHERE f."deletedAt" IS NULL
"""


def laplacian(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="edge")
    return 4 * p[1:-1, 1:-1] - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def features(path: str) -> tuple | None:
    try:
        with Image.open(path) as im:
            im.draft("RGB", (SIZE, SIZE))
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > SIZE:
                s = SIZE / max(w, h)
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32)
    except Exception:
        return None

    g = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = laplacian(g)

    hh, ww = g.shape
    c = lap[hh // 4 : 3 * hh // 4, ww // 4 : 3 * ww // 4]
    ring = np.ones_like(g, dtype=bool)
    ring[hh // 4 : 3 * hh // 4, ww // 4 : 3 * ww // 4] = False

    dx = np.abs(np.diff(g, axis=1))[:-1, :]
    dy = np.abs(np.diff(g, axis=0))[:, :-1]
    grad = dx + dy
    flat = grad < np.percentile(grad, 25)
    vals = lap[:-1, :-1][flat]
    noise = float(np.median(np.abs(vals - np.median(vals)))) if vals.size > 100 else 0.0

    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colour = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    hist = np.bincount((g.clip(0, 255) / 4).astype(np.int32).ravel(), minlength=64).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]

    return (
        float(np.log1p(lap.var())),
        float(np.log1p(c.var())),
        float(np.log1p(lap[ring].var())),
        float((g >= 250).mean()),
        float((g <= 5).mean()),
        float(g.mean()),
        float(g.std()),
        float(np.log1p(noise)),
        colour,
        float(-(p * np.log2(p)).sum()),
        float((g >= 245).mean()),
    )


def face_features() -> pl.DataFrame:
    proc = subprocess.run(
        ["docker", "exec", "-e", "PGCLIENTENCODING=UTF8", IMMICH_CONTAINER, "psql", "-X", "-q",
         "-v", "ON_ERROR_STOP=1", "-U", "postgres", "-d", IMMICH_DB,
         "-c", f"COPY ({FACE_SQL}) TO STDOUT WITH (FORMAT csv, HEADER)"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    rows = proc.stdout.decode().strip().splitlines()
    if len(rows) <= 1:
        return pl.DataFrame({"id": [], "face_area_max": [], "face_area_sum": [], "big_faces": []})
    df = pl.read_csv(io.StringIO("\n".join(rows)), infer_schema_length=0).with_columns(
        pl.col("frac").cast(pl.Float64)
    )
    return df.group_by("id").agg(
        pl.col("frac").max().alias("face_area_max"),
        pl.col("frac").sum().alias("face_area_sum"),
        (pl.col("frac") > 0.01).sum().alias("big_faces"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df, _ = load_aligned()
    if args.limit:
        df = df.head(args.limit)
    paths = df["preview_path"].to_list()
    ids = df["id"].to_list()
    print(f"computing quality features for {len(paths)} previews with {args.workers} workers")

    with Pool(args.workers) as pool:
        rows = list(pool.imap(features, paths, chunksize=32))

    ok = [r for r in rows if r is not None]
    print(f"  ok {len(ok)}   failed {len(rows) - len(ok)}")
    px = pl.DataFrame(np.array(ok, dtype=np.float32), schema=FIELDS[:11], orient="row")
    px = px.with_columns(
        pl.Series("id", [i for i, r in zip(ids, rows) if r is not None])
    )

    q = (
        pl.DataFrame({"id": df["id"]})
        .join(px, on="id", how="left")
        .join(face_features(), on="id", how="left")
        .with_columns(pl.col("face_area_max", "face_area_sum", "big_faces").fill_null(0.0))
    )
    q.write_parquet(DATA_DIR / "quality.parquet", compression="zstd")

    print(f"\nwrote data/quality.parquet  {q.shape}  {q.estimated_size('mb'):.2f} MB")
    print(q.describe())


if __name__ == "__main__":
    main()

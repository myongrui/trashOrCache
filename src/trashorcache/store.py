from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl

from .config import DATA_DIR, LABELS_PATH

MANIFEST = DATA_DIR / "manifest.parquet"
LABELS = LABELS_PATH
SCORES = DATA_DIR / "scores.parquet"
MODEL = DATA_DIR / "model.npz"
QUALITY = DATA_DIR / "quality.parquet"

QUALITY_FIELDS = [
    "sharpness", "sharpness_center", "sharpness_edge", "clip_high", "clip_low",
    "luma_mean", "luma_std", "noise", "colorfulness", "entropy", "flat_bright",
    "face_area_max", "face_area_sum", "big_faces",
]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_aligned() -> tuple[pl.DataFrame, np.ndarray]:
    meta = json.loads((DATA_DIR / "embeddings.meta.json").read_text())
    if sha256(MANIFEST) != meta["manifest_sha256"]:
        raise SystemExit("manifest changed since embeddings were built - re-run embed")

    df = pl.read_parquet(MANIFEST).filter(pl.col("preview_exists"))
    X = np.load(DATA_DIR / "embeddings.npy")
    if df.height != meta["count"] or X.shape[0] != meta["count"]:
        raise SystemExit(f"row mismatch: manifest {df.height}, vectors {X.shape[0]}, meta {meta['count']}")
    return df, X


def read_labels() -> list[dict]:
    if not LABELS.exists():
        return []
    return [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]


def groups_near_dup(X: np.ndarray, thresh: float = 0.95) -> np.ndarray:
    n = len(X)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for start in range(0, n, 500):
        block = X[start : start + 500] @ X.T
        ii, jj = np.where(block > thresh)
        for a, b in zip(ii + start, jj):
            if a < b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def unique_labels() -> list[dict]:
    """Last write wins: a later group decision overrides an earlier single-image label."""
    latest = {}
    for r in read_labels():
        latest[r["asset_id"]] = r
    return list(latest.values())


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    assign = np.zeros(len(X), dtype=np.int32)
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            members = assign == j
            if members.any():
                C[j] = X[members].mean(0)
    return assign


def tabular_raw(df: pl.DataFrame) -> np.ndarray:
    ar = df["aspect_ratio"].fill_null(1.0).clip(0.25, 4.0)
    apps = set(df["source_app"].drop_nulls().to_list()) - {"whatsapp", "camera"}
    feats = [
        (df["ocr_chars"] > 0).cast(pl.Float32).alias("has_ocr"),
        df["ocr_chars"].log1p().alias("log_ocr"),
        (df["face_count"] > 0).cast(pl.Float32).alias("has_faces"),
        df["face_count"].clip(0, 10).cast(pl.Float32).alias("n_faces"),
        ar.cast(pl.Float32).alias("aspect"),
        df["file_size"].log1p().alias("log_bytes"),
        (df["width"] * df["height"]).log1p().alias("log_pixels"),
        (df["source_app"] == "whatsapp").cast(pl.Float32).alias("is_whatsapp"),
        (df["source_app"] == "camera").cast(pl.Float32).alias("is_camera"),
        df["source_app"].is_in(list(apps)).cast(pl.Float32).alias("is_screenshot_app"),
        df["latitude"].is_not_null().cast(pl.Float32).alias("has_gps"),
        df["make"].is_not_null().cast(pl.Float32).alias("has_make"),
        df["exif_taken_at"].is_not_null().cast(pl.Float32).alias("has_exif_date"),
    ]
    return np.nan_to_num(df.select(feats).to_numpy().astype(np.float32))


def quality_raw(df: pl.DataFrame) -> np.ndarray:
    q = pl.read_parquet(QUALITY).select(["id", *QUALITY_FIELDS])
    aligned = pl.DataFrame({"id": df["id"]}).join(q, on="id", how="left")
    return np.nan_to_num(aligned.select(QUALITY_FIELDS).to_numpy().astype(np.float32))


def standardize(T: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mu is None:
        mu, sd = T.mean(0), T.std(0)
        sd[sd == 0] = 1.0
    return (T - mu) / sd, mu, sd


def tabular(df: pl.DataFrame) -> np.ndarray:
    return standardize(tabular_raw(df))[0]


def clusters(X: np.ndarray, k: int = 150, seed: int = 0) -> np.ndarray:
    path = DATA_DIR / f"clusters_k{k}_s{seed}.npy"
    if path.exists() and len(np.load(path)) == len(X):
        return np.load(path)
    assign = kmeans(X, k, seed=seed)
    np.save(path, assign)
    return assign

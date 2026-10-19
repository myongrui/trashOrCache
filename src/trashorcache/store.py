from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl

from .config import DATA_DIR

MANIFEST = DATA_DIR / "manifest.parquet"
LABELS = DATA_DIR / "labels.jsonl"
SCORES = DATA_DIR / "scores.parquet"
MODEL = DATA_DIR / "model.npz"


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


def clusters(X: np.ndarray, k: int = 150, seed: int = 0) -> np.ndarray:
    path = DATA_DIR / f"clusters_k{k}_s{seed}.npy"
    if path.exists() and len(np.load(path)) == len(X):
        return np.load(path)
    assign = kmeans(X, k, seed=seed)
    np.save(path, assign)
    return assign

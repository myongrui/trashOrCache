from __future__ import annotations

import numpy as np
import polars as pl

from ..contract import db
from ..contract.paths import DATA_DIR, DECISIONS, DB, QUEUE, SCORES

ASSET_SELECT = """
SELECT a.asset_id AS id,
       a.original_file_name AS "originalFileName",
       a.source_app, a.width, a.height, a.file_size, a.aspect_ratio,
       a.file_created_at, a.imported_at, a.exif_taken_at, a.make, a.iso,
       a.latitude, a.longitude, a.city, a.country, a.ocr_chars, a.face_count,
       a.preview_path, a.preview_exists, a.decision, a.decided_at
FROM assets a
"""

DATETIME_COLUMNS = ["file_created_at", "imported_at", "exif_taken_at"]


def _frame(con, where: str = "", params: tuple = (), with_features: bool = True) -> pl.DataFrame:
    join = " JOIN vectors v ON v.asset_id = a.asset_id" if with_features else ""
    feats = ", ".join(f"f.{c}" for c in db.FEATURE_COLUMNS) if with_features else ""
    fjoin = " LEFT JOIN features f ON f.asset_id = a.asset_id" if with_features else ""
    select = ASSET_SELECT.replace("FROM assets a", "") 
    cols = select + (f", {feats}" if feats else "") + f" FROM assets a{join}{fjoin} {where} ORDER BY a.asset_id"
    rows = con.execute(cols, params).fetchall()
    df = pl.DataFrame([dict(r) for r in rows], infer_schema_length=None)
    for c in DATETIME_COLUMNS:
        if c in df.columns:
            df = df.with_columns(pl.col(c).str.to_datetime(strict=False))
    if "preview_exists" in df.columns:
        df = df.with_columns(pl.col("preview_exists").cast(pl.Boolean))
    return df


def load_aligned() -> tuple[pl.DataFrame, np.ndarray]:
    """Assets that have a vector, ordered by asset_id, with X row-aligned to df."""
    con = db.connect()
    df = _frame(con, with_features=True)
    vecs = con.execute(
        "SELECT v.vec FROM vectors v JOIN assets a ON a.asset_id = v.asset_id ORDER BY a.asset_id")
    X = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in vecs]).astype(np.float32)
    if len(df) != len(X):
        raise SystemExit(f"row mismatch: frame {len(df)}, vectors {len(X)}")
    models = [r[0] for r in con.execute("SELECT DISTINCT model_id FROM vectors")]
    if len(models) > 1:
        raise SystemExit(f"vectors from more than one model: {models} - re-embed before training")
    return df, X


def signatures(df: pl.DataFrame) -> np.ndarray:
    con = db.connect()
    ids = df["id"].to_list()
    out = np.zeros((len(ids), 1024), dtype=np.float32)
    lookup = {r[0]: r[1] for r in con.execute("SELECT asset_id, sig FROM signatures")}
    for i, a in enumerate(ids):
        blob = lookup.get(a)
        if blob:
            out[i] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    return out


def read_labels() -> list[dict]:
    con = db.connect()
    return [{"asset_id": r[0], "label": r[1], "source": r[2], "at": r[3]}
            for r in con.execute(
                "SELECT asset_id, decision, decision_source, decided_at "
                "FROM assets WHERE decision IS NOT NULL ORDER BY decided_at")]


def unique_labels() -> list[dict]:
    return read_labels()


def training_set(sources: tuple[str, ...] | None = None) -> list[dict]:
    rows = read_labels()
    if sources:
        rows = [r for r in rows if r["source"] in sources]
    return rows


def quality_raw(df: pl.DataFrame) -> np.ndarray:
    return np.nan_to_num(df.select(db.FEATURE_COLUMNS).to_numpy().astype(np.float32))


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
    T = df.select(feats).to_numpy().astype(np.float32)
    return np.nan_to_num(T)


def standardize(T: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mu is None:
        mu, sd = T.mean(0), T.std(0)
        sd[sd == 0] = 1.0
    return (T - mu) / sd, mu, sd


def tabular(df: pl.DataFrame) -> np.ndarray:
    return standardize(tabular_raw(df))[0]


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

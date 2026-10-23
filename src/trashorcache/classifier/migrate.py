"""One-off: import the existing parquet/npy/jsonl artifacts into toc.db.

Verifies the result against the source files before reporting success. After
this, the database is the source of truth and those files are just history.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ..contract import db
from ..contract.paths import EMBEDDINGS, LABELS, MANIFEST, QUALITY, SIGNATURES

ASSET_COLUMNS = [
    "asset_id", "original_file_name", "source_app", "width", "height", "file_size",
    "aspect_ratio", "file_created_at", "imported_at", "exif_taken_at", "make", "iso",
    "latitude", "longitude", "city", "country", "ocr_chars", "face_count",
    "preview_path", "preview_exists",
]


def _iso(v):
    return v.isoformat() if v is not None else None


def main() -> None:
    df = pl.read_parquet(MANIFEST)
    emb = np.load(EMBEDDINGS)
    sig = np.load(SIGNATURES)
    q = pl.read_parquet(QUALITY)
    if not (df.height == len(emb) == len(sig) == q.height):
        raise SystemExit(f"row mismatch: manifest {df.height}, emb {len(emb)}, sig {len(sig)}, q {q.height}")

    meta = json.loads((MANIFEST.parent / "embeddings.meta.json").read_text())
    model_id = f"{meta['model']}@{meta['manifest_sha256'][:12]}"

    con = db.connect()
    rows = []
    for i in range(df.height):
        rows.append((
            df["id"][i],
            df["originalFileName"][i],
            df["source_app"][i],
            df["width"][i],
            df["height"][i],
            df["file_size"][i],
            df["aspect_ratio"][i],
            _iso(df["file_created_at"][i]),
            _iso(df["imported_at"][i]),
            _iso(df["exif_taken_at"][i]),
            df["make"][i],
            df["iso"][i],
            df["latitude"][i],
            df["longitude"][i],
            df["city"][i],
            df["country"][i],
            df["ocr_chars"][i],
            df["face_count"][i],
            df["preview_path"][i],
            1 if df["preview_exists"][i] else 0,
        ))
    con.executemany(
        f"INSERT OR REPLACE INTO assets ({', '.join(ASSET_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(ASSET_COLUMNS))})", rows)

    con.executemany(
        "INSERT OR REPLACE INTO vectors (asset_id, model_id, embedded_at, vec) VALUES (?,?,?,?)",
        [(df["id"][i], model_id, db.now(), emb[i].astype(np.float32).tobytes()) for i in range(df.height)])
    con.executemany(
        "INSERT OR REPLACE INTO signatures (asset_id, sig) VALUES (?,?)",
        [(df["id"][i], sig[i].astype(np.float16).tobytes()) for i in range(df.height)])

    fcols = db.FEATURE_COLUMNS
    con.executemany(
        f"INSERT OR REPLACE INTO features (asset_id, {', '.join(fcols)}) "
        f"VALUES ({', '.join('?' * (len(fcols) + 1))})",
        [(q["id"][i], *[float(q[c][i]) if q[c][i] is not None else 0.0 for c in fcols])
         for i in range(q.height)])

    labelled = 0
    for line in LABELS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        con.execute(
            "UPDATE assets SET decision=?, decision_source=?, decided_at=? WHERE asset_id=?",
            (r["label"], r.get("source", "labeler"), r.get("at", db.now()), r["asset_id"]))
        labelled += 1
    con.commit()

    # ---- verification against the source files ----
    print("verifying...")
    n, decided, _, _ = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE decision IS NOT NULL), 0, 0 FROM assets").fetchone()
    unique_labels = {json.loads(l)["asset_id"]: json.loads(l)["label"]
                     for l in LABELS.read_text().splitlines() if l.strip()}
    mismatch = []
    for r in con.execute("SELECT asset_id, decision FROM assets WHERE decision IS NOT NULL"):
        if unique_labels.get(r[0]) != r[1]:
            mismatch.append(r[0])
    print(f"  assets {n} (manifest {df.height})")
    print(f"  decisions {decided} (unique labels {len(unique_labels)})")
    print(f"  label mismatches: {len(mismatch)}")

    # spot-check vectors byte-for-byte on a sample
    bad = 0
    for i in range(0, df.height, max(1, df.height // 200)):
        v = con.execute("SELECT vec FROM vectors WHERE asset_id=?", (df["id"][i],)).fetchone()[0]
        s = con.execute("SELECT sig FROM signatures WHERE asset_id=?", (df["id"][i],)).fetchone()[0]
        if not np.array_equal(np.frombuffer(v, dtype=np.float32), emb[i]):
            bad += 1
        if not np.array_equal(np.frombuffer(s, dtype=np.float16), sig[i]):
            bad += 1
    print(f"  vector spot-checks failing: {bad} (sampled {df.height // max(1, df.height // 200)})")
    print(f"  db size: {(MANIFEST.parent / 'toc.db').stat().st_size / 1e6:.1f} MB")
    if mismatch or bad:
        raise SystemExit("migration verification FAILED - files left untouched, db can be deleted")
    print("  OK")


if __name__ == "__main__":
    main()

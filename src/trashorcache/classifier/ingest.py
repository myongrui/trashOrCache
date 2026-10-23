from __future__ import annotations

import io
import re
import subprocess

import polars as pl

from ..contract import db
from .config import IMMICH_CONTAINER, IMMICH_DB, LIBRARY_DIR, SQL, THUMBS_DIR

TS = "%Y-%m-%dT%H:%M:%S%.6fZ"

DT_FIELDS = [
    "file_created_at",
    "file_modified_at",
    "local_date_time",
    "imported_at",
    "deleted_at",
    "exif_taken_at",
]

SCREENSHOT_APP = re.compile(r"^Screenshot_\d{8}_\d{6}_(\w+)", re.I)
WHATSAPP = re.compile(r"^IMG-\d{8}-WA\d+", re.I)
CAMERA = re.compile(r"^(IMG|PXL|DSC|DSCF|_MG)[-_]?\d{6,}", re.I)


def source_app(name: str | None) -> str | None:
    if not name:
        return None
    if m := SCREENSHOT_APP.match(name):
        return m.group(1).lower()
    if WHATSAPP.match(name):
        return "whatsapp"
    if CAMERA.match(name):
        return "camera"
    return None


def fetch() -> bytes:
    proc = subprocess.run(
        [
            "docker", "exec", "-e", "PGCLIENTENCODING=UTF8", IMMICH_CONTAINER,
            "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
            "-U", "postgres", "-d", IMMICH_DB,
            "-c", f"COPY ({SQL}) TO STDOUT WITH (FORMAT csv, HEADER)",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def build(raw: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)

    df = df.with_columns(
        pl.col("width").cast(pl.Int32),
        pl.col("height").cast(pl.Int32),
        pl.col("exif_width").cast(pl.Int32),
        pl.col("exif_height").cast(pl.Int32),
        pl.col("iso").cast(pl.Int32),
        pl.col("f_number").cast(pl.Float32),
        pl.col("focal_length").cast(pl.Float32),
        pl.col("file_size").cast(pl.Int64),
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64),
        pl.col("ocr_rows").cast(pl.Int32),
        pl.col("ocr_chars").cast(pl.Int32),
        pl.col("face_count").cast(pl.Int32),
        pl.col("is_favorite").eq("t"),
        pl.col("is_edited").eq("t"),
        pl.col("visibility"),
    )

    for col in DT_FIELDS:
        df = df.with_columns(pl.col(col).str.to_datetime(TS, strict=False))

    df = df.with_columns(
        pl.format(
            str(THUMBS_DIR) + "/{}/{}/{}/{}_preview.jpeg",
            pl.col("ownerId"),
            pl.col("id").str.slice(0, 2),
            pl.col("id").str.slice(2, 2),
            pl.col("id"),
        ).alias("preview_path"),
    )

    df = df.with_columns(
        pl.col("preview_path").map_elements(lambda p: _exists(p), return_dtype=pl.Boolean)
        .alias("preview_exists"),
        pl.col("originalPath")
        .str.replace("/data/", str(LIBRARY_DIR) + "/", literal=True)
        .alias("original_path_host"),
    )

    df = df.drop("originalPath", "ownerId", "type")

    df = df.with_columns(
        pl.col("originalFileName").map_elements(source_app, return_dtype=pl.Utf8).alias("source_app"),
        (pl.col("width") / pl.col("height")).round(4).alias("aspect_ratio"),
        (
            (pl.col("file_modified_at") - pl.col("file_created_at")).dt.total_seconds() / 86400
        ).round(2).alias("modified_minus_created_days"),
    )

    return df


def _exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()


COLUMNS = [
    "asset_id", "original_file_name", "source_app", "width", "height", "file_size",
    "aspect_ratio", "file_created_at", "imported_at", "exif_taken_at", "make", "iso",
    "latitude", "longitude", "city", "country", "ocr_chars", "face_count",
    "preview_path", "preview_exists",
]


def _iso(v):
    return v.isoformat() if v is not None else None


def main() -> None:
    raw = fetch()
    df = build(raw)
    con = db.connect()

    rows = [(
        df["id"][i], df["originalFileName"][i], df["source_app"][i], df["width"][i],
        df["height"][i], df["file_size"][i], df["aspect_ratio"][i],
        _iso(df["file_created_at"][i]), _iso(df["imported_at"][i]), _iso(df["exif_taken_at"][i]),
        df["make"][i], df["iso"][i], df["latitude"][i], df["longitude"][i], df["city"][i],
        df["country"][i], df["ocr_chars"][i], df["face_count"][i], df["preview_path"][i],
        1 if df["preview_exists"][i] else 0,
    ) for i in range(df.height)]

    con.executemany(
        f"INSERT INTO assets ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))}) "
        "ON CONFLICT(asset_id) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in COLUMNS[1:]) + ", purged_at=NULL", rows)

    present = {r[0] for r in con.execute("SELECT asset_id FROM assets")}
    live = set(df["id"].to_list())
    vanished = present - live
    if vanished:
        con.executemany("UPDATE assets SET purged_at=?, preview_exists=0 WHERE asset_id=?",
                        [(db.now(), a) for a in vanished])
    con.commit()

    n = df.height
    have = int(df["preview_exists"].sum())
    print(f"rows           {n}")
    print(f"previews found {have}  ({n - have} missing)")
    print(f"trashed        {int(df['deleted_at'].is_not_null().sum())}")
    print(f"ocr present    {int((df['ocr_chars'] > 0).sum())}")
    print(f"faces present  {int((df['face_count'] > 0).sum())}")
    print(f"gps present    {int(df['latitude'].is_not_null().sum())}")
    c = db.counts(con)
    print(f"newly purged   {len(vanished)} (rows kept, vectors kept)")
    print(f"db             {c}")


if __name__ == "__main__":
    main()

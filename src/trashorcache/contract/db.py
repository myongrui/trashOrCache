"""The one database: assets, their vectors, their features, and their decisions.

Nothing here is ever deleted. An asset purged from Immich keeps its row, its
vector and its label, because that is what the training set is made of.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .paths import DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  asset_id           TEXT PRIMARY KEY,
  original_file_name TEXT,
  source_app         TEXT,
  width              INTEGER,
  height             INTEGER,
  file_size          INTEGER,
  aspect_ratio       REAL,
  file_created_at    TEXT,
  imported_at        TEXT,
  exif_taken_at      TEXT,
  make               TEXT,
  iso                INTEGER,
  latitude           REAL,
  longitude          REAL,
  city               TEXT,
  country            TEXT,
  ocr_chars          INTEGER,
  face_count         INTEGER,
  preview_path       TEXT,
  preview_exists     INTEGER NOT NULL DEFAULT 0,
  purged_at          TEXT,

  decision           TEXT,
  decision_source    TEXT,
  decided_at         TEXT,

  batch_id           TEXT,
  action             TEXT,
  tag                TEXT,
  tag_id             TEXT,
  group_id           TEXT,
  proposed_at        TEXT
);

CREATE TABLE IF NOT EXISTS vectors (
  asset_id    TEXT PRIMARY KEY,
  model_id    TEXT NOT NULL,
  embedded_at TEXT NOT NULL,
  vec         BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS signatures (
  asset_id TEXT PRIMARY KEY,
  sig      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS features (
  asset_id          TEXT PRIMARY KEY,
  sharpness         REAL,
  sharpness_center  REAL,
  sharpness_edge    REAL,
  clip_high         REAL,
  clip_low          REAL,
  luma_mean         REAL,
  luma_std          REAL,
  noise             REAL,
  colorfulness      REAL,
  entropy           REAL,
  flat_bright       REAL,
  face_area_max     REAL,
  face_area_sum     REAL,
  big_faces         REAL
);

CREATE INDEX IF NOT EXISTS assets_pending ON assets(proposed_at) WHERE decision IS NULL;
CREATE INDEX IF NOT EXISTS assets_decided ON assets(decided_at) WHERE decision IS NOT NULL;
CREATE INDEX IF NOT EXISTS assets_purged  ON assets(purged_at) WHERE purged_at IS NOT NULL;
"""

FEATURE_COLUMNS = [
    "sharpness", "sharpness_center", "sharpness_edge", "clip_high", "clip_low",
    "luma_mean", "luma_std", "noise", "colorfulness", "entropy", "flat_bright",
    "face_area_max", "face_area_sum", "big_faces",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(path=None) -> sqlite3.Connection:
    target = path or DB
    con = sqlite3.connect(target, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def counts(con: sqlite3.Connection) -> dict[str, int]:
    row = con.execute("""
        SELECT count(*),
               count(*) FILTER (WHERE decision IS NOT NULL),
               count(*) FILTER (WHERE decision IS NULL AND proposed_at IS NOT NULL),
               count(*) FILTER (WHERE purged_at IS NOT NULL)
        FROM assets""").fetchone()
    return {"assets": row[0], "decided": row[1], "pending": row[2], "purged": row[3]}

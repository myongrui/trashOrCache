"""The proposal ledger: a SQLite file both sides read and write.

One row per asset (re-proposing upserts). `decided_at IS NULL` is the work queue;
rows are deleted once their outcome has been folded into labels.jsonl.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .paths import LEDGER

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
  asset_id    TEXT PRIMARY KEY,
  batch_id    TEXT NOT NULL,
  action      TEXT NOT NULL,
  tag         TEXT NOT NULL,
  tag_id      TEXT,
  group_id    TEXT,
  proposed_at TEXT NOT NULL,
  decided_at  TEXT,
  outcome     TEXT
);
CREATE INDEX IF NOT EXISTS proposals_pending ON proposals(decided_at) WHERE decided_at IS NULL;
"""

COLUMNS = ["asset_id", "batch_id", "action", "tag", "tag_id", "group_id", "proposed_at"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(LEDGER, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, rows: list[dict]) -> int:
    con.executemany(
        f"INSERT INTO proposals ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))}) "
        "ON CONFLICT(asset_id) DO UPDATE SET "
        "batch_id=excluded.batch_id, action=excluded.action, tag=excluded.tag, tag_id=excluded.tag_id, "
        "group_id=excluded.group_id, proposed_at=excluded.proposed_at, decided_at=NULL, outcome=NULL",
        [tuple(r[c] for c in COLUMNS) for r in rows],
    )
    con.commit()
    return len(rows)


def pending(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM proposals WHERE decided_at IS NULL ORDER BY proposed_at, asset_id")]


def pending_ids(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT asset_id FROM proposals WHERE decided_at IS NULL")}


def decided(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM proposals WHERE decided_at IS NOT NULL ORDER BY decided_at")]


def mark_decided(con: sqlite3.Connection, outcomes: dict[str, str]) -> int:
    at = now()
    con.executemany(
        "UPDATE proposals SET decided_at = ?, outcome = ? WHERE asset_id = ? AND decided_at IS NULL",
        [(at, outcome, asset_id) for asset_id, outcome in outcomes.items()],
    )
    con.commit()
    return len(outcomes)


def prune_decided(con: sqlite3.Connection) -> int:
    cur = con.execute("DELETE FROM proposals WHERE decided_at IS NOT NULL")
    con.commit()
    return cur.rowcount


def counts(con: sqlite3.Connection) -> dict[str, int]:
    total, pend = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE decided_at IS NULL) FROM proposals").fetchone()
    return {"total": total, "pending": pend, "decided": total - pend}

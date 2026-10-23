from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..config import PORT, QUEUE_PATH
from ...contract import db

QUEUE = QUEUE_PATH
STATIC = Path(__file__).parent / "static"

app = FastAPI()

QUEUE_ITEMS: list[dict] = json.loads(QUEUE.read_text())
_con = db.connect()
PATHS: dict[str, str] = {r[0]: r[1] for r in _con.execute(
    "SELECT asset_id, preview_path FROM assets WHERE preview_path IS NOT NULL")}

VALID = {"keep", "dispose", "skip"}


class Label(BaseModel):
    asset_id: str
    label: str


class GroupDecision(BaseModel):
    group_id: str
    keep: list[str]
    dispose: list[str]

def read_labels() -> list[dict]:
    con = db.connect()
    return [{"asset_id": r[0], "label": r[1], "source": r[2], "at": r[3], "group_id": r[4]}
            for r in con.execute(
                "SELECT asset_id, decision, decision_source, decided_at, group_id "
                "FROM assets WHERE decision IS NOT NULL ORDER BY decided_at")]


def decide(asset_id: str, label: str, source: str, group_id: str | None = None) -> None:
    con = db.connect()
    con.execute(
        "UPDATE assets SET decision=?, decision_source=?, decided_at=?, group_id=COALESCE(?, group_id) "
        "WHERE asset_id=?", (label, source, db.now(), group_id, asset_id))
    con.commit()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/queue")
def queue() -> JSONResponse:
    labeled = [r["asset_id"] for r in read_labels()]
    return JSONResponse({"items": QUEUE_ITEMS, "labeled": labeled})


@app.get("/api/stats")
def stats() -> JSONResponse:
    rows = read_labels()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return JSONResponse({"total": len(QUEUE_ITEMS), "labeled": len(rows), "counts": counts})


@app.get("/img/{asset_id}")
def image(asset_id: str) -> FileResponse:
    path = PATHS.get(asset_id)
    if path is None:
        raise HTTPException(404, "unknown asset")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/label")
def add(body: Label) -> JSONResponse:
    if body.label not in VALID:
        raise HTTPException(400, f"bad label {body.label}")

    decide(body.asset_id, body.label, "labeler")
    return JSONResponse({"ok": True, "asset_id": body.asset_id, "label": body.label, "at": db.now()})


@app.post("/api/label_group")
def label_group(body: GroupDecision) -> JSONResponse:
    for a in body.keep:
        decide(a, "keep", "labeler", body.group_id)
    for a in body.dispose:
        decide(a, "dispose", "labeler", body.group_id)
    return JSONResponse({"ok": True, "rows": len(body.keep) + len(body.dispose)})


@app.get("/api/agreement")
def agreement() -> JSONResponse:
    lab = {r["asset_id"]: r["label"] for r in read_labels()}
    done = [x for x in QUEUE_ITEMS if x.get("suggest") and x["asset_id"] in lab]
    if not done:
        return JSONResponse({"n": 0})

    def pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 1) if b else 0.0

    hit = sum(1 for x in done if lab[x["asset_id"]] == x["suggest"])
    rand = [x for x in done if x.get("stratum") == "random"]
    rand_hit = sum(1 for x in rand if lab[x["asset_id"]] == x["suggest"])
    out = {
        "n": len(done),
        "hit": hit,
        "miss": len(done) - hit,
        "hit_pct": pct(hit, len(done)),
        "miss_pct": pct(len(done) - hit, len(done)),
    }
    if rand:
        out["random"] = {"n": len(rand), "hit": rand_hit, "hit_pct": pct(rand_hit, len(rand))}
    for label in ("dispose", "keep"):
        calls = [x for x in done if x["suggest"] == label]
        h = sum(1 for x in calls if lab[x["asset_id"]] == label)
        out[f"{label}_calls"] = {"n": len(calls), "hit": h, "hit_pct": pct(h, len(calls))}
    return JSONResponse(out)


@app.post("/api/undo")
def undo() -> JSONResponse:
    con = db.connect()
    last = con.execute(
        "SELECT asset_id, decision, decided_at, group_id FROM assets "
        "WHERE decision IS NOT NULL ORDER BY decided_at DESC, asset_id DESC LIMIT 1").fetchone()
    if last is None:
        raise HTTPException(400, "nothing to undo")
    group_id = last[3]
    if group_id:
        victims = [r[0] for r in con.execute(
            "SELECT asset_id FROM assets WHERE decided_at=? AND group_id=?",
            (last[2], group_id))]
    else:
        victims = [last[0]]
    con.executemany(
        "UPDATE assets SET decision=NULL, decision_source=NULL, decided_at=NULL WHERE asset_id=?",
        [(a,) for a in victims])
    con.commit()
    return JSONResponse({"ok": True,
                         "dropped": {"asset_id": last[0], "label": last[1], "group_id": group_id},
                         "rows": len(victims)})


def main() -> None:
    print(f"db     -> {db.DB}\nqueue  -> {QUEUE}\nserving on http://127.0.0.1:{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

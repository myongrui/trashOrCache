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

from .config import DATA_DIR, LABELS_PATH, PORT, QUEUE_PATH

LABELS = LABELS_PATH
QUEUE = QUEUE_PATH
STATIC = Path(__file__).parent / "static"

app = FastAPI()

QUEUE_ITEMS: list[dict] = json.loads(QUEUE.read_text())
PATHS: dict[str, str] = dict(
    zip(
        pl.read_parquet(DATA_DIR / "manifest.parquet")["id"].to_list(),
        pl.read_parquet(DATA_DIR / "manifest.parquet")["preview_path"].to_list(),
    )
)

VALID = {"keep", "dispose", "skip"}


class Label(BaseModel):
    asset_id: str
    label: str

def read_labels() -> list[dict]:
    if not LABELS.exists():
        return []
    return [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]


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

    row = {
        "asset_id": body.asset_id,
        "label": body.label,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with LABELS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return JSONResponse({"ok": True, **row})


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
    rows = read_labels()
    if not rows:
        raise HTTPException(400, "nothing to undo")
    dropped = rows.pop()
    shutil.copy2(LABELS, LABELS.with_suffix(".jsonl.bak"))
    tmp = LABELS.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, LABELS)
    return JSONResponse({"ok": True, "dropped": dropped})


def main() -> None:
    print(f"labels -> {LABELS}\nqueue  -> {QUEUE}\nserving on http://127.0.0.1:{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

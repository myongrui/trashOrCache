"""Turn the model's internal bands + groups into the public three-action contract."""

from __future__ import annotations

import json

import polars as pl

from ..contract.models import Decision
from ..contract.paths import DATA_DIR, DECISIONS, SCORES
from .store import unique_labels

BAND_TO_ACTION = {
    "auto_dispose": "dispose",
    "redundant": "dispose",
    "upgrade": "review",
    "review": "review",
}


def _why(band: str, source_app: str | None, ocr_chars: int) -> str:
    if band == "redundant":
        return "duplicate of kept"
    if band == "upgrade":
        return "better copy of kept"
    if band == "auto_dispose":
        if source_app == "whatsapp":
            return "whatsapp"
        if ocr_chars > 0:
            return "ocr text"
        return "model"
    return "uncertain"


def build(df: pl.DataFrame) -> tuple[list[Decision], dict]:
    pos = {a: i for i, a in enumerate(df["id"].to_list())}
    sc = pl.read_parquet(SCORES)
    band = dict(zip(sc["id"].to_list(), sc["action"].to_list()))
    score = dict(zip(sc["id"].to_list(), sc["score"].to_list()))
    if len(band) != df.height:
        raise SystemExit("scores.parquet is stale relative to the manifest - run `toc-classifier run`")

    labelled = {r["asset_id"]: r["label"] for r in unique_labels() if not r.get("recheck")}
    app = df["source_app"].to_list()
    ocr = df["ocr_chars"].to_list()

    groups = []
    for name in ("groups.json", "clusters.json"):
        path = DATA_DIR / name
        if path.exists():
            groups += json.loads(path.read_text())

    out: dict[str, Decision] = {}
    stats = {"dispose_all": 0, "keep_one": 0, "groups_skipped": 0}

    def public(asset_id: str) -> str:
        if asset_id in labelled:
            return labelled[asset_id]
        return BAND_TO_ACTION.get(band.get(asset_id, "review"), "review")

    in_group: set[str] = set()
    for g in groups:
        members = [m["asset_id"] for m in g["members"]]
        in_group.update(members)
        actionable = [a for a in members if a not in labelled]
        if not actionable:
            stats["groups_skipped"] += 1
            continue

        keeps = [a for a in members if labelled.get(a) == "keep"]
        dispose_all = not keeps and all(public(a) == "dispose" for a in members)
        gaction = "dispose_all" if dispose_all else "keep_one"
        stats[gaction] += 1

        if dispose_all:
            keeper = members[0]
        elif keeps:
            keeper = keeps[0]
        else:
            keeper = next((a for a in members if labelled.get(a) != "dispose"), members[0])

        for a in members:
            if gaction == "dispose_all":
                act = "dispose"
            elif a == keeper:
                act = "keep"
            else:
                act = public(a)
            i = pos[a]
            out[a] = Decision(
                asset_id=a,
                action=act,
                score=round(float(score.get(a, 0.0)), 4),
                group_id=g["group_id"],
                group=gaction,
                keeper=(a == keeper),
                why=(f"labelled {labelled[a]}" if a in labelled
                     else "group keeper" if a == keeper
                     else _why(band[a], app[i], ocr[i])),
            )

    for a in df["id"].to_list():
        if a in in_group or a in labelled or a in out:
            continue
        i = pos[a]
        out[a] = Decision(
            asset_id=a,
            action=public(a),
            score=round(float(score[a]), 4),
            why=_why(band[a], app[i], ocr[i]),
        )

    return list(out.values()), stats


def write(df: pl.DataFrame) -> dict:
    decisions, stats = build(df)
    with DECISIONS.open("w") as fh:
        for d in decisions:
            fh.write(json.dumps(d.model_dump()) + "\n")

    counts: dict[str, int] = {}
    whys: dict[str, int] = {}
    for d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1
        whys[d.why] = whys.get(d.why, 0) + 1
    return {"rows": len(decisions), "actions": counts, "whys": whys, **stats}

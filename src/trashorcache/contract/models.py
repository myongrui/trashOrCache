"""The interface between classifier and flow. Both sides import this; nothing else."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Action = Literal["dispose", "review", "keep"]
GroupAction = Literal["dispose_all", "keep_one"]


class Decision(BaseModel):
    """One row per asset the flow should act on. Written by the classifier."""

    asset_id: str
    action: Action
    score: float
    group_id: str | None = None
    group: GroupAction | None = None
    keeper: bool | None = None
    why: str = ""


class Proposal(BaseModel):
    """What the flow actually tagged, so the next run can diff against it."""

    asset_id: str
    action: Action
    group_id: str | None = None
    proposed_at: str


class Feedback(BaseModel):
    """What the human did in Immich. Deletions are hard labels; tag removals are keeps."""

    deleted: list[str] = []
    kept: list[str] = []
    source: str = "immich"
    observed_at: str = ""

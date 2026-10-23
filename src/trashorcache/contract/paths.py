"""The file interface between classifier and flow. Single source of truth for paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("TOC_DATA", ROOT / "data"))

MANIFEST = DATA_DIR / "manifest.parquet"
EMBEDDINGS = DATA_DIR / "embeddings.npy"
QUALITY = DATA_DIR / "quality.parquet"
SIGNATURES = DATA_DIR / "signatures.npy"
SCORES = DATA_DIR / "scores.parquet"
MODEL = DATA_DIR / "model.npz"

LABELS = Path(os.environ.get("TOC_LABELS", DATA_DIR / "labels.jsonl"))
QUEUE = Path(os.environ.get("TOC_QUEUE", DATA_DIR / "queue.json"))

DECISIONS = DATA_DIR / "decisions.jsonl"
DB = DATA_DIR / "toc.db"

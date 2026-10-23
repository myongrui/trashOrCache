from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SiglipVisionModel

from ..contract import db
from .config import DATA_DIR, MODEL_ID


class Previews(Dataset):
    def __init__(self, paths: list[str], processor):
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    con = db.connect()
    todo = con.execute(
        "SELECT a.asset_id, a.preview_path FROM assets a "
        "LEFT JOIN vectors v ON v.asset_id = a.asset_id "
        "WHERE a.preview_exists = 1 AND v.asset_id IS NULL ORDER BY a.asset_id").fetchall()
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"all previews already embedded ({db.counts(con)})")
        return
    ids = [r[0] for r in todo]
    paths = [r[1] for r in todo]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device {device}  dtype {dtype}  images {len(paths)}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipVisionModel.from_pretrained(MODEL_ID).to(device).to(dtype).eval()

    loader = DataLoader(
        Previews(paths, processor),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.workers > 0,
    )

    out = np.empty((len(paths), model.config.hidden_size), dtype=np.float32)
    start = time.time()
    done = 0

    with torch.inference_mode():
        for batch in loader:
            emb = model(pixel_values=batch.to(device, non_blocking=True)).pooler_output
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            out[done : done + emb.shape[0]] = emb.cpu().numpy()
            done += emb.shape[0]
            if done % (args.batch_size * 10) == 0 or done == len(paths):
                rate = done / (time.time() - start)
                print(f"  {done}/{len(paths)}  {rate:.0f} img/s", flush=True)

    elapsed = time.time() - start
    if not np.isfinite(out).all():
        raise SystemExit("non-finite values in embeddings")

    con.executemany(
        "INSERT OR REPLACE INTO vectors (asset_id, model_id, embedded_at, vec) VALUES (?,?,?,?)",
        [(ids[i], MODEL_ID, db.now(), out[i].tobytes()) for i in range(len(ids))])
    con.commit()

    meta = {
        "model": MODEL_ID,
        "output": "pooler_output",
        "hidden_size": int(model.config.hidden_size),
        "normalized": True,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "count": int(out.shape[0]),
        "db": str(db.DB),
        "processor": processor.to_dict(),
        "torch": torch.__version__,
        "elapsed_seconds": round(elapsed, 1),
        "images_per_second": round(len(paths) / elapsed, 1),
    }
    (DATA_DIR / "embeddings.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    norms = np.linalg.norm(out, axis=1)
    print(f"wrote {out.shape} float32 -> toc.db ({db.counts(con)['assets']} assets)")
    print(f"norm mean {norms.mean():.4f}  min {norms.min():.4f}  max {norms.max():.4f}")
    print(f"identical rows: {int(len(out) - len(np.unique(out, axis=0)))}")


if __name__ == "__main__":
    main()

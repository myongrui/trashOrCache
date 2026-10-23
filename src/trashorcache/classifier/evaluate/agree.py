"""Label self-agreement: how often two independent passes give the same verdict."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ...contract.paths import DATA_DIR
from ..store import read_labels


def main() -> None:
    first = {r["asset_id"]: r["label"] for r in read_labels()}
    path = DATA_DIR / "labels_agree.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found - run a review round with TOC_LABELS set first")
    second = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    pairs = [(first[r["asset_id"]], r["label"]) for r in second if r["asset_id"] in first]
    if not pairs:
        raise SystemExit("no overlap between the two label files")

    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in ("keep", "dispose"))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    print(f"re-labelled      {len(pairs)}")
    print(f"self-agreement   {po:.1%}  ({int((a == b).sum())}/{len(pairs)})")
    print(f"cohen's kappa    {kappa:.3f}")
    print(f"                 {dict(Counter(zip(a, b)))}")
    print("\nflip directions:")
    for k, v in Counter((x, y) for x, y in zip(a, b) if x != y).most_common():
        print(f"  {k[0]:8s} -> {k[1]:8s}  {v}")
    print(f"\naccuracy ceiling for any model: roughly {po + 0.02:.0%}")
    print(f"majority-class baseline: {(a == Counter(a).most_common(1)[0][0]).mean():.1%}")

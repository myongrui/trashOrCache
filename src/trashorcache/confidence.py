from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from .store import (
    DATA_DIR, SCORES, groups_near_dup, load_aligned, quality_raw, standardize, tabular_raw,
    unique_labels,
)

KNN_K = 25


def oof(F: np.ndarray, V: np.ndarray, y: np.ndarray, folds, C: float) -> np.ndarray:
    lr_o = np.zeros(len(y))
    knn_o = np.zeros(len(y))
    for tr, te in folds:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(F[tr], y[tr])
        k = KNeighborsClassifier(KNN_K, metric="cosine", weights="distance").fit(V[tr], y[tr])
        lr_o[te] = m.predict_proba(F[te])[:, 1]
        knn_o[te] = k.predict_proba(V[te])[:, 1]
    return 0.5 * (rankdata(lr_o) / len(y) + rankdata(knn_o) / len(y))


def band_at_precision(score: np.ndarray, y: np.ndarray, target: float, min_n: int) -> dict | None:
    """Smallest top-band whose cross-validated precision reaches `target`."""
    best = None
    for q in np.arange(0.50, 0.9995, 0.0025):
        t = np.quantile(score, q)
        sel = score >= t
        if sel.sum() < min_n:
            continue
        prec = float(y[sel].mean())
        if prec >= target:
            cand = {"quantile": float(q), "n": int(sel.sum()), "precision": prec,
                    "recall": float(sel[y == 1].mean())}
            if best is None or cand["quantile"] < best["quantile"]:
                best = cand
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--min-n", type=int, default=80)
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    df, X = load_aligned()
    pos = {a: i for i, a in enumerate(df["id"].to_list())}
    rows = [r for r in unique_labels() if r["label"] in ("keep", "dispose") and r["asset_id"] in pos]
    idx = np.array([pos[r["asset_id"]] for r in rows])
    y = np.array([r["label"] == "dispose" for r in rows], dtype=np.int64)
    print(f"{len(y)} unique labels   dispose {y.mean():.1%}")

    T, mu, sd = standardize(tabular_raw(df))
    Q = standardize(quality_raw(df))[0]
    F = np.c_[X, T, Q]
    folds = list(GroupKFold(n_splits=5).split(F[idx], y, groups_near_dup(X[idx])))
    s_oof = oof(F[idx], X[idx], y, folds, args.C)

    print("\n  in-sample-of-actively-sampled-labels tradeoff (treat as optimistic):")
    print(f"    {'target':>7} {'band':>7} {'precision':>10} {'recall':>7} {'n':>6}")
    for target in (0.90, 0.95, 0.98, 0.99):
        b = band_at_precision(s_oof, y, target, args.min_n)
        if b:
            print(f"    {target:7.2f} {100*(1-b['quantile']):6.1f}% {b['precision']:10.3f} "
                  f"{b['recall']:6.1%} {b['n']:6d}")

    chosen = band_at_precision(s_oof, y, args.target_precision, args.min_n)
    if chosen is None:
        raise SystemExit("no band reaches the target precision")

    lr = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(F[idx], y)
    knn = KNeighborsClassifier(KNN_K, metric="cosine", weights="distance").fit(X[idx], y)
    raw = 0.5 * (rankdata(lr.predict_proba(F)[:, 1]) + rankdata(knn.predict_proba(X)[:, 1]))
    thr = float(np.quantile(raw, chosen["quantile"]))
    labeled = np.isin(np.arange(len(df)), idx)
    action = np.where(labeled, "labeled", np.where(raw >= thr, "auto_dispose", "review"))

    pl.DataFrame({
        "id": df["id"], "score": (raw / raw.max()).astype(np.float32),
        "action": action, "labeled": labeled,
    }).write_parquet(SCORES)

    print(f"\n  target precision {args.target_precision:.0%} -> cutoff at top {100*(1-chosen['quantile']):.0f}%")
    for name in ("auto_dispose", "review", "labeled"):
        print(f"    {name:14s} {int((action == name).sum()):6d}")

    sel = df.filter(pl.Series(action == "auto_dispose"))
    print(f"\n  auto_dispose composition ({sel.height}):")
    print(f"    whatsapp filename {int((sel['source_app'] == 'whatsapp').fill_null(False).sum())}")
    print(f"    has OCR text      {int((sel['ocr_chars'] > 0).sum())}")
    print(f"    camera or GPS     {int((sel['make'].is_not_null() | sel['latitude'].is_not_null()).sum())}")

    (DATA_DIR / "confidence_report.json").write_text(json.dumps(
        {"target_precision": args.target_precision, "C": args.C, "band": chosen,
         "threshold": thr, "counts": {n: int((action == n).sum()) for n in set(action)}}, indent=2))
    print("\n  NOTE: this precision is measured on actively-sampled labels. Validate the band")
    print("  with `toc queue --mode validate --band auto_dispose` before tagging anything.")


if __name__ == "__main__":
    main()

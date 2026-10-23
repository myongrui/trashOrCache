"""Method comparison harness. Not part of the pipeline."""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from ..config import DATA_DIR
from ..store import groups_near_dup, load_aligned, quality_raw, read_labels, standardize, tabular


def _score_row(y, p) -> dict:
    pos, neg = y == 1, y == 0
    return {
        "auc": float(roc_auc_score(y, p)),
        "bal": float(0.5 * (((p >= 0.5) & pos).sum() / max(pos.sum(), 1)
                            + ((p < 0.5) & neg).sum() / max(neg.sum(), 1))),
        "acc": float(((p >= 0.5) == y).mean()),
        "p@100": float(y[np.argsort(-p)[:100]].mean()),
        "keep@100": float((y[np.argsort(p)[:100]] == 0).mean()),
    }


def bench_main() -> None:
    ap = argparse.ArgumentParser(prog="toc bench")
    ap.add_argument("--out", default=str(DATA_DIR / "bench_report.json"))
    args = ap.parse_args()

    df, X = load_aligned()
    pos = {a: i for i, a in enumerate(df["id"].to_list())}
    rows = [r for r in read_labels() if r["label"] in ("keep", "dispose") and r["asset_id"] in pos]
    idx = np.array([pos[r["asset_id"]] for r in rows])
    y = np.array([r["label"] == "dispose" for r in rows], dtype=np.int64)
    Xl, Tl = X[idx], tabular(df)[idx]
    Ql = standardize(quality_raw(df))[0][idx]
    hard = df["ocr_chars"].to_numpy()[idx] == 0

    folds = list(GroupKFold(n_splits=5).split(Xl, y, groups_near_dup(Xl)))
    print(f"labelled {len(y)}  dispose {y.mean():.1%}  hard {int(hard.sum())}")

    def oof(fn, *mats):
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = fn(*([m[tr] for m in mats] + [m[te] for m in mats] + [y[tr]]))
        return p

    def lr_pred(a, b, yy, C):
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(a, yy).predict_proba(b)[:, 1]

    def hgb(a, b, yy):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15).fit(
            a, yy).predict_proba(b)[:, 1]

    def knn(a, b, yy):
        return KNeighborsClassifier(25, metric="cosine", weights="distance").fit(a, yy).predict_proba(b)[:, 1]

    M = {"rule wa|ocr": np.where(
        (df["source_app"].to_numpy()[idx] == "whatsapp") | (df["ocr_chars"].to_numpy()[idx] > 0), 1.0, 0.0)}
    M["knn vision"] = oof(knn, Xl)
    for C in (0.03, 0.3):
        M[f"lr vision C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), Xl)
        M[f"lr meta C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), Tl)
        M[f"lr quality C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), Ql)
        M[f"lr meta+qual C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), np.c_[Tl, Ql])
        M[f"lr vis+meta+qual C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), np.c_[Xl, Tl, Ql])
    M["hgb meta"] = oof(hgb, Tl)
    M["hgb quality"] = oof(hgb, Ql)

    allsel = np.ones(len(y), bool)
    summary = {}
    for sel, title in ((allsel, "ALL labels"), (hard, "HARD subset (no OCR text)")):
        print(f"\n{title}")
        print(f"  {'method':24s} {'auc':>6} {'bal':>6} {'acc':>6} {'p@100':>6} {'keep@100':>9}")
        for name, p in sorted(M.items(), key=lambda kv: -roc_auc_score(y[sel], kv[1][sel])):
            r = _score_row(y[sel], p[sel])
            print(f"  {name:24s} {r['auc']:6.3f} {r['bal']:6.3f} {r['acc']:6.3f} "
                  f"{r['p@100']:6.3f} {r['keep@100']:9.3f}")
        summary[title] = {name: _score_row(y[sel], p[sel]) for name, p in M.items()}

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n-> {args.out}")


def main() -> None:
    bench_main()


if __name__ == "__main__":
    main()

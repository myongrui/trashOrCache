from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .store import (
    DATA_DIR, MODEL, SCORES, clusters, load_aligned, read_labels, standardize, tabular_raw,
)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    out = {
        "auc": float(roc_auc_score(y, p)),
        "bal_acc": float(0.5 * (((p >= 0.5) & pos).sum() / max(pos.sum(), 1)
                                + ((p < 0.5) & neg).sum() / max(neg.sum(), 1))),
        "acc": float(((p >= 0.5) == y).mean()),
    }
    order = np.argsort(-p)
    out["precision@100"] = float(y[order[:100]].mean())
    out["keep@100"] = float((y[np.argsort(p)[:100]] == 0).mean())
    return out


def oof(F: np.ndarray, y: np.ndarray, folds, C: float) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in folds:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(F[tr], y[tr])
        p[te] = m.predict_proba(F[te])[:, 1]
    return p


def main() -> None:
    df, X = load_aligned()
    pos = {a: i for i, a in enumerate(df["id"].to_list())}
    rows = [r for r in read_labels() if r["label"] in ("keep", "dispose") and r["asset_id"] in pos]
    idx = np.array([pos[r["asset_id"]] for r in rows])
    y = np.array([r["label"] == "dispose" for r in rows], dtype=np.int64)
    print(f"labelled {len(y)}  dispose {y.mean():.1%}")

    T, mu, sd = standardize(tabular_raw(df))
    F = np.c_[X, T]
    groups = clusters(X, 150, 0)[idx]
    folds = list(GroupKFold(n_splits=5).split(F[idx], y, groups))

    print("\n  feature set comparison (same folds):")
    for name, Fm in (("vision", X), ("metadata", T), ("vision+metadata", F)):
        best_c = max((0.003, 0.01, 0.03, 0.1), key=lambda C: roc_auc_score(y, oof(Fm[idx], y, folds, C)))
        m = metrics(y, oof(Fm[idx], y, folds, best_c))
        print(f"    {name:16s} C={best_c:<6} auc {m['auc']:.3f}  bal_acc {m['bal_acc']:.3f}  keep@100 {m['keep@100']:.3f}")

    best_c, best_p = None, None
    for C in (0.003, 0.01, 0.03, 0.1):
        p = oof(F[idx], y, folds, C)
        if best_p is None or roc_auc_score(y, p) > roc_auc_score(y, best_p):
            best_c, best_p = C, p
    res = metrics(y, best_p)
    print(f"\n  chosen vision+metadata C={best_c}")
    for k, v in res.items():
        print(f"    {k:14s} {v:.3f}")
    print(f"    {'baseline acc':14s} {max(y.mean(), 1 - y.mean()):.3f}   (always dispose)")

    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000).fit(F[idx], y)
    p_all = model.predict_proba(F)[:, 1]
    np.savez(MODEL, coef=model.coef_[0], intercept=model.intercept_[0], C=best_c, mu=mu, sd=sd)
    pl.DataFrame(
        {"id": df["id"], "p_dispose": p_all, "labeled": np.isin(np.arange(len(df)), idx)}
    ).write_parquet(SCORES)
    (DATA_DIR / "probe_report.json").write_text(
        json.dumps({"C": best_c, **res, "n": int(len(y)), "features": "vision+metadata"}, indent=2)
    )

    un = ~np.isin(np.arange(len(df)), idx)
    print(f"\nwrote {MODEL.name}, {SCORES.name}; unlabelled {int(un.sum())}")


if __name__ == "__main__":
    main()

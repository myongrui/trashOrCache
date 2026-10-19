from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from .store import (
    DATA_DIR, MODEL, SCORES, groups_near_dup, load_aligned, quality_raw, read_labels, standardize,
    tabular_raw,
)

KNN_K = 25


def norm_rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def report(name: str, y: np.ndarray, p: np.ndarray, hard: np.ndarray) -> dict:
    def block(sel):
        yy, pp = y[sel], p[sel]
        order = np.argsort(-pp)
        return {
            "auc": float(roc_auc_score(yy, pp)),
            "bal_acc": float(0.5 * (((pp >= 0.5) & (yy == 1)).sum() / max((yy == 1).sum(), 1)
                                    + ((pp < 0.5) & (yy == 0)).sum() / max((yy == 0).sum(), 1))),
            "acc": float(((pp >= 0.5) == yy).mean()),
            "precision@100": float(yy[order[:100]].mean()),
            "keep@100": float((yy[np.argsort(pp)[:100]] == 0).mean()),
        }
    out = {"all": block(np.ones(len(y), bool)), "hard": block(hard)}
    print(f"  {name:14s} auc {out['all']['auc']:.3f} (hard {out['hard']['auc']:.3f})   "
          f"bal_acc {out['all']['bal_acc']:.3f}   keep@100 {out['all']['keep@100']:.3f}")
    return out


def main() -> None:
    df, X = load_aligned()
    pos = {a: i for i, a in enumerate(df["id"].to_list())}
    rows = [r for r in read_labels() if r["label"] in ("keep", "dispose") and r["asset_id"] in pos]
    idx = np.array([pos[r["asset_id"]] for r in rows])
    y = np.array([r["label"] == "dispose" for r in rows], dtype=np.int64)
    hard = df["ocr_chars"].to_numpy()[idx] == 0
    print(f"labelled {len(y)}  dispose {y.mean():.1%}  hard subset {int(hard.sum())}")

    T, mu, sd = standardize(tabular_raw(df))
    Q = standardize(quality_raw(df))[0]
    F = np.c_[X, T, Q]

    folds = list(GroupKFold(n_splits=5).split(F[idx], y, groups_near_dup(X[idx])))

    print("\n  cross-validated (folds grouped by near-duplicate components):")
    best_c, results = None, {}
    for C in (0.003, 0.01, 0.03, 0.1):
        lr_oof = np.zeros(len(y))
        knn_oof = np.zeros(len(y))
        for tr, te in folds:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(F[idx][tr], y[tr])
            k = KNeighborsClassifier(KNN_K, metric="cosine", weights="distance").fit(X[idx][tr], y[tr])
            lr_oof[te] = m.predict_proba(F[idx][te])[:, 1]
            knn_oof[te] = k.predict_proba(X[idx][te])[:, 1]
        bl = 0.5 * (norm_rank(lr_oof) + norm_rank(knn_oof))
        if best_c is None or roc_auc_score(y[hard], bl[hard]) > results["blend"]["hard"]["auc"]:
            best_c = C
            results = {
                "lr": report(f"lr C={C}", y, lr_oof, hard),
                "knn": report("knn vision", y, knn_oof, hard),
                "blend": report("blend", y, bl, hard),
            }
    print(f"\n  chosen C={best_c} (selected on hard-subset AUC)")

    lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000).fit(F[idx], y)
    knn = KNeighborsClassifier(KNN_K, metric="cosine", weights="distance").fit(X[idx], y)
    score = 0.5 * (norm_rank(lr.predict_proba(F)[:, 1]) + norm_rank(knn.predict_proba(X)[:, 1]))

    np.savez(MODEL, coef=lr.coef_[0], intercept=lr.intercept_[0], C=best_c, mu=mu, sd=sd)
    pl.DataFrame(
        {"id": df["id"], "p_dispose": score, "labeled": np.isin(np.arange(len(df)), idx)}
    ).write_parquet(SCORES)
    (DATA_DIR / "probe_report.json").write_text(
        json.dumps({"C": best_c, "knn_k": KNN_K, "cv": results, "n": int(len(y))}, indent=2)
    )

    un = ~np.isin(np.arange(len(df)), idx)
    print(f"  unlabelled scored: {int(un.sum())}")
    print(f"  score is a percentile blend of lr + knn, not a probability")


if __name__ == "__main__":
    main()

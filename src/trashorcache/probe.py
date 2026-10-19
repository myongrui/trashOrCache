from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .store import DATA_DIR, MODEL, SCORES, clusters, load_aligned, read_labels


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    tp = float(((p >= 0.5) & pos).sum()) / max(pos.sum(), 1)
    tn = float(((p < 0.5) & neg).sum()) / max(neg.sum(), 1)
    out = {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "acc_at_0.5": float(((p >= 0.5) == y).mean()),
        "bal_acc_at_0.5": 0.5 * (tp + tn),
    }
    order = np.argsort(-p)
    for k in (50, 100, 200, 400):
        if k <= len(y):
            out[f"precision@{k}"] = float(y[order[:k]].mean())
    asc = np.argsort(p)
    for k in (100, 200):
        if k <= len(y):
            out[f"precision@{k}_lowest"] = float((y[asc[:k]] == 0).mean())
    return out


def tabular(df: pl.DataFrame) -> np.ndarray:
    ar = df["aspect_ratio"].fill_null(1.0).clip(0.25, 4.0)
    apps = set(df["source_app"].drop_nulls().to_list()) - {"whatsapp", "camera"}
    feats = [
        (df["ocr_chars"] > 0).cast(pl.Float32).alias("has_ocr"),
        df["ocr_chars"].log1p().alias("log_ocr"),
        (df["face_count"] > 0).cast(pl.Float32).alias("has_faces"),
        df["face_count"].clip(0, 10).cast(pl.Float32).alias("n_faces"),
        ar.cast(pl.Float32).alias("aspect"),
        df["file_size"].log1p().alias("log_bytes"),
        (df["width"] * df["height"]).log1p().alias("log_pixels"),
        (df["source_app"] == "whatsapp").cast(pl.Float32).alias("is_whatsapp"),
        (df["source_app"] == "camera").cast(pl.Float32).alias("is_camera"),
        df["source_app"].is_in(list(apps)).cast(pl.Float32).alias("is_screenshot_app"),
        df["latitude"].is_not_null().cast(pl.Float32).alias("has_gps"),
        df["make"].is_not_null().cast(pl.Float32).alias("has_make"),
        df["exif_taken_at"].is_not_null().cast(pl.Float32).alias("has_exif_date"),
    ]
    T = df.select(feats).to_numpy().astype(np.float32)
    T = np.nan_to_num(T)
    mu, sd = T.mean(0), T.std(0)
    sd[sd == 0] = 1.0
    return (T - mu) / sd


def main() -> None:
    df, X = load_aligned()
    pos = {a: i for i, a in enumerate(df["id"].to_list())}

    rows = [r for r in read_labels() if r["label"] in ("keep", "dispose")]
    idx = np.array([pos[r["asset_id"]] for r in rows if r["asset_id"] in pos])
    y = np.array([r["label"] == "dispose" for r in rows if r["asset_id"] in pos], dtype=np.int64)
    print(f"labelled {len(y)}  dispose {int(y.sum())} ({y.mean():.1%})  keep {int((~y.astype(bool)).sum())}")

    groups = clusters(X, 150, 0)[idx]
    folds = list(GroupKFold(n_splits=5).split(X[idx], y, groups))

    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 1.0):
        oof = np.zeros(len(y))
        for tr, te in folds:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
            m.fit(X[idx[tr]], y[tr])
            oof[te] = m.predict_proba(X[idx[te]])[:, 1]
        res = metrics(y, oof)
        print(f"  C={C:<6} auc {res['auc']:.3f}  brier {res['brier']:.3f}  "
              f"bal_acc {res['bal_acc_at_0.5']:.3f}  p@200 {res.get('precision@200', float('nan')):.3f}")
        if best is None or res["auc"] > best[1]["auc"]:
            best = (C, res, oof)

    C, res, oof = best
    print(f"\nchosen C={C}")
    for k, v in res.items():
        print(f"  {k:16s} {v:.3f}")

    baseline = max(y.mean(), 1 - y.mean())
    print(f"\n  baseline always-dispose accuracy {baseline:.3f}   baseline AUC 0.500")

    app = df["source_app"].fill_null("-").to_list()
    labels_src = [app[i] for i in idx]
    print("\n  AUC by filename source (top 6 by count):")
    counts = pl.Series("source_app", labels_src).value_counts().sort("count", descending=True)
    for name in counts["source_app"][:6]:
        sel = np.array([s == name for s in labels_src])
        if sel.sum() >= 20 and len(set(y[sel])) == 2:
            print(f"    {name:12s} n={sel.sum():4d}  auc {roc_auc_score(y[sel], oof[sel]):.3f}")
        else:
            print(f"    {name:12s} n={sel.sum():4d}  (too few / single class)")

    print("\n  metadata-only baseline (no pixels):")
    T = tabular(df)
    oof_t = np.zeros(len(y))
    for tr, te in folds:
        m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        m.fit(T[idx[tr]], y[tr])
        oof_t[te] = m.predict_proba(T[idx[te]])[:, 1]
    rt = metrics(y, oof_t)
    print(f"    auc {rt['auc']:.3f}  bal_acc {rt['bal_acc_at_0.5']:.3f}  "
          f"p@200 {rt.get('precision@200', float('nan')):.3f}")
    both = np.c_[X[idx], T[idx]]
    oof_b = np.zeros(len(y))
    for tr, te in folds:
        m = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000)
        m.fit(both[tr], y[tr])
        oof_b[te] = m.predict_proba(both[te])[:, 1]
    rb = metrics(y, oof_b)
    print(f"    vision+metadata auc {rb['auc']:.3f}  bal_acc {rb['bal_acc_at_0.5']:.3f}  "
          f"p@200 {rb.get('precision@200', float('nan')):.3f}")

    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X[idx], y)
    np.savez(MODEL, coef=model.coef_[0], intercept=model.intercept_[0], C=C)

    p_all = model.predict_proba(X)[:, 1]
    pl.DataFrame(
        {"id": df["id"], "p_dispose": p_all, "labeled": np.isin(np.arange(len(df)), idx)}
    ).write_parquet(SCORES)

    (DATA_DIR / "probe_report.json").write_text(json.dumps({"C": C, **res, "n": int(len(y))}, indent=2))
    print(f"\nwrote {MODEL.name}, {SCORES.name}, probe_report.json")
    print(f"unlabelled scored: {int((~np.isin(np.arange(len(df)), idx)).sum())}")
    print(f"  of those, p>0.9: {int(((p_all > 0.9) & ~np.isin(np.arange(len(df)), idx)).sum())}"
          f"   p<0.1: {int(((p_all < 0.1) & ~np.isin(np.arange(len(df)), idx)).sum())}")


if __name__ == "__main__":
    main()

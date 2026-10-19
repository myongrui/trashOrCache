from __future__ import annotations

import json

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from .store import groups_near_dup, load_aligned, quality_raw, read_labels, standardize, tabular

def mlp_fit_predict(Xtr, ytr, Xte, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.4),
        torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 1),
    )
    w = torch.tensor([len(ytr) / max(2 * ytr.sum(), 1), len(ytr) / max(2 * (len(ytr) - ytr.sum()), 1)])
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w[1] / w[0])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(250):
        opt.zero_grad()
        loss = lossf(net(xt).squeeze(1), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32)).squeeze(1)).numpy()


def score_row(y, p) -> dict:
    pos, neg = y == 1, y == 0
    out = {
        "auc": roc_auc_score(y, p),
        "bal": 0.5 * (((p >= 0.5) & pos).sum() / max(pos.sum(), 1) + ((p < 0.5) & neg).sum() / max(neg.sum(), 1)),
        "acc": ((p >= 0.5) == y).mean(),
    }
    o = np.argsort(-p)
    out["p@100"] = y[o[:100]].mean()
    out["keep@100"] = (y[np.argsort(p)[:100]] == 0).mean()
    return out


def main() -> None:
    df, X = load_aligned()
    pos = {a: i for i, a in enumerate(df["id"].to_list())}
    rows = [r for r in read_labels() if r["label"] in ("keep", "dispose") and r["asset_id"] in pos]
    idx = np.array([pos[r["asset_id"]] for r in rows])
    y = np.array([r["label"] == "dispose" for r in rows], dtype=np.int64)
    Xl, Tl = X[idx], tabular(df)[idx]
    Ql = standardize(quality_raw(df))[0][idx]

    hard = (df["ocr_chars"].to_numpy()[idx] == 0)

    g = groups_near_dup(Xl)
    folds = list(GroupKFold(n_splits=5).split(Xl, y, g))
    print(f"labelled {len(y)}  dispose {y.mean():.1%}  near-dup groups {len(set(g))}  "
          f"hard subset {hard.sum()} (dispose {y[hard].mean():.1%})\n")

    def oof(fn, *mats):
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = fn(*([m[tr] for m in mats] + [m[te] for m in mats] + [y[tr]]))
        return p

    def lr_pred(Xtr, Xte, ytr, C):
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    methods = {}
    methods["rule (wa or ocr)"] = np.where(
        (df["source_app"].to_numpy()[idx] == "whatsapp") | (df["ocr_chars"].to_numpy()[idx] > 0), 1.0, 0.0
    )
    methods["knn vision k=25"] = oof(
        lambda a, b, yy: KNeighborsClassifier(n_neighbors=25, metric="cosine", weights="distance")
        .fit(a, yy).predict_proba(b)[:, 1], Xl)
    for C in (0.003, 0.03, 0.3):
        methods[f"lr vision C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), Xl)
    methods["mlp vision"] = oof(lambda a, b, yy: mlp_fit_predict(a, yy, b), Xl)
    for C in (0.03, 0.3):
        methods[f"lr meta C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), Tl)
    methods["hgb meta"] = oof(
        lambda a, b, yy: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15)
        .fit(a, yy).predict_proba(b)[:, 1], Tl)
    for C in (0.03, 0.3):
        methods[f"lr quality C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), Ql)
    methods["hgb quality"] = oof(
        lambda a, b, yy: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15)
        .fit(a, yy).predict_proba(b)[:, 1], Ql)
    for C in (0.03, 0.3):
        methods[f"lr meta+quality C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), np.c_[Tl, Ql])
    for C in (0.03, 0.3):
        methods[f"lr vis+meta+qual C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), np.c_[Xl, Tl, Ql])
    methods["mlp vis+meta+qual"] = oof(lambda a, b, yy: mlp_fit_predict(a, yy, b), np.c_[Xl, Tl, Ql])
    methods["mlp vision + meta"] = oof(lambda a, b, yy: mlp_fit_predict(a, yy, b), np.c_[Xl, Tl])
    for C in (0.003, 0.03):
        methods[f"lr vision + meta C={C}"] = oof(lambda a, b, yy, C=C: lr_pred(a, b, yy, C), np.c_[Xl, Tl])

    def table(sel, title):
        print(title)
        print(f"  {'method':26s} {'auc':>6} {'bal_acc':>8} {'acc':>6} {'p@100':>6} {'keep@100':>9}")
        for name, p in sorted(methods.items(), key=lambda kv: -roc_auc_score(y[sel], kv[1][sel])):
            r = score_row(y[sel], p[sel])
            print(f"  {name:26s} {r['auc']:6.3f} {r['bal']:8.3f} {r['acc']:6.3f} "
                  f"{r['p@100']:6.3f} {r['keep@100']:9.3f}")
        print()

    table(np.ones(len(y), bool), "ALL labels")
    table(hard, "HARD subset (no OCR text)")

    summary = {
        name: {"all": score_row(y, p), "hard": score_row(y[hard], p[hard])} for name, p in methods.items()
    }
    (open("data/bench_report.json", "w")).write(json.dumps(summary, indent=2, default=float) + "\n")
    print("wrote data/bench_report.json")


if __name__ == "__main__":
    main()

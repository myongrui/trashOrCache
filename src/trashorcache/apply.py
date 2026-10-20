from __future__ import annotations

import argparse
import collections

import polars as pl

from .store import DATA_DIR, SCORES, load_aligned, unique_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "dispose_candidates.csv"))
    args = ap.parse_args()

    df, _ = load_aligned()
    labels = unique_labels()
    print(f"{len(labels)} unique labelled assets (last write wins)")

    lab = pl.DataFrame(
        {
            "id": [r["asset_id"] for r in labels],
            "label": [r["label"] for r in labels],
            "origin": ["group" if r.get("group_id") else "single" for r in labels],
            "at": [r["at"] for r in labels],
        }
    )
    d = df.join(lab, on="id", how="inner")
    if SCORES.exists():
        d = d.join(pl.read_parquet(SCORES).select("id", "score"), on="id", how="left")

    counts = collections.Counter(d["label"])
    print(f"  dispose {counts['dispose']}  keep {counts['keep']}")
    print(f"  of the disposals: {collections.Counter(d.filter(pl.col('label')=='dispose')['origin'])}")

    dis = d.filter(pl.col("label") == "dispose").sort("file_size", descending=True)
    cols = ["id", "originalFileName", "origin", "file_created_at", "width", "height",
            "file_size", "source_app", "ocr_chars", "original_path_host"]
    if "score" in dis.columns:
        cols.insert(2, "score")
    out = dis.select(cols).rename({"id": "asset_id", "originalFileName": "file_name"})
    out.write_csv(args.out)

    total = dis["file_size"].sum()
    print(f"\n{out.height} dispose candidates, {total / 1e9:.2f} GB")
    print(f"  by source: {dict(collections.Counter(dis['source_app'].fill_null('-').to_list()).most_common(5))}")
    print(f"  -> {args.out}")
    print("\nnothing has been deleted or tagged. this is a CSV for review.")


if __name__ == "__main__":
    main()

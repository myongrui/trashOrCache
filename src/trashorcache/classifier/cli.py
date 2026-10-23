from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "ingest": ("ingest", "pull Immich metadata into toc.db (read-only)"),
    "embed": ("embed", "SigLIP2 -> toc.db vectors"),
    "features": ("features", "pixel / face / signature features -> toc.db"),
    "dupes": ("group", "near-identical groups (corr >= 0.98)"),
    "variants": ("group", "same-day variant groups (corr >= 0.92)"),
    "run": ("model", "fit the filter, write scores + action bands"),
    "bootstrap": ("bootstrap", "cold start: rules -> decisions.jsonl, no model"),
    "review": ("review.server", "serve the labeler UI"),
    "queue": ("review.queue", "build a review queue"),
    "bench": ("evaluate.bench", "method comparison harness"),
    "agree": ("evaluate.agree", "label self-agreement (kappa)"),
    "export": ("export", "write labels (and optionally vectors) out as files"),
    "migrate": ("migrate", "import legacy parquet/npy/jsonl into toc.db"),
    "restore": ("restore", "restore decisions from a labels jsonl export"),
}

KINDS = {"dupes": "dupes", "variants": "variants"}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: toc-classifier <command> [options]\n")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:10s} {desc}")
        print("\nexamples:\n  toc-classifier run\n  toc-classifier review")
        return

    name = argv[0]
    if name not in COMMANDS:
        raise SystemExit(f"unknown command {name!r}. run `toc-classifier --help`")

    mod_name, _ = COMMANDS[name]
    module = importlib.import_module(f".{mod_name}", __package__)
    if name in KINDS:
        module.KIND = KINDS[name]
    sys.argv = [f"toc-classifier {name}", *argv[1:]]
    module.main()


if __name__ == "__main__":
    main()

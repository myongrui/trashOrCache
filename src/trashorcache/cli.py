from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "extract": "extract the Immich manifest",
    "embed": "SigLIP2 embeddings",
    "quality": "pixel + face + hash features",
    "dupes": "near-duplicate groups",
    "clusters": "variant groups (looser than duplicates)",
    "queue": "build a review queue",
    "labeler": "serve the review UI",
    "probe": "train LR + kNN, score the library",
    "confidence": "precision-targeted bands + action column",
    "immich": "create/assign tags in Immich",
    "apply": "export dispose candidates to CSV",
    "agree": "self-agreement / kappa report",
    "bench": "method comparison harness",
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: toc <command> [options]\n")
        for name, desc in COMMANDS.items():
            print(f"  {name:12s} {desc}")
        print("\nexamples:\n  toc confidence --target-precision 0.95\n  toc labeler")
        return

    name = argv[0]
    if name not in COMMANDS:
        raise SystemExit(f"unknown command {name!r}. run `toc --help`")

    module = importlib.import_module(f".{name}", __package__)
    sys.argv = [f"toc {name}", *argv[1:]]
    module.main()


if __name__ == "__main__":
    main()

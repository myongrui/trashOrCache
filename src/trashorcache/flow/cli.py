from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "propose": ("propose", "apply decisions.jsonl to Immich tags"),
    "tag": ("immich", "tag a raw band directly (debugging)"),
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: toc-flow <command> [options]\n")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:10s} {desc}")
        return

    name = argv[0]
    if name not in COMMANDS:
        raise SystemExit(f"unknown command {name!r}. run `toc-flow --help`")

    mod_name, _ = COMMANDS[name]
    module = importlib.import_module(f".{mod_name}", __package__)
    sys.argv = [f"toc-flow {name}", *argv[1:]]
    module.main()


if __name__ == "__main__":
    main()

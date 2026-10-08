"""LANG v0 Gate 3A: safe preflight; optional evaluation of an already completed run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aster.training.lang_gate3 import preflight_report, evaluate_completed_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, help="existing runs/<id>; required for evaluate")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "preflight":
        if args.run is not None:
            parser.error("--run is not valid for preflight")
        print(json.dumps(preflight_report(root), ensure_ascii=False, indent=2))
    else:
        if args.run is None:
            parser.error("evaluate requires --run")
        output = evaluate_completed_run(root, root / args.run)
        print(json.dumps({"gate3_evaluation": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Register one verified DecisionModel artifact in the local Aster Service catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.service.artifacts import ArtifactCatalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    catalog = ArtifactCatalog(args.root)
    ref = catalog.register_decision_model(args.source)
    print(json.dumps(ref.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

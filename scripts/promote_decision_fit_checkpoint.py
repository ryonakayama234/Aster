"""Promote a verified decision_fit checkpoint and register it for ACT v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from aster.model.decision_promotion import promote_decision_fit_checkpoint
from aster.service.artifacts import ArtifactCatalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    catalog = ArtifactCatalog(root)
    staging_base = root / "artifacts"
    staging_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".decision-promotion-", dir=staging_base) as temporary:
        promoted = Path(temporary) / "model-artifact"
        manifest = promote_decision_fit_checkpoint(args.source, promoted)
        ref = catalog.register_decision_model(promoted)

    print(
        json.dumps(
            {
                "source_checkpoint_id": manifest["source_checkpoint_id"],
                **ref.to_dict(),
                "calibration_status": "not_run",
                "routing_status": "not_configured",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

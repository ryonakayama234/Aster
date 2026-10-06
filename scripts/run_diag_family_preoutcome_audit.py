#!/usr/bin/env python3
"""Materialize DIAG v3 pre-outcome confounding/collinearity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.training.diag_family_preoutcome import run_preoutcome_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--feature-run-id", required=True)
    args = parser.parse_args()

    run_path = run_preoutcome_audit(args.root, args.feature_run_id)
    result = json.loads(
        (run_path / "diag-family-preoutcome-audit.json").read_text(encoding="utf-8")
    )
    populations = result["populations"]
    design_order = result["design_order"]
    summary = {
        "status": result["status"],
        "source_feature_run_id": result["source_feature_run_id"],
        "source_features_sha256": result["source_features_sha256"],
        "feature_schema_sha256": result["feature_schema_sha256"],
        "families": result["families"],
        "outcome_joined": result["outcome_joined"],
        "continuous_screening_columns": populations["all30"][
            "continuous_screening_columns"
        ],
        "collinearity_cluster_counts": {
            population: len(values["collinearity_clusters"])
            for population, values in populations.items()
        },
        "design_order_confounded_columns": sorted(
            name
            for name, values in design_order.items()
            if values["design_order_confounded"]
        ),
        "next_gate": result["next_gate"],
    }
    print(run_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

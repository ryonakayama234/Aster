#!/usr/bin/env python3
"""Run DIAG v3 Gate 2A frozen outcome join and 30-family screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.training.diag_family_outcome_screening import run_outcome_screening


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preoutcome-run-id", required=True)
    parser.add_argument("--learn-evidence-root", type=Path, default=None)
    parser.add_argument(
        "--learn-evidence-mode",
        choices=("original", "reproduction"),
        default="original",
    )
    args = parser.parse_args()

    run_path = run_outcome_screening(
        args.root,
        preoutcome_run_id=args.preoutcome_run_id,
        learn_evidence_root=args.learn_evidence_root,
        learn_evidence_mode=args.learn_evidence_mode,
    )
    result = json.loads(
        (run_path / "diag-family-outcome-screening.json").read_text(encoding="utf-8")
    )
    continuous = result["continuous_screening"]
    lead_columns = sorted(
        name
        for name, values in continuous.items()
        if values["lead"]["lead_kind"] is not None
    )
    unconfounded_groups = sorted(
        {
            values["feature_group"]
            for values in continuous.values()
            if values["lead"]["qualifies_unconfounded_group"]
        }
    )
    summary = {
        "status": result["status"],
        "source_preoutcome_run_id": result["source_preoutcome_run_id"],
        "source_feature_run_id": result["source_feature_run_id"],
        "learn_evidence_mode": result["learn_evidence_source"]["mode"],
        "learn_evidence_campaign_run_id": result["learn_evidence_campaign_run_id"],
        "families": result["families"],
        "outcome_joined": result["outcome_joined"],
        "diag_v2_overlay_joined": result["diag_v2_overlay_joined"],
        "classification": result["classification"],
        "continuous_screening_columns": len(continuous),
        "categorical_screening_columns": len(result["categorical_screening"]),
        "lead_columns": lead_columns,
        "unconfounded_lead_groups": unconfounded_groups,
        "inferential_statistics_emitted": result["inferential_statistics_emitted"],
        "next_gate": result["next_gate"],
    }
    print(run_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

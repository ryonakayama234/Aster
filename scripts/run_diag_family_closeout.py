#!/usr/bin/env python3
"""Close DIAG v3 with the frozen DIAG v2 four-family overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from aster.training.diag_family_closeout import run_feature_audit_closeout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--outcome-screening-run-id",
        required=True,
        help="Completed Gate 2A diag_family_outcome_screening Run ID",
    )
    parser.add_argument(
        "--no-write-reports",
        action="store_true",
        help="Keep final evidence only under runs/ instead of also writing reports/",
    )
    args = parser.parse_args()

    run_path = run_feature_audit_closeout(
        args.root,
        outcome_screening_run_id=args.outcome_screening_run_id,
        write_reports=not args.no_write_reports,
    )
    result = json.loads(
        (run_path / "diag-family-feature-audit.json").read_text(encoding="utf-8")
    )
    inputs = cast(dict[str, object], result["classification_inputs"])
    pairs = cast(list[dict[str, object]], result["pair_inspection"])
    summary = {
        "status": result["status"],
        "classification": result["classification"],
        "families": result["families"],
        "outcome_joined": result["outcome_joined"],
        "diag_v2_overlay_joined": result["diag_v2_overlay_joined"],
        "causal_claim": result["causal_claim"],
        "confirmatory_verdict": result["confirmatory_verdict"],
        "inferential_statistics_emitted": result["inferential_statistics_emitted"],
        "source_outcome_screening_run_id": result["source_outcome_screening_run_id"],
        "source_diag_v2_run_id": result["source_diag_v2_run_id"],
        "lead_columns": inputs["lead_columns"],
        "qualifying_unconfounded_groups": inputs["qualifying_unconfounded_groups"],
        "pair_ids": [
            [pair["loss_family_id"], pair["control_family_id"]]
            for pair in pairs
        ],
        "next_causal_question": result["next_causal_question"],
        "stop_rule_satisfied": result["stop_rule_satisfied"],
        "new_training_arm_introduced": result["new_training_arm_introduced"],
        "reports_written": not args.no_write_reports,
    }
    print(run_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

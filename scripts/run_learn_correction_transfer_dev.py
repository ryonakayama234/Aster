"""Run the fixed LEARN-v0 correction-transfer development probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aster.training.learn_probe import DEFAULT_ACT_RUN_ID, run_learn_dev_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--act-run-id", default=DEFAULT_ACT_RUN_ID)
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_path = run_learn_dev_probe(
        args.root,
        act_run_id=args.act_run_id,
    )
    experiment = json.loads(
        (run_path / "correction-transfer.json").read_text(encoding="utf-8")
    )
    local_transfer = {}
    for arm in ("parent", "replay", "correction"):
        benchmark = json.loads(
            (run_path / f"benchmark-{arm}" / "benchmark.json").read_text(encoding="utf-8")
        )
        raw = benchmark["test"]["raw"]
        local_transfer[arm] = {
            "accuracy": raw["accuracy"],
            "nll": raw["nll"],
        }

    sequential = experiment.get("sequential_transfer")
    sequential_summary = None
    if isinstance(sequential, dict):
        arms = sequential.get("arms")
        if isinstance(arms, dict):
            sequential_summary = {}
            for arm, value in arms.items():
                if not isinstance(value, dict):
                    continue
                summary = value.get("summary")
                if not isinstance(summary, dict):
                    continue
                evaluation = summary.get("evaluation")
                if not isinstance(evaluation, dict):
                    continue
                sequential_summary[arm] = {
                    "run_id": value.get("run_id"),
                    "task_success": evaluation.get("task_success"),
                    "goal_verified": evaluation.get("goal_verified"),
                    "steps": evaluation.get("steps"),
                    "stop_reason": evaluation.get("stop_reason"),
                    "teacher_disagreements": summary.get("teacher_disagreements"),
                    "first_teacher_divergence_step": summary.get(
                        "first_teacher_divergence_step"
                    ),
                    "first_tool_failure_step": summary.get("first_tool_failure_step"),
                }

    compact = {
        "run_id": run_path.name,
        "repair": experiment["repair"],
        "local_transfer_raw": local_transfer,
        "candidate_artifacts": experiment["candidate_artifacts"],
        "sequential_transfer": sequential_summary,
    }
    print(run_path)
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

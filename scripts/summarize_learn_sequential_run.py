"""Summarize model-only Sequential Transfer evidence from one completed LEARN run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path, run_id: str) -> dict[str, object]:
    run_path = root / "runs" / run_id
    experiment = json.loads(
        (run_path / "correction-transfer.json").read_text(encoding="utf-8")
    )
    sequential = experiment.get("sequential_transfer")
    if not isinstance(sequential, dict):
        raise ValueError("Run has no sequential_transfer evidence")
    arms = sequential.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("Sequential evidence has no arms")

    result: dict[str, object] = {
        "run_id": run_id,
        "scope": sequential.get("scope"),
        "arms": {},
    }
    output_arms: dict[str, object] = {}
    for arm in ("parent", "replay", "correction"):
        data = arms.get(arm)
        if not isinstance(data, dict):
            continue
        child_id = data.get("run_id")
        if not isinstance(child_id, str):
            raise ValueError(f"Sequential arm {arm} has no run_id")
        child = root / "runs" / child_id
        summary = json.loads((child / "sequential.json").read_text(encoding="utf-8"))
        trajectory = [
            json.loads(line)
            for line in (child / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        teacher = [
            json.loads(line)
            for line in (child / "teacher-comparison.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(trajectory) != len(teacher):
            raise ValueError(f"Sequential arm {arm} evidence length mismatch")

        steps = []
        for transition, comparison in zip(trajectory, teacher, strict=True):
            action = transition["action"]
            teacher_action = comparison["teacher_action"]
            steps.append(
                {
                    "step": transition["step"],
                    "student": {
                        "kind": action["kind"],
                        "name": action["name"],
                        "arguments": action["arguments"],
                    },
                    "teacher": {
                        "kind": teacher_action["kind"],
                        "name": teacher_action["name"],
                        "arguments": teacher_action["arguments"],
                    },
                    "agrees": comparison["agrees"],
                    "execution_ok": transition["observation"]["ok"],
                    "goal_satisfied": transition["evaluation"]["goal_satisfied"],
                    "terminal": transition["evaluation"]["terminal"],
                }
            )
        output_arms[arm] = {
            "run_id": child_id,
            "evaluation": summary["evaluation"],
            "first_teacher_divergence_step": summary["first_teacher_divergence_step"],
            "first_tool_failure_step": summary["first_tool_failure_step"],
            "final_memory": summary["final_memory"],
            "steps": steps,
        }
    result["arms"] = output_arms
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.run_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixed, observable Agent recipe used by the local Aster Service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from aster.agent.policy import RuleBasedPolicy
from aster.evaluator.episode import write_episode_evaluation
from aster.evaluator.verifier import TaskEvaluator
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import JsonValue
from aster.runtime.context import RuntimeContext
from aster.runtime.loop import run_loop
from aster.tools.builtin.calculator import CALCULATOR_SPEC, calculator
from aster.tools.builtin.memory import MEMORY_GET_SPEC, MEMORY_PUT_SPEC, memory_get, memory_put
from aster.tools.executor import ToolExecutor
from aster.tools.registry import ToolRegistry

RECIPE_ID = "agent-calculate-store-v0"
POLICY_ID = "rule-calculate-store-v0"
TASK: dict[str, JsonValue] = {
    "kind": "calculate_and_store",
    "operation": "add",
    "left": 40,
    "right": 2,
    "store_as": "total",
}


def build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC, calculator)
    registry.register(MEMORY_PUT_SPEC, memory_put)
    registry.register(MEMORY_GET_SPEC, memory_get)
    return ToolExecutor(registry)


def run_calculate_store_recipe(root: str | Path, *, max_steps: int = 8) -> Path:
    """Run one fixed safe Agent episode and persist replayable evidence."""
    root = Path(root)
    context = RuntimeContext(task=TASK)
    run = RunLog(
        root,
        "agent",
        {
            "recipe_id": RECIPE_ID,
            "policy_id": POLICY_ID,
            "task": dict(TASK),
            "max_steps": max_steps,
        },
        producer="runtime",
    )
    recorder = TrajectoryRecorder()

    try:
        run_loop(
            policy=RuleBasedPolicy(),
            executor=build_executor(),
            evaluator=TaskEvaluator(),
            context=context,
            recorder=recorder,
            max_steps=max_steps,
        )
        trajectory_path = run.path / "trajectory.jsonl"
        recorder.write_jsonl(trajectory_path)
        evaluation_path = run.path / "evaluation.json"
        episode_evaluation = write_episode_evaluation(evaluation_path, trajectory_path)

        trajectory_payload = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        agent_bundle = {
            "schema_version": "aster-agent-bundle-0",
            "recipe_id": RECIPE_ID,
            "policy_id": POLICY_ID,
            "task": dict(TASK),
            "trajectory_file": "trajectory.jsonl",
            "evaluation_file": "evaluation.json",
            "trajectory": trajectory_payload,
            "evaluation": evaluation_payload,
            "final_memory": dict(context.memory),
            "summary": episode_evaluation.to_dict(),
        }
        _write_json(run.path / "agent-bundle.json", agent_bundle)

        run.event("episode_evaluated", {"evaluation_file": "evaluation.json"})
        run.event("rolled_out", episode_evaluation.to_dict())
        run.finish(
            "completed",
            agent="agent-bundle.json",
            trajectory="trajectory.jsonl",
            evaluation="evaluation.json",
        )
        return run.path
    except BaseException as error:
        run.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_path = run_calculate_store_recipe(args.root)
    print(run_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

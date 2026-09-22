"""Append-only collection and JSONL serialization of agent experience."""

import json
from pathlib import Path

from aster.records.trajectory import Trajectory
from aster.records.transition import Transition


class TrajectoryRecorder:
    def __init__(self):
        self._transitions: list[Transition] = []

    def record(self, transition: Transition) -> None:
        if transition.step != len(self._transitions):
            raise ValueError("Transition steps must be contiguous and start at zero")
        self._transitions.append(transition)

    def trajectory(self) -> Trajectory:
        return Trajectory(tuple(self._transitions))

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for transition in self._transitions:
                stream.write(json.dumps(transition.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def read_jsonl(path: Path) -> Trajectory:
        transitions: list[Transition] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    transitions.append(Transition.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid trajectory JSONL at line {line_number}") from exc
        for expected_step, transition in enumerate(transitions):
            if transition.step != expected_step:
                raise ValueError("Trajectory steps must be contiguous and start at zero")
        return Trajectory(tuple(transitions))

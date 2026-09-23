"""Versioned benchmark cases kept separate from training and execution records."""

from dataclasses import dataclass

from aster.records.decision import DecisionExample

_SPLITS = frozenset({"train", "calibration", "dev", "test"})


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One decision example with explicit split, slice, and leakage identity."""

    case_id: str
    split: str
    slice_name: str
    leakage_group: str
    example: DecisionExample

    def __post_init__(self):
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if self.split not in _SPLITS:
            raise ValueError(f"Unsupported benchmark split: {self.split}")
        if not self.slice_name:
            raise ValueError("slice_name must not be empty")
        if not self.leakage_group:
            raise ValueError("leakage_group must not be empty")

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-decision-benchmark-case-0",
            "case_id": self.case_id,
            "split": self.split,
            "slice": self.slice_name,
            "leakage_group": self.leakage_group,
            "example": self.example.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """A deterministic set of cases with split-leakage checks."""

    suite_id: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self):
        if not self.suite_id:
            raise ValueError("suite_id must not be empty")
        if not self.cases:
            raise ValueError("Benchmark suite requires at least one case")

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case IDs must be unique")

        group_splits: dict[str, set[str]] = {}
        for case in self.cases:
            group_splits.setdefault(case.leakage_group, set()).add(case.split)
        leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        if leaked:
            raise ValueError(
                "Leakage groups must not cross benchmark splits: " + ", ".join(leaked)
            )

    def cases_for(self, split: str) -> tuple[BenchmarkCase, ...]:
        if split not in _SPLITS:
            raise ValueError(f"Unsupported benchmark split: {split}")
        return tuple(case for case in self.cases if case.split == split)

    def training_examples(self) -> tuple[DecisionExample, ...]:
        return tuple(case.example for case in self.cases_for("train"))

    def to_dict(self) -> dict:
        return {
            "schema_version": "aster-decision-benchmark-suite-0",
            "suite_id": self.suite_id,
            "cases": [case.to_dict() for case in self.cases],
        }

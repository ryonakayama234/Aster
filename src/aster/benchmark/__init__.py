"""Generalization and calibration benchmarks for learned Aster policies."""

from aster.benchmark.case import BenchmarkCase, BenchmarkSuite
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aster.benchmark.runner import (
        BenchmarkBundle, run_decision_benchmark, run_logged_decision_benchmark,
        write_benchmark_artifacts,
    )
    from aster.benchmark.suite import build_calculate_and_store_suite


def __getattr__(name: str):
    # Schema/preflight imports should not load the optional training dependency.
    module = "suite" if name == "build_calculate_and_store_suite" else "runner"
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(f"aster.benchmark.{module}"), name)
    globals()[name] = value
    return value

__all__ = [
    "BenchmarkBundle",
    "BenchmarkCase",
    "BenchmarkSuite",
    "build_calculate_and_store_suite",
    "run_decision_benchmark",
    "run_logged_decision_benchmark",
    "write_benchmark_artifacts",
]

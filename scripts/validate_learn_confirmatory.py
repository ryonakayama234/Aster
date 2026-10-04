#!/usr/bin/env python3
"""Validate the frozen LEARN-v0 confirmatory protocol without running training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.training.learn_confirmatory import (
    DEFAULT_MANIFEST_PATH,
    EXPECTED_MANIFEST_SHA256,
    confirmatory_manifest_sha256,
    load_confirmatory_manifest,
    validate_confirmatory_candidate_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Aster repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = load_confirmatory_manifest(root / DEFAULT_MANIFEST_PATH)
    sha256 = confirmatory_manifest_sha256(manifest)
    if sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Frozen confirmatory manifest digest mismatch: {sha256}"
        )
    coverage = validate_confirmatory_candidate_coverage(manifest)

    design = manifest["family_design"]
    training = manifest["training"]
    endpoints = manifest["endpoints"]
    if not isinstance(design, dict) or not isinstance(training, dict) or not isinstance(endpoints, dict):
        raise ValueError("Validated manifest has an unexpected structure")
    primary = endpoints["primary"]
    if not isinstance(primary, dict):
        raise ValueError("Validated manifest primary endpoint is invalid")

    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "manifest_sha256": sha256,
                "families": design["planned_families"],
                "seeds": training["seeds"],
                "episode_horizon": training["max_episode_steps"],
                "primary_endpoint": primary["metric"],
                "candidate_coverage": coverage,
                "measurement_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Safely ingest the private LANG v0 Drive snapshot into the local raw pool.

The source directory may be flat or nested. Files are discovered recursively by
Drive-style basenames such as prose001.txt and dialogue015.txt. Private text is
copied only after the whole source set passes the public active-manifest checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("configs/lang-v0-corpus.json")
SOURCE_NAME = re.compile(r"^(prose|dialogue)(\d{3})\.txt$")


def _source_name(pool_path: str) -> str:
    path = PurePosixPath(pool_path)
    parts = path.parts
    if (
        len(parts) != 3
        or parts[0] not in {"prose", "dialogue"}
        or parts[1] != "ja"
        or re.fullmatch(r"\d{3}\.txt", parts[2]) is None
    ):
        raise ValueError(f"Unsupported LANG v0 pool path: {pool_path}")
    return f"{parts[0]}{parts[2]}"


def load_contract(root: Path) -> dict[str, dict]:
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, dict] = {}
    total_bytes = 0
    for item in manifest["expected_files"]:
        source_name = _source_name(item["path"])
        if source_name in expected:
            raise ValueError(f"Duplicate source basename in manifest: {source_name}")
        expected[source_name] = item
        total_bytes += int(item["bytes"])

    declared_count = manifest.get("expected_file_count")
    if declared_count is not None and int(declared_count) != len(expected):
        raise ValueError("Manifest expected_file_count does not match expected_files")
    declared_bytes = manifest.get("expected_total_bytes")
    if declared_bytes is not None and int(declared_bytes) != total_bytes:
        raise ValueError("Manifest expected_total_bytes does not match expected_files")
    return expected


def discover_source_files(source_dir: Path) -> dict[str, Path]:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    found: dict[str, Path] = {}
    for path in sorted(source_dir.rglob("*.txt")):
        if not path.is_file() or SOURCE_NAME.fullmatch(path.name) is None:
            continue
        if path.name in found:
            raise ValueError(
                f"Duplicate source basename: {path.name}: {found[path.name]} and {path}"
            )
        found[path.name] = path
    return found


def build_plan(source_dir: Path, root: Path) -> tuple[list[dict], int]:
    expected = load_contract(root)
    found = discover_source_files(source_dir)

    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError("Missing LANG v0 source file(s): " + ", ".join(missing))

    extra = sorted(set(found) - set(expected))
    if extra:
        raise ValueError("Unexpected LANG v0 source file(s): " + ", ".join(extra))

    plan: list[dict] = []
    total_bytes = 0
    for source_name in sorted(expected):
        item = expected[source_name]
        source = found[source_name]
        data = source.read_bytes()
        expected_bytes = int(item["bytes"])
        if len(data) != expected_bytes:
            raise ValueError(
                f"Source byte mismatch: {source_name}: expected {expected_bytes}, got {len(data)}"
            )

        destination = root / "data/raw/pool" / item["path"]
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink destination: {destination}")

        state = "copy"
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Destination is not a regular file: {destination}")
            if destination.read_bytes() != data:
                raise ValueError(
                    f"Destination already exists with different bytes: {item['path']}"
                )
            state = "unchanged"

        plan.append(
            {
                "source_name": source_name,
                "source": source,
                "destination": destination,
                "pool_path": item["path"],
                "bytes": len(data),
                "data": data,
                "state": state,
            }
        )
        total_bytes += len(data)
    return plan, total_bytes


def _atomic_write(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".ingest-tmp", dir=destination.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def ingest(
    source_dir: Path,
    *,
    root: Path = ROOT,
    run_inventory: bool = True,
    dry_run: bool = False,
) -> dict:
    root = root.resolve()
    plan, total_bytes = build_plan(source_dir, root)

    copied = 0
    unchanged = 0
    if not dry_run:
        for item in plan:
            if item["state"] == "unchanged":
                unchanged += 1
                continue
            _atomic_write(item["destination"], item["data"])
            copied += 1

        if run_inventory:
            subprocess.run(
                [sys.executable, str(root / "scripts/inventory_corpus.py")],
                cwd=root,
                check=True,
            )
    else:
        copied = sum(item["state"] == "copy" for item in plan)
        unchanged = sum(item["state"] == "unchanged" for item in plan)

    return {
        "files": len(plan),
        "bytes": total_bytes,
        "copied": copied,
        "unchanged": unchanged,
        "dry_run": dry_run,
        "inventory_ran": bool(run_inventory and not dry_run),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing Drive-downloaded proseNNN.txt/dialogueNNN.txt files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the full source set and destination conflicts without writing",
    )
    parser.add_argument(
        "--no-inventory",
        action="store_true",
        help="Copy verified files but do not run scripts/inventory_corpus.py afterwards",
    )
    args = parser.parse_args()
    result = ingest(
        args.source_dir,
        run_inventory=not args.no_inventory,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

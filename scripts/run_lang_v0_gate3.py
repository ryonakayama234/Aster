"""Gate 3A preflight or read-only evaluation of an *existing* frozen pilot Run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Public WSL entrypoint is the trust root: lock module -> spec -> evaluator/sources.
# Check lock bytes before importing any Aster package code.
EXPECTED_LOCK_GIT_BLOB_SHA = '53068dfbeedfba1df84c2fb886724a1016b03392'
SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'src'
lock_bytes = (SOURCE_ROOT / 'aster/training/lang_gate3_lock.py').read_bytes()
lock_header = b'blob ' + str(len(lock_bytes)).encode() + b'\x00'
if hashlib.sha1(lock_header + lock_bytes).hexdigest() != EXPECTED_LOCK_GIT_BLOB_SHA:
    raise RuntimeError('Gate 3 frozen spec lock module changed')

# Match the standalone TinyLM training script when run from the WSL checkout.
sys.path.insert(0, str(SOURCE_ROOT))

from aster.training.lang_gate3 import preflight, report_existing_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--spec', type=Path, default=Path('configs/lang-v0-gate3.json'))
    parser.add_argument('--run', type=Path, help='Completed pilot Run path; omitted means preflight only')
    args = parser.parse_args()
    spec = args.spec if args.spec.is_absolute() else args.root / args.spec
    if args.run is None:
        _, _, _, audit = preflight(args.root, spec)
        print(json.dumps({'status': 'preflight_passed', **audit}, ensure_ascii=False, indent=2))
    else:
        path = args.run if args.run.is_absolute() else args.root / args.run
        print(report_existing_run(args.root, spec, path))


if __name__ == '__main__':
    main()

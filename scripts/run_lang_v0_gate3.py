"""Gate 3A preflight or read-only evaluation of an *existing* frozen pilot Run."""

import argparse
import json
from pathlib import Path

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

"""Copy two matching private replay bundles into one convenient local folder."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-run', required=True)
    parser.add_argument('--tokenizer-run', required=True)
    args = parser.parse_args()
    if any(len(s) != 32 or any(c not in '0123456789abcdef' for c in s) for s in [args.corpus_run,args.tokenizer_run]):
        parser.error('Use exact run IDs')
    corpus = (ROOT/'runs'/args.corpus_run/'observation-bundle.json').read_bytes()
    tokenizer = (ROOT/'runs'/args.tokenizer_run/'tokenizer-bundle.json').read_bytes()
    cid = json.loads(corpus)['view_id']
    if cid != json.loads(tokenizer)['evaluation']['experiment']['view_id']:
        raise ValueError('Bundle view IDs differ')
    dest = ROOT/'artifacts/workbench-inputs'/cid
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in [('observation-bundle.json',corpus),('tokenizer-bundle.json',tokenizer)]:
        target = dest/name
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Existing bundle differs')
        target.write_bytes(content)
    print(dest)


if __name__ == '__main__':
    main()

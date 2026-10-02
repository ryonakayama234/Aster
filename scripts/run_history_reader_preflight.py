"""Execute and independently audit a history-to-fact dataset without torch."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from aster.benchmark.history_reader import run_preflight

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard'], cwd=repo, text=True).splitlines()
    source = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()),
                  python_file_sha256={p: hashlib.sha256((repo/p).read_bytes()).hexdigest()
                                      for p in sorted(set(paths)) if p.endswith('.py') and (repo/p).is_file()},
                  protocol_sha256=hashlib.sha256((repo/'docs/HistoryFactReader-v0.md').read_bytes()).hexdigest())
    path = run_preflight(args.root.resolve(), source)
    report = json.loads((path/'report.json').read_text(encoding='utf-8'))
    print(json.dumps({'run_path': str(path), **{k: report[k] for k in ('counts', 'max_byte_tokens', 'suite_sha256')}}, ensure_ascii=False))

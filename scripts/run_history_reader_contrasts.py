"""Register or measure frozen reader contrasts; never train or open reserved tests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from aster.benchmark.history_contrasts import audit_contrasts, build_contrasts
from aster.records.runlog import RunLog


def source_metadata(repo):
    paths = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard'], cwd=repo, text=True).splitlines()
    return dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                git_tree=subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=repo, text=True).strip(),
                dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()),
                python_file_sha256={p: hashlib.sha256((repo/p).read_bytes()).hexdigest()
                                    for p in sorted(set(paths)) if p.endswith('.py') and (repo/p).is_file()},
                protocol_sha256=hashlib.sha256((repo/'docs/HistoryReaderFrozenContrasts-v0.md').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'measure'))
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--previous-run', type=Path)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--expected-suite')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = source_metadata(repo)
    if args.mode == 'register':
        run = RunLog(args.root.resolve(), 'history_reader_contrast_preflight', source, producer='evaluator')
        try:
            rows, pairs = build_contrasts()
            result = audit_contrasts(rows, pairs)
            for name, data in [('registration.json', result), ('cases.json', rows), ('pairs.json', pairs)]:
                (run.path/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            if audit_contrasts(json.loads((run.path/'cases.json').read_text()),
                               json.loads((run.path/'pairs.json').read_text())) != result:
                raise ValueError('Saved contrast registration mismatch')
            run.finish('completed', registration='registration.json')
        except BaseException as error:
            run.finish('failed', error=str(error))
            raise
        print(run.path, flush=True)
    else:
        if args.previous_run is None or args.registration is None or args.expected_suite is None:
            parser.error('measure requires --previous-run, --registration, --expected-suite')
        registered = json.loads(args.registration.read_text())
        if registered['suite_sha256'] != args.expected_suite:
            raise ValueError('Explicit contrast suite digest mismatch')
        if source['dirty']:
            raise ValueError('Formal contrast measurement requires a clean source commit')
        import torch
        from aster.training.history_contrasts import run_diagnostic
        torch.set_num_threads(2)
        print(run_diagnostic(args.root.resolve(), args.previous_run.resolve(), registered, source), flush=True)

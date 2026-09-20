#!/usr/bin/env python3
"""Aster Workbench bridge v0.1. Run with the Aster virtualenv, --root REPO.
Only fixed BPE evaluation jobs are accepted. No shell commands or uploads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

ORIGIN = 'https://aster-learning-lab-zhong.rynaka0112.chatgpt.site'

def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

class Jobs:
    def __init__(self, root):
        self.root = root.resolve()
        self.base = self.root / 'runs' / 'workbench'
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # Interrupted bridge sessions must never appear as active forever.
        for p in self.base.glob('*/job.json'):
            j = json.loads(p.read_text())
            if j['status'] == 'running':
                j.update(status='interrupted', ended_at=now(), error='接続口の終了により監視が中断しました。再実行してください。')
                write_json(p, j)

    def views(self):
        base = self.root / 'data' / 'training'
        return sorted(p.name for p in base.glob('*') if re.fullmatch(r'[0-9a-f]{64}', p.name) and p.is_dir() and not p.is_symlink() and (p / 'manifest.json').is_file())

    def start(self, view, vocab):
        if view not in self.views() or type(vocab) is not int or vocab not in (256, 512, 1024):
            raise ValueError('学習用本文の版、または語彙数が不正です。')
        with self.lock:
            if any(json.loads(p.read_text())['status'] == 'running' for p in self.base.glob('*/job.json')):
                raise ValueError('BPE処理は一度に一つだけ実行できます。')
            jid = uuid4().hex
            folder = self.base / jid
            folder.mkdir()
            job = dict(run_id=jid, status='running', kind='tokenizer', view_id=view, vocab_size=vocab, started_at=now())
            write_json(folder / 'job.json', job)
        threading.Thread(target=self.work, args=(folder, job), daemon=True).start()
        return jid

    def work(self, folder, job):
        env = dict(os.environ, PYTHONPATH=str(self.root / 'src'))
        command = [sys.executable, '-m', 'aster.training.tokenizer_run', '--root', str(folder), '--view', str(self.root / 'data/training' / job['view_id']), '--vocab-size', str(job['vocab_size'])]
        try:
            with (folder / 'process.log').open('w') as log:
                result = subprocess.run(command, cwd=self.root, env=env, stdout=log, stderr=log, timeout=1800)
            if result.returncode:
                raise RuntimeError((folder / 'process.log').read_text()[-3000:])
            paths = list((folder / 'runs').glob('*/run.json'))
            if len(paths) != 1 or json.loads(paths[0].read_text())['status'] != 'completed':
                raise RuntimeError('完了した実行記録を確認できません。')
            job['status'] = 'completed'
        except Exception as e:
            job.update(status='failed', error=str(e))
        job['ended_at'] = now()
        write_json(folder / 'job.json', job)
        write_json(folder / 'run-bundle.json', self.read(job['run_id']))

    def read(self, jid):
        if not re.fullmatch(r'[0-9a-f]{32}', jid):
            raise ValueError('実行IDが不正です。')
        folder = self.base / jid
        job = json.loads((folder / 'job.json').read_text())
        events = []
        tokenizer = None
        for p in sorted((folder / 'runs').glob('*/events.jsonl')):
            for line in p.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A concurrently appended final line is read on next poll.
            bp = p.parent / 'tokenizer-bundle.json'
            if bp.exists() and job['status'] == 'completed':
                tokenizer = json.loads(bp.read_text())
        return dict(schema_version='aster-workbench-run-0', job=job, events=events, tokenizer=tokenizer)

    def summaries(self):
        return sorted([json.loads(p.read_text()) for p in self.base.glob('*/job.json')], key=lambda j:j['started_at'], reverse=True)[:50]

def make_server(jobs, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed(self, auth=True):
            host = self.headers.get('Host')
            origin = self.headers.get('Origin')
            return host in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}') and origin in (None, ORIGIN) and (not auth or secrets.compare_digest(self.headers.get('Authorization',''), 'Bearer ' + token))

        def respond(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            if self.headers.get('Origin') == ORIGIN:
                self.send_header('Access-Control-Allow-Origin', ORIGIN)
                self.send_header('Vary', 'Origin')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            if not self.allowed(False):
                return self.respond(403, {'error':'Origin/Host denied'})
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', ORIGIN)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
            self.end_headers()

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, {'error':'接続キーまたは接続元を確認してください。'})
            try:
                if self.path == '/status':
                    data = dict(version='aster-bridge-0.1', views=jobs.views(), jobs=jobs.summaries())
                elif self.path.startswith('/jobs/'):
                    data = jobs.read(self.path.removeprefix('/jobs/'))
                else:
                    return self.respond(404, {'error':'Not found'})
                self.respond(200, data)
            except (ValueError, OSError) as e:
                self.respond(400, {'error':str(e)})

        def do_POST(self):
            if not self.allowed():
                return self.respond(403, {'error':'接続キーまたは接続元を確認してください。'})
            try:
                if self.path != '/jobs':
                    return self.respond(404, {'error':'Not found'})
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 1024:
                    raise ValueError('Request too large or empty')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {'view_id','vocab_size'}:
                    raise ValueError('Only view_id and vocab_size are accepted')
                jid = jobs.start(body['view_id'], body['vocab_size'])
                self.respond(202, {'run_id':jid})
            except (ValueError, OSError, TypeError) as e:
                self.respond(400, {'error':str(e)})
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--once', metavar='VIEW_ID', help='Run without a browser; export run-bundle.json')
    parser.add_argument('--vocab-size', type=int, default=512)
    args = parser.parse_args()
    # One bridge per repository. Avoid simultaneous mutation/recovery of job records.
    import fcntl
    args.root = args.root.resolve()
    if not (args.root / 'src/aster/training/tokenizer_run.py').is_file():
        parser.error('Asterリポジトリのルートで実行してください。')
    lockfile = args.root / 'runs/workbench.lock'
    lockfile.parent.mkdir(exist_ok=True)
    with lockfile.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('接続口はすでに起動しています。')
        jobs = Jobs(args.root)
        if args.once:
            jid = jobs.start(args.once, args.vocab_size)
            import time
            while jobs.read(jid)['job']['status'] == 'running':
                time.sleep(.2)
            path = jobs.base / jid / 'run-bundle.json'
            while not path.exists():
                time.sleep(.05)
            bundle = json.loads(path.read_text())
            print(path)
            return 0 if bundle['job']['status']=='completed' else 1
        token = secrets.token_urlsafe(32)
        server = make_server(jobs, token, args.port)
        print(f'接続先: http://127.0.0.1:{server.server_port}\n接続キー: {token}\n終了: Ctrl+C', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0

if __name__ == '__main__':
    sys.exit(main())

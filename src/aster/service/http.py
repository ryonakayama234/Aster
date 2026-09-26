"""Local HTTP transport for the Aster service contract."""

from __future__ import annotations

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import time
from urllib.parse import urlsplit

from aster.service.contracts import JobSpec
from aster.service.jobs import JobBusyError, JobManager

ORIGIN = "https://aster-learning-lab-zhong.rynaka0112.chatgpt.site"


def make_server(jobs: JobManager, token: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def allowed(self, auth: bool = True) -> bool:
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            server_port = int(getattr(self.server, "server_port"))
            valid_host = host in (
                f"127.0.0.1:{server_port}",
                f"localhost:{server_port}",
            )
            valid_origin = origin in (None, ORIGIN)
            valid_auth = not auth or secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            )
            return valid_host and valid_origin and valid_auth

        def respond(self, status: int, data: object) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if self.headers.get("Origin") == ORIGIN:
                self.send_header("Access-Control-Allow-Origin", ORIGIN)
                self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            if not self.allowed(False):
                self.respond(403, {"error": "Origin/Host denied"})
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", ORIGIN)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_GET(self) -> None:
            if not self.allowed():
                self.respond(403, {"error": "接続キーまたは接続元を確認してください。"})
                return
            try:
                path = urlsplit(self.path).path
                if path == "/status":
                    data = jobs.status()
                elif path == "/capabilities":
                    status = jobs.status()
                    data = {
                        "schema_version": "aster-capabilities-0",
                        "capabilities": status["capabilities"],
                    }
                elif path == "/recipes":
                    data = {"schema_version": "aster-recipes-0", "recipes": jobs.recipes.list()}
                elif path == "/artifacts":
                    data = {"schema_version": "aster-artifacts-0", "artifacts": jobs.catalog.all()}
                elif path.startswith("/jobs/"):
                    data = jobs.read(path.removeprefix("/jobs/"))
                else:
                    self.respond(404, {"error": "Not found"})
                    return
                self.respond(200, data)
            except (ValueError, OSError) as error:
                self.respond(400, {"error": str(error)})

        def do_POST(self) -> None:
            if not self.allowed():
                self.respond(403, {"error": "接続キーまたは接続元を確認してください。"})
                return
            try:
                if urlsplit(self.path).path != "/jobs":
                    self.respond(404, {"error": "Not found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 4096:
                    raise ValueError("Request too large or empty")
                body = json.loads(self.rfile.read(length))
                legacy = isinstance(body, dict) and set(body) == {"view_id", "vocab_size"}
                if legacy:
                    spec = jobs.legacy_tokenizer_spec(body["view_id"], body["vocab_size"])
                else:
                    spec = JobSpec.from_dict(body)
                job_id = jobs.start(spec)
                response = {"job_id": job_id}
                if legacy:
                    response["run_id"] = job_id
                self.respond(202, response)
            except JobBusyError as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, OSError, TypeError, json.JSONDecodeError) as error:
                self.respond(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--once", metavar="VIEW_ID", help="Legacy one-shot tokenizer job")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not (args.root / "src" / "aster" / "training" / "tokenizer_run.py").is_file():
        parser.error("Asterリポジトリのルートで実行してください。")

    lockfile = args.root / "runs" / "workbench.lock"
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with lockfile.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("接続口はすでに起動しています。")
        jobs = JobManager(args.root)
        if args.once:
            job_id = jobs.start(jobs.legacy_tokenizer_spec(args.once, args.vocab_size))
            while True:
                current = jobs.read(job_id).get("job")
                if not isinstance(current, dict) or current.get("status") not in ("accepted", "running"):
                    break
                time.sleep(0.2)
            path = jobs.base / job_id / "run-bundle.json"
            while not path.exists():
                time.sleep(0.05)
            bundle = json.loads(path.read_text(encoding="utf-8"))
            job = bundle.get("job") if isinstance(bundle, dict) else None
            print(path)
            return 0 if isinstance(job, dict) and job.get("status") == "completed" else 1

        token = secrets.token_urlsafe(32)
        server = make_server(jobs, token, args.port)
        print(
            f"接続先: http://127.0.0.1:{server.server_port}\n接続キー: {token}\n終了: Ctrl+C",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

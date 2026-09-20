"""Download a small, source-pinned raw inventory; never feed it to training directly."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/corpus/external-seed-manifest.json"
DEST = ROOT / "data/raw/pool/external-seed"
SOURCES = [
    ("gsm8k", "openai/grade-school-math", "master", [
        "LICENSE", "README.md", "grade_school_math/data/train.jsonl",
    ]),
    ("python-tutorial", "python/cpython", "v3.13.0", [
        "LICENSE", "Doc/license.rst", "Doc/copyright.rst",
        "Doc/tutorial/introduction.rst", "Doc/tutorial/controlflow.rst",
        "Doc/tutorial/datastructures.rst", "Doc/tutorial/errors.rst",
    ]),
]


def fetch(url):
    request = Request(url, headers={"User-Agent": "AsterCorpus-v0-seed"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main():
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    entries = []
    for name, repo, ref, paths in SOURCES:
        old = next((s for s in previous["sources"] if s["name"] == name), None) if previous else None
        revision = old["revision"] if old else json.loads(
            fetch(f"https://api.github.com/repos/{repo}/commits/{ref}")
        )["sha"]
        files = []
        for path in paths:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
            target = DEST / name / path
            data = target.read_bytes() if target.exists() else fetch(url)
            data.decode("utf-8", errors="strict")
            digest = hashlib.sha256(data).hexdigest()
            if old:
                expected = next(f for f in old["files"] if f["upstream_path"] == path)
                if digest != expected["sha256"]:
                    raise ValueError(f"Hash mismatch: {target}")
            elif target.exists():
                raise FileExistsError(f"Untracked raw file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
            if path.endswith("train.jsonl"):
                records = [json.loads(line) for line in data.decode().splitlines()]
                assert records and all(isinstance(r.get("question"), str) and isinstance(r.get("answer"), str) for r in records)
            files.append({"upstream_path": path, "local_path": target.relative_to(ROOT).as_posix(),
                          "url": url, "sha256": digest, "bytes": len(data)})
        entries.append({"name": name, "repository": repo, "requested_ref": ref,
                        "revision": revision, "files": files})
    result = {"status": "raw-inventory-only", "retrieved_at": previous["retrieved_at"] if previous else datetime.now(timezone.utc).isoformat(), "sources": entries}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(entries), "files": sum(len(s["files"]) for s in entries), "bytes": sum(f["bytes"] for s in entries for f in s["files"]) }))


if __name__ == "__main__":
    main()

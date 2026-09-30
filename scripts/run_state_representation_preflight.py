"""Execute the development preflight; no torch or model checkpoint required."""
import argparse
import subprocess
import hashlib
from pathlib import Path

from aster.benchmark.state_representation import run_preflight, digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    paths = subprocess.run(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=repo, capture_output=True, text=True, check=True).stdout.splitlines()
    hashes = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sorted(set(paths))
              if p.endswith(".py") and (repo / p).is_file()}
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    print(run_preflight(args.root, source_git_sha=sha, source_provenance={
        "dirty": dirty, "python_source_sha256": digest(hashes), "python_file_sha256": hashes}))

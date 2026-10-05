"""Run LANG v0 training-view build plus train-only BPE audit."""

from pathlib import Path

from aster.tokenizer.lang_v0 import run


if __name__ == "__main__":
    artifact, run_path = run(Path.cwd())
    print(f"tokenizer_artifact={artifact}")
    print(f"run={run_path}")

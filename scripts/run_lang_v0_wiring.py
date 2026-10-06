"""Run LANG v0 Gate 2 TinyLM wiring against pinned local artifacts."""

import json
from pathlib import Path

from aster.training.pretrain import TrainConfig, train


ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = json.loads((ROOT / "configs/lang-v0-wiring.json").read_text(encoding="utf-8"))
    if spec.get("schema_version") != "aster-lang-wiring-0":
        raise ValueError("Unsupported LANG wiring schema")

    view = ROOT / "data/training" / spec["training_view_id"]
    tokenizer = ROOT / "artifacts/tokenizers/lang-v0" / spec["tokenizer_artifact_id"]
    config_path = ROOT / spec["tiny_lm_config"]

    if not view.is_dir():
        raise ValueError(f"Missing pinned training view: {view}")
    if not tokenizer.is_dir():
        raise ValueError(f"Missing pinned tokenizer artifact: {tokenizer}")

    config = TrainConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    if config.mode != "overfit" or config.overfit_windows != 1:
        raise ValueError("Gate 2 requires the fixed one-window overfit wiring config")

    run = train(ROOT, view, tokenizer, config)
    print(f"wiring_run={run}")


if __name__ == "__main__":
    main()

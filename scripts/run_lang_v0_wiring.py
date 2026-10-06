"""Run LANG v0 Gate 2 TinyLM wiring against pinned local artifacts."""

import json
from pathlib import Path

from aster.corpus.pipeline import digest
from aster.records.runlog import RunLog
from aster.training.pretrain import TrainConfig, train


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Record the attempt before reading or validating any local input.
    runlog = RunLog(ROOT, "lang_v0_wiring", {
        "wiring_spec_path": "configs/lang-v0-wiring.json",
    }, producer="wiring_runner")
    try:
        spec_bytes = (ROOT / "configs/lang-v0-wiring.json").read_bytes()
        runlog.summary["inputs"]["wiring_spec_sha256"] = digest(spec_bytes)
        runlog.save()
        spec = json.loads(spec_bytes)
        if not isinstance(spec, dict):
            raise ValueError("LANG wiring spec must be an object")
        selected = {name: spec[name] for name in ("training_view_id", "tokenizer_artifact_id")
                    if name in spec}
        runlog.summary["inputs"].update(selected)
        runlog.event("inputs_selected", runlog.summary["inputs"])
        runlog.save()
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

        run = train(ROOT, view, tokenizer, config,
                    tokenizer_artifact_id=spec["tokenizer_artifact_id"],
                    wiring_spec_sha256=digest(spec_bytes))
        runlog.finish("completed", training_run_id=run.name)
        print(f"wiring_run={run}")
    except BaseException as error:
        runlog.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()

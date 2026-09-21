"""Append-only pipeline observations, separate from deterministic artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RunLog:
    def __init__(self, root: Path, kind: str, inputs: dict, *, producer: str = "pipeline"):
        self.id = uuid4().hex
        self.path = root / "runs" / self.id
        self.path.mkdir(parents=True)
        self.seq = 0
        self.producer = producer
        self.summary = {"schema_version": "aster-run-0", "run_id": self.id,
                        "kind": kind, "inputs": inputs, "status": "running"}
        self.event("started", inputs)
        self.save()

    def save(self):
        (self.path / "run.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def event(self, kind, data):
        self.seq += 1
        event = {"schema_version": "aster-event-0", "run_id": self.id, "seq": self.seq,
                 "producer": self.producer, "kind": kind,
                 "time": datetime.now(timezone.utc).isoformat(), "data": data}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finish(self, status, **data):
        self.event(status, data)
        self.summary.update(status=status, last_seq=self.seq, **data)
        self.save()


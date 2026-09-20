"""Canonical candidate records with explicit adapters and immutable build outputs.

No network access, source execution, training selection, or split assignment.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def normalize(data: bytes) -> str:
    text = data.decode("utf-8-sig", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip() or "\x00" in text:
        raise ValueError("empty text or NUL character")
    return text


def parse_conversation(text: str) -> list[dict]:
    """Only exact level-1 Japanese role headings outside fenced code are roles."""
    messages = []
    role = None
    content = []
    start = None
    fence = None
    lines = list(io.StringIO(text))
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\n"))
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not tail.strip():
                fence = None
        next_role = {"# 質問": "user", "# 解答": "assistant", "# 回答": "assistant"}.get(stripped) if fence is None and not marker else None
        if next_role:
            if role is not None:
                value = "".join(content)
                if not value.strip():
                    raise ValueError("empty conversation turn")
                messages.append({"role": role, "content": value,
                                 "source_lines": [start, line_number - 1]})
            elif "".join(content).strip():
                raise ValueError("unlabelled conversation preamble")
            if next_role != ("user" if role in {None, "assistant"} else "assistant"):
                raise ValueError("ambiguous role order")
            role, content, start = next_role, [], line_number + 1
        else:
            content.append(line)
    if fence is not None:
        raise ValueError("unclosed code fence; role boundary is uncertain")
    if role != "assistant" or not "".join(content).strip():
        raise ValueError("conversation must end with a nonempty assistant turn")
    messages.append({"role": role, "content": "".join(content),
                     "source_lines": [start, len(lines)]})
    return messages


def adapt(name: str, text: str):
    """Return (upstream identity, body, group suffix, transform notes) items."""
    if name in {"text_document", "rst_document"}:
        return [("document", {"kind": "document", "text": text,
                               "text_format": "rst" if name == "rst_document" else "plain"},
                 None, ["rst_preserved_not_rendered"] if name == "rst_document" else [])]
    if name == "headed_conversation":
        return [("conversation", {"kind": "conversation", "messages": parse_conversation(text)},
                 None, ["exact_role_headings_removed", "execution_displays_not_promoted_to_tool_messages"])]
    if name == "gsm8k":
        output = []
        for line_number, line in enumerate(io.StringIO(text), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("question", "answer")):
                raise ValueError(f"invalid GSM8K fields at line {line_number}")
            question, answer = item["question"], item["answer"]
            if not re.search(r"(?m)^####\s*\S", answer):
                raise ValueError(f"missing final answer at line {line_number}")
            answer = re.sub(r"<<[^<>]*>>", "", answer)
            output.append((f"line:{line_number}", {"kind": "document", "text_format": "plain",
                           "text": f"Question:\n{question}\n\nAnswer:\n{answer}\n"},
                           "gsm8k-question:" + digest(question.strip().encode()),
                           ["calculation_annotations_removed", "question_answer_template_added"]))
        if not output:
            raise ValueError("empty GSM8K dataset")
        return output
    raise ValueError(f"unknown adapter: {name}")


def build(root: Path, recipe_path: Path | None = None) -> Path:
    root = root.resolve()
    pool = root / "data/raw/pool"
    recipe_path = recipe_path or root / "configs/canonical-v0.json"
    input_paths = {"recipe": recipe_path, "inventory": root / "docs/corpus/pool-inventory-v0.json",
                   "source_notes": root / "docs/corpus/source-notes-v0.json"}
    blobs = {key: path.read_bytes() for key, path in input_paths.items()}
    recipe, inventory = json.loads(blobs["recipe"]), json.loads(blobs["inventory"])
    if recipe["schema_version"] != "aster-canonical-0":
        raise ValueError("unsupported canonical schema")
    known = {"text_document", "rst_document", "headed_conversation", "gsm8k"}
    if not set(recipe["adapters_by_origin"].values()) <= known:
        raise ValueError("unknown recipe adapter")
    if digest(blobs["source_notes"]) != inventory["source_notes_sha256"]:
        raise ValueError("Source notes changed: regenerate inventory")
    listed = [e["path"] for e in inventory["files"]]
    actual = {p.relative_to(pool).as_posix() for p in pool.rglob("*") if p.is_file()}
    if len(set(listed)) != len(listed) or set(listed) != actual:
        raise ValueError("Pool membership changed: regenerate inventory")
    raw = {}
    for entry in inventory["files"]:
        path = pool / entry["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(pool.resolve()):
            raise ValueError("Input must stay within pool without file symlinks")
        raw[entry["path"]] = path.read_bytes()
        if digest(raw[entry["path"]]) != entry["sha256"]:
            raise ValueError(f"Raw hash changed: {entry['path']}; regenerate inventory")
    records, outcomes = [], []
    for entry in inventory["files"]:
        path = entry["path"]
        adapter = recipe["adapters_by_origin"].get(entry["origin"])
        if entry["status"] not in recipe["candidate_statuses"] or adapter is None:
            outcomes.append({"path": path, "outcome": "deferred", "reason": entry["status"], "records": 0})
            continue
        try:
            text = normalize(raw[path])
            items = adapt(adapter, text)
            if not entry.get("group_id"):
                raise ValueError("source group is required")
        except (ValueError, UnicodeError) as error:
            outcomes.append({"path": path, "outcome": "deferred", "reason": str(error), "records": 0})
            continue
        for upstream_id, body, group_override, changes in items:
            record = {"schema_version": recipe["schema_version"], **body,
                      "domain": "math" if adapter == "gsm8k" else "python" if adapter == "rst_document" else path.split("/")[0],
                      "language": "en" if adapter in {"gsm8k", "rst_document"} else "ja",
                      "group_id": group_override or entry["group_id"], "split": None,
                      "provenance": {"raw_path": "data/raw/pool/" + path, "raw_sha256": entry["sha256"],
                                     "upstream_id": upstream_id, "origin": entry["origin"],
                                     "source_url": entry.get("source_url"),
                                     "source_revision": entry.get("source_revision"),
                                     "source_verification": entry["source_verification"],
                                     "license_status": entry["license_status"],
                                     "participants": entry.get("participants")},
                      "transform": {"adapter": adapter, "version": 1,
                                    "operations": ["utf8_bom_removed_if_present", "newlines_to_lf"] + changes},
                      "review": {"training_status": "not_selected", "response_exemplar": False,
                                 "content_verified": False, "flags": entry["flags"] + changes}}
            record["id"] = "rec_" + digest(json_bytes(record))
            records.append(record)
        outcomes.append({"path": path, "outcome": "converted_candidate", "adapter": adapter, "records": len(items)})
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate canonical record IDs")
    encoded = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in records)
    report = {"files": outcomes, "record_count": len(records), "training_ready": False,
              "notes": ["No splitting, near-deduplication, correctness checking, or training selection performed.",
                        "Canonical preserves RST markup and conversation execution displays; these need a training view."]}
    manifest = {"schema_version": recipe["schema_version"], "inputs": {k: digest(v) for k, v in blobs.items()},
                "pipeline_sha256": digest(Path(__file__).read_bytes()), "records_sha256": digest(encoded),
                "record_count": len(records), "report_sha256": digest(json_bytes(report))}
    payloads = {"records.jsonl": encoded, "report.json": json_bytes(report), "manifest.json": json_bytes(manifest)}
    build_id = digest(payloads["manifest.json"])
    destination = root / "data/canonical/builds" / build_id
    if destination.exists():
        if {p.name for p in destination.iterdir()} != set(payloads) or any((destination / k).read_bytes() != v for k, v in payloads.items()):
            raise ValueError("Existing canonical build differs; refusing to overwrite")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only the completed directory; TemporaryDirectory cleans failures.
    with tempfile.TemporaryDirectory(prefix=".building-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "payload"
        staged.mkdir()
        for name, content in payloads.items():
            (staged / name).write_bytes(content)
        os.rename(staged, destination)
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(build(args.root))

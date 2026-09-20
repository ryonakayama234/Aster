"""Read pool without changing it; regenerate metadata-only corpus review artifacts."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/raw/pool"
DOCS = ROOT / "docs/corpus"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_format(text):
    try:
        lines = [json.loads(line) for line in text.split("\n") if line.strip()]
        if not lines:
            return "empty", 0
        return "jsonl", len(lines)
    except json.JSONDecodeError:
        try:
            json.loads(text)
            return "multiline_json", 1
        except json.JSONDecodeError:
            return "design_text_or_invalid_json", None


def build_inventory():
    notes = json.loads((DOCS / "source-notes-v0.json").read_text(encoding="utf-8"))
    sources = {}
    for group in notes["groups"]:
        for path in group["paths"]:
            if path in sources or not (POOL / path).is_file():
                raise ValueError(f"Duplicate or missing source mapping: {path}")
            sources[path] = group
    seed = json.loads((DOCS / "external-seed-manifest.json").read_text(encoding="utf-8"))
    external = {f["local_path"]: (s, f) for s in seed["sources"] for f in s["files"]}
    entries = []
    for path in sorted(POOL.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Review symlink before indexing: {path}")
        rel = path.relative_to(POOL).as_posix()
        data = path.read_bytes()
        entry = dict(path=rel, bytes=len(data), sha256=sha(data), chars=None,
                     origin="unknown", source_url=None, source_verification="unconfirmed",
                     group_id=None, split=None, usage="undecided", status="needs_source",
                     license_status="not_reviewed", flags=[], embedded_urls=[])
        text = None
        if path.suffix in {".txt", ".jsonl", ".json", ".py", ".rst", ".md"} or path.name == "LICENSE":
            try:
                text = data.decode("utf-8", errors="strict")
                entry["chars"] = len(text)
                entry["lf_text_sha256"] = sha(text.replace("\r\n", "\n").replace("\r", "\n").encode())
                entry["embedded_urls"] = sorted(set(re.findall(r'https?://[^\s<>"\)]+', text)))
                if not text.strip():
                    entry["flags"].append("empty_text")
                if "\x00" in text:
                    entry["flags"].append("nul_character")
                if re.search(r'!\[[^\]]*\]\(', text):
                    entry["flags"].append("image_reference_requires_text_only_review")
                if "Browsing web" in text or "utm_source=chatgpt.com" in text:
                    entry["flags"].append("separate_answer_citations_and_execution_displays")
            except UnicodeDecodeError:
                entry["flags"].append("invalid_utf8")
        if ":Zone.Identifier" in path.name:
            entry.update(status="excluded_os_metadata", usage="none")
        elif rel.startswith("image/"):
            entry.update(status="held_outside_text_v0", usage="reference_asset")
        elif path.relative_to(ROOT).as_posix() in external:
            source, original = external[path.relative_to(ROOT).as_posix()]
            if entry["sha256"] != original["sha256"]:
                raise ValueError(f"External raw hash changed: {rel}")
            entry.update(origin="external_dataset" if source["name"] == "gsm8k" else "external_documentation",
                         source_url=original["url"], source_verification="pinned_manifest_hash_verified",
                         source_revision=source["revision"], license_status="license_files_retained_see_sourcing_spec")
            if path.name == "LICENSE" or path.name in {"README.md", "license.rst", "copyright.rst"}:
                entry.update(status="metadata_only", usage="provenance")
            else:
                entry.update(status="candidate_needs_transform", usage="reading_candidate")
                entry["group_id"] = "derive_problem_ids_before_split" if source["name"] == "gsm8k" else f"python-chapter:{path.stem}"
        elif rel in sources:
            source = sources[rel]
            entry.update(origin=source["origin"], source_url=source["url"],
                         source_verification=source["verification"], group_id=source["group"],
                         usage="reading_candidate_not_response_exemplar",
                         usage_status=notes["default_usage_status"])
            if "participants" in source:
                entry["participants"] = source["participants"]
                entry["flags"].append("generated_answer_not_verified_fact_or_tool_observation")
            if source["origin"] in {"external_article", "user_ai_conversation"}:
                entry["status"] = "candidate_needs_review"
            else:
                entry["status"] = "needs_original_source"
            if source["group"].startswith("unresolved-") or source["group"].endswith("-batch"):
                entry["flags"].append("conservative_batch_group_pending_original_identity")
        elif rel.startswith("structured/"):
            entry.update(status="design_reference_needs_execution", usage="design_reference")
            entry["flags"].append("generator_and_execution_provenance_unconfirmed")
        if path.suffix == ".jsonl" and text is not None:
            entry["format"], entry["record_count"] = json_format(text)
            if entry["format"] != "jsonl":
                entry["flags"].append("not_line_delimited_json")
        if text is not None and entry["status"] not in {"metadata_only", "excluded_os_metadata"}:
            if any(f in entry["flags"] for f in ["invalid_utf8", "nul_character", "empty_text"]):
                entry["status"] = "needs_format_review"
        entries.append(entry)
    groups = defaultdict(list)
    for entry in entries:
        if entry.get("lf_text_sha256"):
            groups[entry["lf_text_sha256"]].append(entry["path"])
    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    return {"schema_version": 1, "stage": "inventory_not_training_release",
            "source_notes_sha256": sha((DOCS / "source-notes-v0.json").read_bytes()),
            "files": entries, "exact_lf_text_duplicates": duplicates,
            "near_duplicate_check": "not_performed", "current_written": False}


def render(inventory):
    entries = inventory["files"]
    lines = ["# AsterCorpus-v0 採用表（初回調査）", "",
             "この表は自動生成。出典の変更は `source-notes-v0.json` に記録する。", "",
             "**候補は採用確定ではない。currentへのコピー・分割・学習はまだ行っていない。**", "",
             "記事のURLはユーザー指定。閲覧できたことは全文一致や利用条件の確認完了を意味しない。",
             "本文内リンクは参考リンクとしてJSON索引に保存し、原典URLと区別する。",
             "ChatGPTの質問はユーザー、回答はChatGPT。モデル版と元会話の境界は未確認。",
             "用途はまず読む素材という提案。標準応答の口調としての採用は未決定。", "",
             "## 集計", "", "| 領域 | ファイル数 | 文字数（改行・記号含む） | bytes |", "| --- | ---: | ---: | ---: |"]
    for prefix in ["prose/ja/", "dialogue/ja/", "math/", "technical/", "structured/", "external-seed/", "image/"]:
        subset = [e for e in entries if e["path"].startswith(prefix)]
        lines.append(f"| {prefix} | {len(subset)} | {sum(e['chars'] or 0 for e in subset):,} | {sum(e['bytes'] for e in subset):,} |")
    lines += ["", f"全{len(entries)}ファイル。LF改行統一後の完全一致重複：{len(inventory['exact_lf_text_duplicates'])}組。近似重複と意味内容の正しさは未検査。", "",
              "## 状態", "", "| 状態 | 意味 | 件数 |", "| --- | --- | ---: |"]
    labels = {"candidate_needs_review": "候補：内容・用途・利用条件を確認",
              "candidate_needs_transform": "候補：出典固定済み、整形とレコード単位の分割が必要",
              "needs_original_source": "確認待ち：原典・作品・個別ページが未特定",
              "needs_source": "確認待ち：作り手・出典が未確認",
              "design_reference_needs_execution": "設計資料：実行・検証済み経験としては未採用",
              "metadata_only": "出典資料として保存、学習本文に入れない",
              "held_outside_text_v0": "画像：テキスト版v0では保留",
              "excluded_os_metadata": "OS付加情報：学習対象外", "needs_format_review": "形式確認待ち"}
    for status, count in sorted(Counter(e["status"] for e in entries).items()):
        lines.append(f"| {status} | {labels[status]} | {count} |")
    lines += ["", "## ファイルごとの採用案", "", "原典グループは未確定のものを広めに仮まとめしている。同じ著者・同じ会話という断定ではない。", "",
              "| pool内のファイル | 文字数 | 由来 | 原典・出典 | グループ | 状態 |", "| --- | ---: | --- | --- | --- | --- |"]
    for e in entries:
        url = f"[出典]({e['source_url']})" if e["source_url"] else "未特定／対象外"
        lines.append(f"| `{e['path']}` | {e['chars'] if e['chars'] is not None else '—'} | {e['origin']} | {url} | {e['group_id'] or '未設定'} | {e['status']} |")
    lines += ["", "## 次に確定すること", "",
              "1. 漫画の作品名と巻・話、Scrapboxの個別ページ、Xの元投稿。原典不明のミームは原本を保管して保留する。",
              "2. ChatGPT対話の口調を応答のお手本にするか。元会話が共通なら同一groupへまとめる。",
              "3. math/ja・math/en・technical/jaの出典または作り手。structured資料の生成元・実行履歴。",
              "4. 記事の抽出範囲と利用条件、対話の発言境界、画像参照、実行表示を点検する。",
              "5. 採用ファイル・採用範囲を固定し、原典group単位で分割してからcurrentへコピーする。", "",
              "構造化資料はJSON解析成功だけでは実行済みと扱わない。形式結果と注意点は `pool-inventory-v0.json` に保存。",
              "この調査では原本を書き換えず、外部コードも実行していない。", ""]
    return "\n".join(lines)


def main():
    inventory = build_inventory()
    (DOCS / "pool-inventory-v0.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DOCS / "selection-review-v0.md").write_text(render(inventory), encoding="utf-8")
    print(json.dumps({"files": len(inventory["files"]), "statuses": dict(Counter(e["status"] for e in inventory["files"])), "duplicate_groups": len(inventory["exact_lf_text_duplicates"])}))


if __name__ == "__main__":
    main()

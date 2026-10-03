# Aster

**作って、動かして、仕組みを理解するための自作TinyLM。**

AIの仕組み、Python、数学、コンピューターサイエンスを、ひとつの小さな言語モデル作りにつなげます。
出来上がったモデルだけでなく「なぜこう設計したか」「どう確かめたか」を残します。

将来の目標は、外部の計算器・記録・検索・実行環境を使い、自分の処理を進められるAsterです。
道具の呼び出し方に加え、返った結果を読み、失敗を修正し、記録を次の行動に使うところまで扱います。

## コーパスの方針

| 領域 | 作り手 | 役割 |
| --- | --- | --- |
| 日本語prose | ユーザーの自作文・選定した外部文章 | 語り方・説明・視点 |
| 日本語dialogue | ユーザーの自作文・選定した対話・AI対話 | 対話表現。応答のお手本への採用は別途選定 |
| Math | 外部調達 | 問題、式、解答の関係 |
| Python | 外部調達 | 基礎文法、短い処理、入出力 |
| structured / JSON / tool | Asterプロジェクト内 | 構造、道具の実行、観測、記録と再利用 |

詳細は [AsterCorpus-v0仕様案](docs/corpus/AsterCorpus-v0.md) と
[外部調達仕様](docs/corpus/external-sourcing-v0.md) を参照してください。
これらは初回提案で、v0の完成・凍結を意味しません。

2026-09-20：ユーザー選定の外部日本語・AI対話も候補に含める方針へ更新しました。
[採用表](docs/corpus/selection-review-v0.md) と [出典台帳](docs/corpus/source-notes-v0.json) を入口に整理します。
出典台帳は人が編集する情報、採用表は `scripts/inventory_corpus.py` で再生成する調査結果です。

## 現在あるもの

- UTF-8 byte tokenizerと、隣接する頻出ペアをまとめるBPE。
- Tokenizerの保存・読み込みとCLI。[既存BPE仕様](src/docs/AsterBPE-v0.1.md)。
- テキストを `id / text / source` のJSONLへ変換する処理。
- 小さなサンプルとテスト。
- [TinyLM本体と事前学習の最小ループ](docs/TinyLM-v0.md)。CPUで少量を覚える実験、train/dev評価、重みの保存・再読込・生成。
- 外部資料の原本収集と、[canonicalへの自動変換](docs/corpus/canonical-v0.md)。
- [学習用本文の抽出・分割・配合](docs/corpus/training-view-v0.md)、BPE実験、UIで使う観測bundle。[最初の結果](reports/training-pilot-v0.md)。
- Agent KernelのPolicy → ToolExecutor → Observation → State → Evaluatorと、固定`agent-calculate-store-v0`をAster Serviceから実行・観測する経路。[ServiceContract-v1](docs/ServiceContract-v1.md)。
- TinyLM shared backbone + scalar Decision Head、Decision benchmark、temperature calibration、model/fallback/abstain routing。
- [Decision CPU Baseline v0](docs/DecisionBaseline-v0.md)。BPE/weight update=train、temperature=calibration、design observation=dev、test=別runという分離、DecisionModel Artifact、別process reload、rule/model/model+fallback比較を実装。

JSONLは一行に一つのJSONを入れる保存形式です。保存にJSONを使うことと、モデルにJSONの生成を教えることは別です。
現在のTokenizerビルダーは `.txt` を読みます。コーパス変換器のJSONLをそのまま渡す接続にはなっていません。

## 学ぶ順序

1. **Corpus**：文字列、ファイル、文字コード、出典、学習と評価の分割。
2. **Tokenizer**：bytes、辞書、ペア頻度、圧縮、可逆変換。
3. **TinyLM**：ベクトル、行列、確率、次token予測、損失と勾配。
4. **実験**：未知の入力、過学習、比較条件、再現性。
5. **道具と記憶**：型、状態、実行、検証、保存と再利用。

各段階で「小さな例を手で追う → 実装を見る → 一つ変更する → 結果を比べる」を基本にします。

## 手元で確認する

WSLのリポジトリルートで、既存の仮想環境を使う例です。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/fetch_external_seed.py
.venv/bin/python scripts/build_canonical.py
.venv/bin/python scripts/build_training.py
```

後者はGSM8Kのtrain原本とPython教材の選定章・権利表示を取得します。
[収集manifest](docs/corpus/external-seed-manifest.json) に固定revisionとSHA-256を記録し、再実行時には一致を検査します。
原本はGit管理外の `data/raw/pool/external-seed/` へ保存します。
canonical変換は台帳からpoolを直接読み、`data/canonical/builds/`に版ごとの共通データを保存します。
`data/raw/current/`は採用原本を集める用途として残しますが、変換の必須入力にはしません。
採用・train/dev/test分割を経たpilot学習用入力は`data/training/<view ID>/`へ出力します。
train本文 → Tokenizerでtoken ID列へ変換 → TinyLM内のembeddingでベクトルへ変換、という流れです。
embeddingの数値はTinyLMの学習で更新します。実行方法は [TinyLM-v0](docs/TinyLM-v0.md) を参照してください。
収集済みというだけで学習可能とは扱わず、採用前に仕様の確認を行います。

Decision baselineはtestを自動開封しません。まずdevelopment runを作り、条件を固定した後だけ保存済みartifactをtestします。

```bash
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 train
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 test \
  --artifact runs/<development-run-id>/model-artifact
```

学習支援・開発時の約束は [AGENTS.md](AGENTS.md) にまとめています。

## 次に作るもの

2026-10-03: **ACT v0はユーザーWSL2実機で確認済み**。PR #49はmainへmerge済み、Issue #48は完了としてcloseした。実Run ID / bundle IDは当時のrepo記録に残っていないため、値は推測して補わない。

現在の研究Gateは Issue #20 / branch `feat/learn-v0-correction-transfer` の **LEARN v0 — Correction Transfer**。

```text
P0 Parent
  ├─ R1 Replay control
  └─ C1 Correction child

student rollout
→ student-visited state
→ teacher correction
→ matched-budget candidate update
→ corrected state
→ uncorrected sibling
→ model-only episode
```

中心仮説は「自分が実際に訪れた失敗状態への訂正が、同じ追加update量のReplayより未訂正の近縁状態へ転移するか」。

次を別々に判定する。

- **LEARN-Wiring**: Parent→rollout→correction→R1/C1→save/reload→same evaluatorが一周したか。
- **LEARN-Repair**: 訂正state自身が改善したか。
- **LEARN-Transfer**: 学習へ入れていないsibling stateでCorrection固有効果があるか。
- **LEARN-Sequential**: model-only episode完了へ効果が移ったか。

仕様は [LearnCorrectionTransfer-v0](docs/LearnCorrectionTransfer-v0.md)、作業順は
[TODO](docs/TODO/LearnCorrectionTransfer-v0.md)、進捗は
[Progress](docs/Progress/LearnCorrectionTransfer-v0.md) を正本にする。

このGateではRL / KLPO / new Tokenizer / new serializer / model scaling / sealed testを混ぜない。
Aster Service / aster-webは任意shellやrepository-local pathを公開せず、Job / Run / Artifact / Research evidenceの出典を分離する方針を維持する。

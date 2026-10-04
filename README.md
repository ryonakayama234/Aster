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

2026-10-03: **ACT v0はユーザーWSL2実機で確認済み**。PR #49はmainへmerge済み、Issue #48は完了。後続調査で実ACT Run ID `5a8076571dca446fa07190cf4fc62509` を回収し、LEARNのparent lineageとして固定した。未記録のbundle ID等は推測して補わない。

**LEARN v0 — Correction Transfer** は2026-10-04にconfirmatory measurementを完了し、**Not supported** でcloseした。

```text
30 family × 3 seeds = 90 units

primary: uncorrected sibling teacher-prefix accuracy
Correction vs matched Replay

wins / losses / ties = 6 / 19 / 5
median family delta = -0.25
exact sign-test p = 0.01463329792022705
verdict = Not supported
```

frozen key-shift benchmark distributionでは、Correction固有のLocal Transferは支持されず、matched Replayより悪い方向が優勢だった。corrected-state Repairと未訂正sibling Transferは別能力として扱う。

closeoutは [report](reports/learn-correction-transfer-v0.md)、作業履歴は
[TODO](docs/TODO/LearnCorrectionTransfer-v0.md) / [Progress](docs/Progress/LearnCorrectionTransfer-v0.md) を参照する。

次の研究Gateは **DIAG v0 — Correction Interference Curve**。
学習step 0/10/25/50/100でRepair / sibling transfer / base retention / parameter driftを追い、overfit・local interference・operation-specific representation・broad forgettingを切り分ける。現行trainingは決定論的なのでseed反復を独立trajectoryとして数えず、optimizer instabilityは必要なら別Trialで扱う。

今後の実験は [Experiment Cycle v0](docs/ExperimentCycle-v0.md) に従い、**Trial** と **Batch** を区別する。

- **Trial**: 小さく速く、mechanismやfailure modeを削る。既定は最大6 family程度 × 3 seed程度。途中結果を見てよいがconfirmatory claimには使わない。
- **Batch**: 大規模・凍結。manifest / code revision / endpoint / retry規則を固定し、resume可能なunitへ分けて実行する。

DIAG v0 Trial 0は実機instrumentation bugでpartialとなり、旧SHAのunitを新SHAへ混ぜないためsupersedeした。現在は Trial 1（`diag-correction-interference-trial-1`）を6 family × 1 deterministic trajectory = 6 training unitsで最初から回し、5 checkpointずつ30 repeated observationsを得る。checkpointは独立sampleではない。training-unit数はLEARN v0 Batchの90 unitの1/15だが、snapshot/evaluation overheadを含む実wall timeが1/15になるとは主張しない。

RL / KLPO / new Tokenizer / new serializer / model scaling / sealed testは、DIAGで原因仮説が絞れるまで混ぜない。
Aster Service / aster-webは任意shellやrepository-local pathを公開せず、Job / Run / Artifact / Research evidenceの出典を分離する方針を維持する。

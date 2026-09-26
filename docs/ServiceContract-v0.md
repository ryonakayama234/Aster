# Aster Service Contract v0

## 目的

SitesをAster内部のPython関数やshell commandへ直接結びつけず、Asterが公開する能力だけを安全に操作・観測する境界を固定する。

```text
Sites
  ↓ versioned JSON contract
Aster Service
  ↓ allowlisted recipe
Corpus / Tokenizer / Training / Runtime / Evaluator
  ↓
Run / Event / Artifact
```

Sitesは選択・開始・観測・比較を担当し、Aster本体は前処理・学習・推論・実行・評価・保存を担当する。

## v0の範囲

v0は既存Tokenizer Workbenchを壊さず、次の2 capabilityだけを公開する。

- `tokenizer.train`: 既存のbyte-level BPE評価job。
- `model.pretrain`: `configs/tinylm-overfit-v0.json`を使うTinyLM overfit job。

Raw upload、任意shell、任意Python、Agent execution、job queue、remote access、automatic model promotionはv0の対象外とする。

## JobとRun

`Job`はSitesからServiceへ出された実行要求、`Run`はAster本体が実際に開始して`RunLog`へ記録した実験である。同一IDとして扱わない。

```text
POST /jobs
   ↓
Job accepted
   ↓
subprocess start
   ↓
Aster RunLog
   ↓
Run started / events / output
```

Service再起動時に`accepted`または`running`だったJobは`interrupted`へ遷移する。Aster Runのcanonical evidenceは既存`run.json` / `events.jsonl`であり、Serviceはそれを別の事実として書き換えない。

## Artifact

Sitesへrepository-local pathを公開しない。v0では論理IDを使う。

```text
training_view:<64 hex digest>
tokenizer:<64 hex digest>
```

Serviceだけが論理IDを`data/training/...`または`artifacts/tokenizers/...`へ解決する。symlink、未知kind、不完全artifactはfail closedとする。

Tokenizerには既存`evaluation.json`の`experiment.view_id`がある。Pretrain開始前にTraining ViewとTokenizerの対応を検査し、異なる組み合わせを拒否する。Bridge内の一時領域で完成したTokenizer artifactは、content-addressedな同一内容であることを確認してAster本体の`artifacts/tokenizers/`へ昇格し、次のPretrainから選択可能にする。

## Recipe

Sitesはcommandを送らない。SitesはAsterが公開したrecipeを選ぶ。

v0:

```text
tokenizer-bpe-v0
  kind: tokenizer
  inputs: training_view
  parameters: vocab_size ∈ {256, 512, 1024}

tinylm-overfit-v0
  kind: pretrain
  inputs: training_view, tokenizer
  parameters: none
  config: configs/tinylm-overfit-v0.json
```

実行するmodule、path、timeout、CLI引数はAster Serviceが組み立てる。requestにshell commandやmodule名を指定する欄は作らない。

## JobSpec

```json
{
  "schema_version": "aster-job-spec-0",
  "kind": "pretrain",
  "recipe_id": "tinylm-overfit-v0",
  "inputs": {
    "training_view": "training_view:<digest>",
    "tokenizer": "tokenizer:<digest>"
  },
  "parameters": {}
}
```

未知field、未知recipe、余分なinput/parameterは拒否する。

## Job state

```text
accepted → running → completed
                   ↘ failed
accepted/running -- service restart --> interrupted
```

v0は同時に1 compute jobだけを許可する。新しい要求はHTTP 409で拒否し、queueingはしない。

## HTTP

Serviceは`127.0.0.1`だけでlistenし、既存Sites Origin allowlistとBearer tokenを維持する。

```text
GET  /status
GET  /capabilities
GET  /recipes
GET  /artifacts
POST /jobs
GET  /jobs/{job_id}
```

`POST /jobs`は上記JobSpecを受ける。また現行Sitesとの移行期間だけ、既存 `{view_id, vocab_size}` requestを`tokenizer-bpe-v0`へ変換する。

## Job bundle

```json
{
  "schema_version": "aster-service-job-bundle-0",
  "job": {"job_id": "...", "run_id": "...", "status": "running"},
  "run": {"schema_version": "aster-run-0"},
  "events": [],
  "outputs": {
    "training": {"schema_version": "aster-training-bundle-0"}
  }
}
```

実行中も保存済み`events.jsonl`と`training-bundle.json`をpollで読む。UI専用のlossやgenerationをServiceが再計算して捏造しない。

## Security invariants

- localhost以外へbindしない。
- Host / Origin / Bearer tokenを検査する。
- requestからshell commandを受けない。
- recipe、input、parameterをallowlistする。
- artifact IDからrepository外pathへ脱出させない。
- training subprocessをService processから分離する。
- Raw uploadと任意tool実行はv0へ混ぜない。

## 観測上の注意

`tinylm-overfit-v0`は接続とmemorizationを観測するrecipeであり、generalization改善の証拠ではない。Sitesは`training-bundle.json`の実測値を表示し、train loss低下をモデル全体の性能向上と同一視しない。

## 完了条件

- 現行Tokenizer requestがService経由でも成立する。
- typed JobSpecからTinyLM overfit jobを開始できる。
- Job IDとAster Run IDを区別して保存・表示できる。
- 実行中のeventsとtraining bundleをpollできる。
- 完了後のcheckpointを既存training bundleから追跡できる。
- Service再起動後、active jobが永続的に`running`へ残らない。
- 任意command・未知recipe・不一致Tokenizer/Viewを拒否する。
- pytestとPyrightを通す。

## 次の拡張

v0が安定した後に、Raw staging/upload、checkpoint artifact、inference、Decision、Agent runtime、trajectory/evaluation、learning candidate、model promotionを同じJob/Run/Artifact契約へ追加する。

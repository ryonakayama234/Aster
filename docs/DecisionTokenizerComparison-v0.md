# Decision Tokenizer比較 v0（Issue #28）

## 目的

PR #27のデータ4条件比較では、元8判断の専用BPEを固定したままtrain例を32 unique decisionへ増やしても、fitと数値・キー変更episodeの改善はseed間で安定しなかった。次の一変数比較として、**同じ32判断・同じAction候補・同じ教師・同じ更新順を使い、入力Tokenizerだけを専用BPEとByteTokenizerで比較する**。

この実験は自由文章生成や引数生成ではなく、用意されたAction候補をDecisionModelが順位付けする課題である。改善を保証せず、Tokenizer原因の確定を目的にしない。

## 現在の実装状態

- 実装PR: #30（Draft）。
- `src/aster/training/decision_tokenizer.py` に比較harnessを追加。
- `scripts/run_decision_tokenizer_comparison.py` にCPU実行CLIを追加。
- GitHub標準CPU runner上の1-epoch debug wiringで、BPE/Byte両armの学習、checkpoint再読込、dev診断、model-only episode記録まで通した。
- このdebug runは条件が異なるため **Issue #28の実測結果には数えない**。
- seed42/43/44のpreregistered measurementはまだ実行していない。

## 固定する学習課題

`DecisionDataIntervention-v0` の `numbers-and-keys` armそのものから、32 unique decisionだけを抽出する。

- operation: add / subtract
- number slot: original / changed
- key slot: original / changed
- phase: 0..3
- 合計: 2 × 2 × 2 × 4 = 32 unique decisions

両Tokenizerでcase ID、serialized decision content、candidate集合、target、splitを同一にする。training suite / development suite / split manifest / serialization / candidate-target内容のSHA-256をRunへ保存する。

## Tokenizer 2条件

### A. 専用BPE対照

PR #27で使った、元8 train判断だけをanchorとして作られた保存済みBPE artifactを**読み込む**。本測定時には再学習しない。

公開済みartifact digest:

`d68bd292d060625256cdcf16398d70db7d379542b8fe8aea45cbc7da611f9692`

CLIはこのdigestを既定で要求する。別artifactを暗黙に受け入れない。

### B. ByteTokenizer

既存 `src/aster/tokenizer/byte.py` と同じUTF-8 byte ID 0..255を、DecisionModelのArtifact interfaceへ接続する。

- raw byte: 0..255
- BOS: 256
- EOS: 257
- merges: なし
- vocab size: 258（special token込み）

byte列のencode/decodeは既存byte codecと一致することをテストする。

## 初期重みの対応

Tokenizer間で語彙数が違うため、全weightを同一とは扱わない。

1. 同じseedから両DecisionModelを初期化する。
2. shapeが同じstate tensor（Transformer、position embedding、LayerNorm、Decision Head等）はBPE側からByte側へ完全copyする。
3. `token_embedding` と未使用LM headはshapeが違うため、raw byte 0..255とBOS/EOSの**意味が対応する行だけ**共有する。
4. BPE merge token固有行はBPE側だけに残る。
5. full-shape共有tensorとsemantic-row共有部分を別々のhashとして保存する。

したがって「同じseed = 同じ初期weight」とは書かず、共有できた範囲を機械可読に残す。

## 学習条件

本測定の固定値:

|項目|値|
|---|---:|
|width|16|
|heads|2|
|layers|1|
|context|2048|
|learning rate|0.001|
|epoch|50|
|updates / arm|32 × 50 = 1,600|
|example exposures / arm|1,600|
|shuffle|epochごと、seed固定|
|seed|42 → 43 → 44|
|CPU threads|2|
|stability window|最後3epoch|

Wolfram再検算: 2 arm × 3 seedで9,600 updates/exposures。epoch0を含むtrain評価は1 armあたり32×51=1,632、全6 armで9,792 decision。

seed内ではBPE/Byteに同じcase順を与え、保存済み `epoch-order.jsonl` が完全一致することをrun終了条件にする。

## fit判定

既存Decision fitの定義をそのまま使う。

- 最後3epochすべてで32/32正答
- 全caseのtarget margin > 0
- candidate NLLを併記

一瞬の32/32はfit成立にしない。

## Token / compute監査

BPEとByteは同じexposureでも計算量が同じではない。そのため各armで次を保存する。

- tokenizer manifest / digest / vocab size
- candidateごとのtoken ID
- tokenごとのUTF-8 byte range / hex /表示用断片
- serialized input digest
- 最大candidate長
- 1 passあたりのsemantic token数
- right-paddingを含むtoken数
- 50epoch分のtoken budget
- 総parameter数
- Decision forward pathで使われるparameter数
- token embedding parameter数
- Decision forwardでは使わないLM head parameter数
- update / epoch-end eval / reload時間
- process max RSS（両armを順番に走らせたprocess全体の最大値）
- Python / PyTorch / thread数

TinyLMはDecisionModelで `encode()` だけを使う一方、語彙依存LM head自体はmoduleに保持される。このため総parameter数とDecision forward pathのparameter数を分ける。

## context / padding / round-trip

trainとdevについて、学習開始前に両Tokenizerで以下を検査する。

- UTF-8 bytesへ完全復元できる
- decode(encode(text)) == text
- BOS/EOSが本文byte rangeと混ざらない
- 全candidateがcontext 2048以内
- candidate axisとbatch paddingを混同しない
- `DecisionHead` が各candidateの実長末尾をpoolする

context超過があればそのarmだけ切り詰めず、比較全体をinvalidとして止める。

**予約testは長さ確認目的でもtokenizeしない。** testはsplit整合のためsuite/digestだけを保存し、scored_cases=0を維持する。

## dev評価

各50epoch checkpointを再読込して、PR #27と同じ56 dev decisionと8 model-only episodeを評価する。

- seen control
- numeric shift
- key shift
- joint shift
- candidate permutation
- reachable delay（teacher-prefix診断）

既存diagnosticのaccuracy/NLL/marginに加え、model-only episodeでは各stepのteacher Action、全candidate score、selected Actionを保存する。最初の誤選択では各candidateのserialized input digestとtoken/byte分割も保存し、その後のtrajectoryを同じepisode recordへ残す。

fallbackは使わない。teacher candidate coverage、評価前後weight一致、checkpoint reload一致を検査する。

## 分割

- train: 同じ32 unique decisions
- dev: PR #27と同じ56 development probes。親train groupと重なる診断probeであり、独立task-family holdoutとは主張しない。
- old test: sealed / not evaluated
- reserved redundant-calculation prefix: sealed / scored_cases=0
- calibration: not run

Tokenizer比較の条件選択にtest scoreを使わない。

## CPU実行順

保存済みBPE artifactの実際のpathを指定する。

```bash
.venv/bin/python scripts/run_decision_tokenizer_comparison.py \
  --root . \
  --bpe-tokenizer runs/<source-run>/tokenizer \
  --threads 2 \
  --seed 42
```

seed42の両armで配線・資源上限が成立したら、結果の良し悪しに関係なく同じ条件で43/44へ進む。

```bash
.venv/bin/python scripts/run_decision_tokenizer_comparison.py --root . --bpe-tokenizer runs/<source-run>/tokenizer --threads 2 --seed 43
.venv/bin/python scripts/run_decision_tokenizer_comparison.py --root . --bpe-tokenizer runs/<source-run>/tokenizer --threads 2 --seed 44
```

提案上限は1 armの学習＋評価30分、6 arm合計180分。timeout/OOM/例外も結果として保存し、片側だけepochやLRを延長しない。

## Run成果物

主な保存物:

- `study.json`
- `training-suite.json`
- `development-probes.json`
- `sealed-test-suite.json`
- `split-manifest.json`
- `protocol-digests.json`
- `shared-initialization.json`
- `tokenizers/bpe/`, `tokenizers/byte/`
- `*-train-input-audit.jsonl`
- `*-dev-token-audit.jsonl`
- `arms/<bpe|byte>/learning-curve.jsonl`
- `arms/<bpe|byte>/epoch-predictions.jsonl`
- `arms/<bpe|byte>/epoch-order.jsonl`
- `arms/<bpe|byte>/checkpoint/`
- `arms/<bpe|byte>/dev-evaluation/diagnostics.json`
- `arms/<bpe|byte>/dev-evaluation/predictions.jsonl`
- `arms/<bpe|byte>/dev-evaluation/episodes.jsonl`
- `arms/<bpe|byte>/dev-evaluation/episode-details.jsonl`

重み・大量logはRun artifactとして保持し、Gitへ結果を載せる場合は軽量summary/reportだけにする。

## 解釈の分岐

1. Byteでfitと変更課題がseed間で安定改善 → 状態依存課題設計へ進む候補。
2. 両方fitするがdev失敗 → Tokenizerより状態/構造表現を次の独立比較へ。
3. 両方fit不安定 → データ固定のままLRまたは予算を一変数ずつ調べる。
4. 改善なし / Byteの計算費用が大き過ぎる → この比較を終了し、残る仮説を記録する。

どの分岐でも自動promotionしない。**比較可能な証拠と次の判断が残ること**を完了条件にする。

## 2026-09-30 測定前の確認・継続契約

観測根拠: #27で元課題のfitが成立しても変更課題が崩れ、32判断への拡張ではfit未成立も残った。
仮説: 専用BPEの長い結合が学習成立または変更対応の障害の一部である。
反証になる結果: 同じ32判断・更新予算でByteに変えても3seedでfit/変更episodeが安定改善しなければ、今回の条件で主因として採用しない。
Byteもfitしない場合、BPEの影響がないと証明したことにはならず、最適化・容量・系列長を残す。
達成目標: 全6条件の成績・誤答・CPU費用・未取得・次の分岐を残す。能力改善は完了条件ではない。

実装の正本はPR #30 head 64b08709163e4a1478b620448d2717898c4a223e。
初期化はこのPRのTransformer/head共有に加え、raw-byte/BOS/EOSの意味対応行共有を維持する。
以前手元に残った「語彙embeddingを共有しない」計画とは条件が異なり、その計画の試運転/途中実行は本測定に含めない。
元8trainからBPE artifactを決定的に復元し、公表digestの完全一致を確認したファイルを入力する。
学習harnessではBPEを再学習しない。既存の原artifactそのものを回収したとは扱わない。

今回Linux CPUでの測定前に、各armの学習・reload・dev合計30分をSIGALRMで強制する。
timeout/例外なら比較を終了しRunへ記録、未取得を0点扱いしない。片側のみ延長しない。
RSSは両armを順次実行したprocess全体の最大値であり、arm別メモリ比較は行わない。
ユーザーは今回私的素材の提供や新しい学習例作成をする必要がない。
必要になった時点でWSL再現手順と返してほしいstudy.jsonを渡す。

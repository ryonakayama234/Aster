# Decision状態診断 v0

Issue #24の最初の診断実装。#23のtrain fit成立後、保存済みモデルを更新せず、手順暗記と状態判断を区別する入口を作る。
旧testを開封せず、旧suiteを変更せず、新しいdev診断suiteとして保存する。

## 最初の48 decision

| slice | 件数 | 変更するもの |
|---|---:|---|
| seen_control | 8 | 元trainの2課題・4段階。再読込の基準 |
| reachable_delay | 16 | 各段階の前に、無関係キーへmemory.put、必要ならmemory.getを実行 |
| numeric_shift | 8 | 足し算6+10、引き算20−7。操作とキーは固定 |
| key_shift | 8 | 数値・操作は固定、保存先キーだけ変更 |
| candidate_permutation | 8 | seen_controlの候補を逆順にする |

元課題はadd(2,3,total)、subtract(9,4,result)。prefixのcalculator/put/getと遅延操作は既存ToolExecutorで実行し、RuntimeContext snapshot・Observation・TaskEvaluatorの評価を保存する。stepを書き換える加工はしない。teacher targetも候補に含まれることと実行可能性を確認する。

例：通常のstep1はmemory.putだが、計算前に無関係キーへputしたstep1はcalculator。同じphaseでも遅延1回なら判断材料に関係ない履歴が増え、正解Actionは同じ。
これは課題に無関係なmemory操作の挿入であり、全く同一のstate/historyから異なる教師を要求するものではない。

## 対照と採点

- step-only: 元suiteのtrainだけからstep→Action種別の最頻値を学ぶ。候補中の対応Actionを選ぶ。stepが未知、または対応候補がなければabstainとして不正解。数値/キー引数は既存candidate builderが供給する。
- first-candidate: 常に先頭候補を選ぶ弱いcandidate-only対照。あらゆるcandidate shortcutを検出するものではない。
- model: teacherの正しいprefix履歴で候補をscoreし、Action accuracy / raw NLL / target probability / marginを記録。temperatureをfitしない。
- permutation: 同じ意味のActionを選ぶか、逆順に揃えたlogit差も保存。候補を独立scoreする現実装ではほぼ不変が期待される。汎化能力の強い証拠として扱わない。
- model-only episode: seen/numbers/keyの6初期taskを既存run_loopで最大8step実行。RuleBasedPolicyも同じtaskで対照実行。terminalかつgoal_satisfiedを完了とする。fallbackなし。全trajectoryと、modelが実際に訪れた状態でのteacher candidate coverageを保存する。
- delayed-prefixからのmodel-only継続はこの版では未実装。遅延sliceはteacher-prefix decision診断のみ。

候補にはcalculator結果を使ったmemory.put引数やgoal_verifiedが含まれる。候補生成器がすでに与える情報と、モデルの能力を区別する。正解はこの限定課題のRuleBasedPolicyが選ぶAction一つとの完全一致。一般課題の複数有効Action採点は未実装。

## 分割と由来

全48caseはdev診断。train/calibration/testとして新しいデータを生成・利用しない。
同じ元課題から派生したseen/delay/numbers/key/permutationは同じleakage groupにまとめるため、groupは2件だけ。
元train課題由来の反例なので、独立した未知テンプレートのholdoutではない。numeric/keyは本文を変えたprobeであり、新task familyと呼ばない。
モデルごとに同じsuite digestを照合し、case全文・候補/target・case digest・入力digest・token長を保存する。
Tokenizer・モデルを固定したまま評価する。旧testへの推論・旧testからの学習採用・calibration調整は行わない。

## CLI

PR #25上のfit checkpoint（eight-fixed/eight-shuffle）を読み込む。元suite digestとcheckpoint identity/model/Tokenizer hashを検査し、不整合は停止。

```bash
.venv/bin/python scripts/run_decision_state_diagnostics.py \
  --root . --threads 2 \
  --checkpoint runs/96d2e995bf15464c9733b618f7e7da69/arms/eight-shuffle/checkpoint
```

checkpointが別checkoutにある場合は、その手元の絶対パスをCLIへ渡せる。これはローカルCLIだけの設定で、Service/Webにpath入力APIを追加しない。
新しいRunへrun.json/events.jsonl/diagnostic-suite.json/diagnostics.json/predictions.jsonl/episodes.jsonlを保存する。元checkpointは上書きしない。
実行前後にモデルtensorが一致することを検査する。モデル更新・temperature fitting・promotion・optimizer resumeはない。

## 初回結果と次

[実測報告](../reports/decision-state-diagnostics-v0.md) と [ユーザーfit報告](../reports/decision-fit-user-pc-v0.md) を参照。
この実装は#24の診断用入口。新しい学習suite・構造的holdout・改善学習の実装完了ではない。
失敗のslice/Actionを分析して、必要なtrain拡張を小さく事前定義し、新しいdev/test分割を決めてから次の比較へ進む。

## Service/Webへの関係

Aster側が実行・評価・canonical Runを保存し、Webは将来の表示・比較を担当する。
本PRはoffline CLIと軽量reportのみ。ServiceContract-v0/v1、公開recipe、artifact catalog、aster-webは変更しない。
将来allowlisted recipeへ接続する際は、論理checkpoint ID・suite版・model_onlyとruleの区別・分母・由来・weights更新有無を公開bundleへ定義する。

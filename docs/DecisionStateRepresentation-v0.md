# 状態判断と短縮入力：学習前の検証 v0

2026-10-01 JST。親研究はIssue #24。今回の完成点は事前実験計画と、実行で検証された小さなdev対照課題群。学習比較runnerや改善実測は次の変更で行う。

## どんなふうに考えたか

PR #30ではByteが32判断のfitを3/3 seedで満たした一方、数値変更の自力完了は全条件0/2だった。Tokenizerだけで変更対応を説明する根拠は得られていない。現在のDecisionModelは、候補生成器が数値・キー・計算結果を組み立てたActionを順位付けする。計算や引数生成の能力とは区別する。

次の仮説は「長い状態/履歴のJSONから必要な事実を抽出する負担が判断を妨げている」。反証可能な比較として、同じ状態・履歴・候補を現行入力と短縮入力で表す。短縮入力で改善しなくても、結果を残す。専門的な情報選択をプログラムへ外部化する介入なので、成功しても汎用的な読解能力の証拠にはしない。

## Aster全体での位置

|経路|目的|次の成果|
|---|---|---|
|判断：#24|状態から次のActionを選ぶ|今回のpreflight→入力比較→誤った訪問状態の診断|
|言語：#21|次token予測と文章生成|文書group分割、学習前後loss、固定prompt生成|
|観測：#29→aster-web#2|失敗と変更を人が追う|保存済みRunを再生し、候補得点と最初の誤りを比較|
|育成：#20|経験から何が変わるか測る|訂正→更新なし対照→候補学習→親子比較|

Decision成功を言語pilotの必須条件にしない。Web側は実行・評価を再実装せず、Asterが保存する証拠を表示する。今回のCLI出力はService公開bundleではなく、Webへ導入済みと扱わない。

## 課題と作業方法

足し算/引き算、元の数値・キーと既存の同時変更probe、4種類の関連履歴、無関係操作0/1/2回で48件。遅延は実際に無関係キーへmemory.put、続いてmemory.getを実行する。phaseは生成と対照組の指定にだけ使い、短縮入力には入れない。

- 同じstepで正しいActionが異なる32組。
- stepが異なっても正しいActionと短縮入力が一致する32組。
- 各組はケースを共有する。64組を独立した64課題と数えない。
- 全caseはdev。2つの元train親groupからの派生であり、独立holdoutではない。
- 新しいtrain/test分割を完成したとは扱わない。旧test/予約testを生成・採点する入口は呼ばない。

全prefixを別のRuntimeContextへ再実行し、state before/after、Observation、Evaluation、候補・teacherを照合する。途中状態からルールを最大8追加Actionで継続し、terminalかつgoal_satisfiedを確認する。既存run_loopはstepを0から開始するため、この検査はprefix長から連番で記録する局所的な実行補助を使う。既存runtimeの意味論は変更しない。

## 短縮入力の契約

`compact_input(state, history, candidate)` は教師exampleを受け取らない。

|field|情報源|
|---|---|
|task|観測可能な元taskの全field|
|memory|対象キーの存在と現在値|
|events|全calculatorイベントと対象キーのput/get。元Action・Observationを保持|
|last_failure|最後のObservationが失敗ならその記録。成功/履歴なしならnull|
|candidate|得点対象のAction|

step、phase、teacher、target_index、evaluation、正解計算値を追加しない。計算結果は実Toolの観測からのみ得る。無関係履歴を落とすのはこの限定課題の設計選択。削除・上書き・別taskでのcalculator結果・失敗後の回復などに十分な表現だとは主張しない。

同一候補集合内の全短縮入力が完全一致するのに教師が異なる衝突を検査する。衝突0件はこの48件での確認であり、あらゆる訪問状態での十分性の証明ではない。次のmodel rolloutでも毎回監査する。

候補のみの曖昧性は、全候補ActionのJSONが同一のcaseをまとめて調べる。そのgroupに異なる正解があれば、決定的なcandidate-onlyモデルでは全件正答できない。dev正解から計算する上限は診断用のin-sample oracle上限であり、学習済みbaselineの精度ではない。数値/キーの転移能力も示さない。

## 次の学習比較の事前計画（runnerは未実装）

1. PR #30の比較コードを依存として採用する場合は、そのmerge状態とSHAを再確認。今回のpreflightはmain `098a009`から独立して実装している。
2. 既存numbers-and-keysのtrain32判断を固定し、新dev48件をtrainへ混ぜない。データmanifest/candidate digestを学習前に保存。
3. UTF-8 Byte・width16/heads2/layers1/context2048・LR0.001・50epoch・CPU2threadsを両armで固定。原入力と短縮入力で同一weight初期値・提示順・候補順を使う。Tokenizerを再学習しない。切詰めは禁止し、超過は失敗として記録。
4. seed42でrunnerの動作確認後、条件を変えず43/44も測る。成功したseedだけ選ばない。各armの学習/保存/reload/devを30分以内とし、打切りを欠損として記録。更新/提示数は一致させ、token数・時間は差を記録する。
5. train accuracy/NLL/min marginと最後3epoch fit、devのteacher-prefix判断、対照組の両方正答、遅延prefixからmodel-only継続、最初の誤り、候補coverageを記録。途中stateはmodelが実際に訪れたものも保存する。fallback成功を合算しない。
6. step-only、first-candidateと、trainだけで学習するcandidate-only対照を同じケースで評価する。現在のoracle上限をcandidate-only学習結果の代用にしない。
7. calibrationは今回は行わずraw score/NLL/marginを記録。旧/予約testを開封しない。この結果は診断に限定し、新たな構造familyのsplit/testは別protocolで事前固定する。

この計画は新しい測定前に固定・再確認する。データ量・モデル容量・入力方式を同時に増やさない。短縮入力成功なら「専門的な情報整理を外部化したとき、この限定判断が改善した」と解釈する。train未fitなら最適化、teacher-prefix成功でも継続失敗なら訪問状態・誤り伝播を次に切り分ける。

## 達成目標と再現

今回：課題を実行で検証し、必要情報の欠落・候補shortcutを学習前に検査できること。モデルの改善は今回の完成条件ではない。

```bash
.venv/bin/python scripts/run_state_representation_preflight.py --root .
.venv/bin/python -m pytest -q tests/benchmark/test_state_representation.py
```

新Runにsuite/pairs/preflight/casesとrun/eventsを保存する。casesは元/短縮入力と継続trajectoryを含む。Python sourceのfile hash・snapshot digest・git SHA/dirtyを記録する。benchmark packageの学習依存は遅延importし、preflightはtorchなしでも動く。

ユーザーの追加素材やPCでの長時間学習は今回不要。エージェントが実装・検査・記録を担当し、ユーザーには後の観測結果を見て利用場面を具体化してもらう。

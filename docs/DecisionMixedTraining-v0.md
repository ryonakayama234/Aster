# Aster：全体像と次の混合学習比較 v0

2026-10-01。調査・実験計画のみ。新規学習、能力測定、Service/Web実装は未実施。
基準はPR #36 head `5291a377abb3140cbf4255ec094738cbf0b7c3aa`（open/Draft/未merge）。
添付「Aster 計画.pdf」（1頁）と両repoのREADME、Issue、PR、実測reportを照合した。

## 着手前の考え方・作業方法・達成目標

考え方：正常例置換の負の結果と旧v0モデルへのv1診断を受け、正常経路の支持を残すデータ配分を比較する。
「学習量を増やせば改善」「容量不足」「履歴回数だけを暗記」のどれも現時点では確定しない。
作業：資料照合→競合仮説と研究文献確認→比較の算術監査→学習前に必要な生成・検証契約の明文化。
達成目標：次の実装が条件を都合よく変更せず着手でき、結果が悪くても次の分岐を選べる計画を残す。
今回の完了は計画の保存であり、以下のrunner/preflight/新モデルは未実装・未測定。

## 全体像と現在の責務

|経路|現在あるもの・制約|次に問うこと|
|---|---|---|
|素材→Tokenizer→TinyLM→推論|一般Corpusと言語学習の経路。Decision scorerはランダム初期化で、一般言語能力を継承する比較ではない|固定本文の暗記と未知本文予測を区別する（#21）|
|状態→候補→Decision→実Tool→観測→評価|候補生成器が引数を作る。モデルは候補を順位付けする。算術/自由引数生成の評価ではない|候補が同じでも現在状態に応じて選択を変えられるか（#24）|
|Run/Artifact→Service→aster-web|Asterが実行・評価・記録の正本、Webが操作・表示・比較。研究結果公開bundleは未実装|保存済み2条件の同じ失敗を比較できるか（#29→aster-web#2）|
|実行→訂正→候補更新→親子比較|実験入口の後続計画。使用しただけで自動更新する設計ではない|自力で訪問した状態への訂正が、追加更新だけより効くか（#20）|

PDFの直列工程は学習上の地図として使う。言語pilotはDecision成功を必須前提にせず、
保存結果の観測もモデル改善を待たずに進められる。素材登録の経路はaster-web#3。
Web READMEの旧「未接続」記述と本人の接続成功報告は別に保持し、今回ブラウザ/WSL再検証済みとは扱わない。

## 直近の証拠と限界

正本：[固定v1実測](../reports/decision-fixed-v1-diagnostics-v0.md) と同名JSON。

- baselineは既知正常8判断を全seed保持、正常例置換モデルは6/5/6。
- 誤値-999/-777による選択変化は0〜2/32組。1回→2回の誤保存では9〜28/32組。
  関連履歴回数・入力長・EOS位置が同時に変わり、位置符号化だけの因果効果ではない。
- 非重複の同候補/異target対照24組の両方正答はbaseline5/4/2、置換0/2/2。
- 数値変更の初期自力完了は全6モデル0/2。候補coverage成立だけではranking成立を示さない。
- 旧v0で学習したモデルをv1へ診断した結果。新仕様未学習と容量/表現の問題を混同しない。
- 2親group由来dev probeであり独立holdoutではない。派生例や3seedを独立課題数に加算しない。
- PR #33はfresh trainingであり、逐次学習の意味で「破滅的忘却」と呼ばない。

## 優先する問い

**v1で正常例を残し、正解Action頻度と総更新を揃えて復帰例を混ぜると、
正常保持を損なわずに同候補/異targetの選び分けが改善するか？**

競合仮説：
1. 正常例の支持範囲が不足した。正常保持の回復をまず測る。
2. 一回誤保存の経験が復帰判断を助ける。非重複対照と二回反復を別に測る。
3. 正常例保持だけでは不十分で、履歴の表現に弱点がある。
4. v1で新たに要求される再確認が未学習。今回の混合例は完了取消を教えないので、
   そのsliceだけの失敗を表現/容量の反証としない。
初手から誤る既存結果は、自力rolloutでの分布変化だけでは説明できない。

## 次の比較仕様：学習前に生成物を固定する案

両条件を新規初期化から学習し、教師・候補・prefix実行・replay・Evaluatorを
`calculate-and-store-current-v1`へ揃える。旧モデルからの追加学習ではない。
旧v0例をラベルだけ変更せず、実Toolのv1 Sessionで作り直す。

### train生成規則

`training/decision_data.py`のTRAIN_NUMBERS × TRAIN_KEYSだけを使用。
add: (2,3)/(4,7) × total/sum_slot、subtract: (9,4)/(12,3) × result/difference_slot。
計8task、各正常4判断で32unique。devの数値/キー/joint条件を新trainに移さない。

復帰16判断：8train taskで各2prefix。
A: 空状態から対象キーへ-999を1回put → calculatorが正解。
B: 正常calculator後に対象キーへ-999を1回put → 修復putが正解。
それぞれ実行・観測・状態・候補・教師の到達可能性を検証する。
正常get/stop反復16slotは、正常32のphase2/3（各8件）を再提示する。
ここでいう反復slotは同一例の追加提示であり、二回put履歴を学習することではない。

|項目|normal-repeat-v1|normal-plus-recovery-v1|
|---|---|---|
|64提示slot/epoch|正常32×2|正常32＋復帰16＋正常get/stop再提示16|
|unique判断（preflightで確認）|32|48|
|calculator/put/get/stop教師slot|各16|各16|
|epoch/単例更新|25 / 1,600|同じ|
|各正解Actionの総提示|400|400|
|正常例への総提示|1,600|1,200|
|復帰例への総提示|0|400|

元の各正常例は両条件に含まれる。ただし混合ではcalculator/put正常例の提示が50→25回、
get/stop正常例は50回のまま。正常例への個別exposureは同じではない。
比較は「固定総予算でのデータ配分」の介入であり、復帰データだけの純粋因果効果ではない。
正常例exposureを厳密に揃える比較が必要なら、追加更新controlを持つ別protocolへ分ける。

### 固定するモデルと資源

Byte、compact serializer `aster-decision-compact-input-0`、width16/heads2/layers1/context2048、
LR0.001、shuffle単例更新、CPU2threads、seed42/43/44、各arm学習+評価30分上限。
seed内で同じ全初期weightと64slotのindex permutationを共有しhashを保存する。
提示index順が同じでも内容は異なる。token総量・padding量・wall/CPU timeは別に記録。
モデル拡大、事実flag追加、Tokenizer変更、正規化、LR変更を同時に混ぜない。
配線/資源を確認した後は結果の良し悪しでseedを選ばず全6armを実施する。
debugと正式Runを分離し、timeout/OOM/context超過/例外を保存する。

### 必須preflight（未実施）

- 元train task32正常/16復帰の実Tool replay、v1教師・候補coverage・独立rule継続。
- unique件数、正解Action各16slot、全初期weight/slot順、可逆Byte encodeとcontext上限。
- 教師/phase/goal正解からの入力漏洩なし、同一model-visible入力に異targetなし。
- train全arm unionとv1診断128入力を照合。重複は保持確認へ分離し、
  旧reportの「非重複24組」を新train下でも固定件数と仮定しない。
- 同候補・異target主対照のcase ID一覧、split/source/serializer/target/candidate/input digestを固定。
- 旧/予約testは構築・長さ点検・採点を行わない。calibrationなし、raw score/T=1。
- 生成件数や教師頻度が予定に合わなければ測定前に停止・改版。黙って例を削除しない。

## 評価・達成目標・停止条件

epoch0と毎epoch末に、重複slotでなく全unique train判断を同checkpointで評価。
accuracy、candidate NLL、min target margin、個別予測を保存。
fitは最後3epochの全unique判断正答かつ各target margin>0。NLLを併記。
単一候補のmarginはNoneとして別件数を示し、正のmargin判定の対象外にする。
32対48件の全train accuracyを同じ難易度の尺度として扱わず、共通正常32の値も保存する。

devはPR #36のv1 128prefixを使用し、全slice・重複/非重複・対照組を事前manifestで固定する。
一次能力指標は非重複の同候補/異target対照の両方正答。
正常保持、誤保存1回/2回、完了取消4種、誤値因子、数値/キー/jointを別集計。
改善はseed42/43/44のpaired件数と差を全て示し、都合のよい平均/seedのみを報告しない。

teacher-prefix判断、各prefixから最大8行動のmodel-only継続、初期8episodeを分離。
Action完全一致と停止時task_successを別評価。候補のみlookup、rule、coverageを対照として保持。
未停止は非成功。context超過invalidは理由と全開始数を示し、通常失敗へ合算しない。
両arm共通で評価可能なcaseの比較と、全開始caseの集計を併記する。
reload一致、評価中weights/RNG不変、実行/採点の再照合を確認する。
元checkpointと旧reportを上書きしない。独立holdoutやmodel promotionの主張はしない。

完了は事前条件・全arm証拠・CPU費用・限界・次の分岐が残ること。
能力目標は正常保持と状態選び分けの両立。未達でも条件を追加調整せず結果を残して終了できる。

## 結果別の次の一段

|観測|次の判断|
|---|---|
|正常保持＋非重複対照が全seed改善|混合案を次候補へ。二回反復、完了取消、初期自力完了の未解決を残す|
|正常保持が回復、反復対照が崩れる|関係を観測事実から表すserializer対照を別protocol化。teacher正解を入力しない|
|train fit未成立|同じデータでfit診断へ。汎化や容量だけに帰属しない|
|teacher-prefixは成立、自力だけ崩れる|訪問状態の訂正→親子比較（#20）の根拠にする|
|完了取消だけ失敗|未学習v1 mutationの訓練支持を別に問う|
|改善なし/打切り|この比較を閉じる。追加データ・モデル拡大へ自動移行しない|

## 一次研究から借りる診断方法

- Geirhos et al., Shortcut Learning in Deep Neural Networks:
  https://arxiv.org/abs/2004.07780
  学習内成功だけで意図した手掛かりの使用を確定しない。Asterでは意味が必要な対照を測る。
- Ross, Gordon & Bagnell, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning:
  https://proceedings.mlr.press/v15/ross11a.html
  行動が次の入力分布を変えるため、一判断と自力実行を分離する。
  今回は固定prefixの混合学習で、DAgger実装やその保証の適用ではない。
- Zaremba & Sutskever, Learning to Execute:
  https://arxiv.org/abs/1410.4615
  課題難度の配分を学習条件として扱う視点。今回の固定混合がAsterで改善する保証ではない。

Wolframの算術照合：64×25=1,600/arm、2×3×1,600=9,600更新。
正解Actionは16×25=400/種。混合の正常提示48×25=1,200、復帰16×25=400。
epoch0を含むunique train評価はnormal32×26=832、mixed48×26=1,248判断/arm。
算術検査は能力測定ではない。

## 今回の成果と役割

今回：PDF/main/最新Draftの現状照合、一次研究3本、コード上の生成入口、算術監査、計画保存。
未実施：生成preflight、runner実装、学習、dev新測定、ユーザーWSL確認。
実装時にはpreflight生成物/digest/測定sourceを正式登録し、その後に測定する。
Web観測は#29→aster-web#2、言語pilotは#21、raw登録はaster-web#3へ継続。
今回はAPI/UI/Site公開版を変更せず、追加プラグインやユーザーPC作業は必要ない。

## 2026-10-01追記：生成preflight完了

[実測report](../reports/decision-mixed-v1-preflight-v0.md)とJSONに固定生成物/digestを保存した。正常32＋復帰16unique、両64slot/Action各16、衝突0、dev128のunion重複14、非重複主対照24組。モデルscoring/更新0。上の未実施記述は計画作成時点の履歴で、runner実装・比較学習・新能力測定は引き続き未実施。次は保存digest一致を要求するrunner。

## 2026-10-01追記：固定条件の比較学習完了

上の未実施記述は登録時の履歴。[正式protocol](DecisionMixedMeasurement-v0.md)をclean sourceへ保存後、全6条件を測定した。[実測](../reports/decision-mixed-v1-study-v0.md)では両条件全seed stable fitと正常保持が成立、主対照は対照4/4/4対混合4/2/2。prefixのseed依存改善と混合seed44の数値初期1/2は保持するが、汎化の安定改善や採用成立とは扱わない。次は関係明示serializerによる診断を別登録する。

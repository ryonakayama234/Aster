# Decision raw/compact比較 v0：実測と次の判断

2026-10-01 JST。Issue #24。[事前protocol](../docs/DecisionRepresentationComparison-v0.md)。PR #30/#31の研究実装を統合したブランチで実施。

## 何を考え、何を測ったか

長い状態/履歴から必要な事実を取り出す負担が判断を妨げている、という仮説を検証した。学習データ32判断・Byte・同一初期weight・shuffle/candidate順・LR0.001・50epoch・width16/heads2/layers1/context2048・CPU2threadsを固定し、原入力rawと観測事実の短縮入力compactを比較した。
compactは無関係な履歴を除く専門的な情報整理の外部化であり、教師・phase・評価器の答えを追加しない。入力長だけの効果や汎用読解能力の証明とは扱わない。

## 全6arm結果

prefix成功は48件の開始状態のうち完了した件数。invalidは訪問した入力のcontext超過であり失敗へ加算しない。最終train32/32でも最後3epochで揃わなければfit未成立。

|seed|入力|最終train|安定fit|状態判断|prefix成功件数|prefix invalid|数値初期episode|キー初期episode|
|---|---|---:|---|---:|---:|---:|---:|---:|
|42|raw|31/32|未成立|22/48|26|3|0/2|0/2|
|42|compact|32/32|成立|39/48|36|0|0/2|1/2|
|43|raw|32/32|成立|21/48|29|2|0/2|0/2|
|43|compact|32/32|成立|36/48|36|0|0/2|1/2|
|44|raw|32/32|未成立|17/48|24|2|0/2|0/2|
|44|compact|32/32|成立|33/48|33|0|0/2|0/2|

seen初期episodeは全6armで2/2。joint初期episodeは全6armで0/2。compactのキー成功はseed42/43で同じsubtract課題のみで、独立成功課題が増えたとは数えない。

|seed|入力|同step・異target：両方正答|異step・同target：両方正答|
|---|---|---:|---:|
|42|raw|6/32|9/32|
|42|compact|21/32|26/32|
|43|raw|5/32|8/32|
|43|compact|19/32|24/32|
|44|raw|2/32|4/32|
|44|compact|18/32|22/32|

対照組はケースを共有し、64独立課題とは数えない。48件は元train親2groupのanchor/jointから派生した診断probe。未知の独立課題familyの評価ではない。

## 自力継続の公平な読み方

両armとも評価可能だった同じprefixだけを比較すると次の件数となる。評価不能を除くこと自体が選択を伴うため、上表の全48件/invalid数も併記する。rawのcontext超過解消もcompactの実用上の効果だが、正答率改善とは分ける。

|seed|共通して評価可能なprefix|raw成功|compact成功|
|---|---:|---:|---:|
|42|45|26|33|
|43|46|29|34|
|44|46|24|31|

最初のteacher不一致を即失敗とは扱わない。例えば冗長な再計算後にgoalへ到達できる場合もあり、判断精度とepisode成功は別に記録する。prefixをteacherに作ってもらってからの継続は、最初から自力で解く能力とは別である。

## 失敗が教えていること

compactは3seedすべてで状態判断・対照組・共通評価可能prefixの完了件数を改善した。一方、数値変更の最初からの自力完了は全条件0/2。入力整理はこの限定判断を助けたが、未知の値への対応を解決していない。

compactのadd/numbersでは全seedがstep0でcalculatorを選ばず停止した。subtract/numbersでは計算後のstep1で、seed42/43はmemory.putの代わりにmemory.get、seed44は冗長なcalculator再実行を選んだ。計算器が計算を誤った結果ではない。モデルが訪れた正しい観測から必要な次行動を選べない失敗として保存した。引数は候補生成器が作り、モデルが生成したとは扱わない。

## 対照・検証・資源

- Rule preflight48/48、step-only16/48、first-candidate12/48。trainだけでfitしたexact候補集合頻度lookupは18/48正答、24/48をcoverageし、残りは未知集合でabstain。ニューラルcandidate-only対照ではない。
- train入力32×51×6=9,792行のaccuracy/NLL/margin/選択Actionを保存値から再計算し全曲線と一致。新旧dev104×6=624行のargmax/Actionを照合。計10,416予測行を監査した。
- 実trajectoryの連番・最後のevaluationとsuccess・全訪問入力のUTF-8 Byte/BOS/EOS・weights/RNG不変・reload・提示順・教師候補coverageを照合。旧/予約testは構築・採点0件。
- 全pytest139成功、最終メタデータ/数値guard変更後の関連13test成功、Pyright0 errors/0 warnings、diff check成功。テストは能力実測とは別。
- ChatGPT Linux CPU / Python3.12.14 / torch2.14.1+cpu / 2threads。ユーザーWSL測定ではない。#30のtorch2.14.0およびBPE基準の共有初期化とは異なるので、今回rawを#30 Byteの完全再現と扱わない。今回のraw/compact内では全初期weightが同一。
- 各arm30分deadline内に完了。RSSは両arm共通processの最大値でarm別比較に使わない。

|seed|入力|学習/reload/dev秒|更新で処理した非padding tokens|更新で処理したpadding込みtokens|
|---|---|---:|---:|---:|
|42|raw|36.33|3636600|3735300|
|42|compact|34.44|3559200|3657900|
|43|raw|36.17|3636600|3735300|
|43|compact|33.73|3559200|3657900|
|44|raw|36.05|3636600|3735300|
|44|compact|32.72|3559200|3657900|

## 次の反証可能な段階

入力方式の効果が確認できても、現行trainは固定4段階から作られており手順番号だけで解ける余地がある。次はcompact/Byte/モデル/更新予算を固定し、train内容だけを「同stepでも必要行動が異なる実状態の対照」へ変更する別protocolを設計する。候補32判断の一案は各元taskについてphase0+delay1、phase1+delay0、phase2+delay1、phase3+delay0を採用すること。step1でcalculator/put、step3でget/stopが競合し、元32判断との等exposure比較ができる。これは未採用案で、保持課題・devの重複・新family分割・教師入力の十分性を確認してから固定する。

数値/キーの一貫した役割対応をどう扱うかは別の表現仮説。状態対照train追加と値の正規化を同時変更しない。条件を変えずに古いdevで探索を続けることを独立holdoutとは呼ばない。#20の訂正→親子比較、#21の言語pilot、#29→aster-web#2の保存再生は別の達成条件。

## 成果物・公開状態

全3Runの設定・curve・token予算・evidence hash・共通prefix比較は[JSON](decision-representation-comparison-v0.json)。重みと全epoch/入力/trajectoryログはGitへ同梱せず、再読込可能な実測bundleとして別保存する。

実測sourceはローカル研究commit `4f71339`、clean tree。ソースsnapshotおよびprotocol hashを各Run/JSONへ保存。GitHub connectorでの提出はGit author/commit日時が異なるためSHAが変わるがPython内容は実測snapshotと一致し、後続の変更は報告書だけである。PR #31をbaseとするDraftとして提出し、#30の前提実装も統合している。main/Service/Web公開・model promotionは行わない。今回ユーザー側の追加作業は不要。

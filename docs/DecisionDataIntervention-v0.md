# Decisionデータ比較 v0（初回測定前の条件）

Issue #24 / PR #26の48 dev診断を受けた次の実験。元の8判断だけでfitしたモデルが数値・キー変更で崩れるため、学習課題の多様性の効果を測る。改善を保証しない。

## 固定条件

CPU / 2threads / LR0.001 / 50epochs / shuffle / TinyLM width16・heads2・layers1・context2048。
seed42で実装/fitを確認してから43・44を実行する。seedごとに4条件は同じ初期weight・同じshuffle index順列。別seedを独立した新問題として数えない。
Tokenizerは元trainのadd(2,3,total)、subtract(9,4,result)の8判断だけから生成し、全条件で固定。BPE512・min_pair_frequency1で、元fit専用Tokenizerと同じ生成条件。一般Corpus用AsterTokenizer-v0.1とは別。
元の保存weightからの継続学習ではなく、同じseedで新規初期化するデータ介入。

## 4条件

|arm|数値|キー|unique task / decision|各epochの提示枠|
|---|---|---|---|---|
|repeat-control|元のみ|元のみ|2 / 8|32（各判断4回）|
|numbers-only|元＋新|元のみ|4 / 16|32（各判断2回）|
|keys-only|元のみ|元＋新|4 / 16|32（各判断2回）|
|numbers-and-keys|元＋新|元＋新の全組合せ|8 / 32|32（各判断1回）|

新trainはadd(4,7)、subtract(12,3)、sum_slot/difference_slot。診断devの6+10、20−7、diagnostic_total/diagnostic_resultをtrainへ入れない。
2操作×2数値slot×2key slot×4phaseの32枠。反復slotに別IDを与えてもunique例数や独立群数は増えない。
各条件1,600 optimizer updates / exposures。4条件×3seed計19,200 exposures。旧50epochの400 exposuresと等予算比較ではないので、repeat-controlを新しく同じ予算で実行する。

## 分割と封印

- training-suiteはtrainだけ。派生を元add/subtractの2groupへまとめる。BPEも勾配もtrainだけ。
- development-probesは前回48件＋数値/キー同時変更8件、計56件。元train由来groupとの重なりをsplit-manifestへ明示する。これは独立holdoutではなく開発用の反例測定であり、同じ生成familyを独立分割したとは主張しない。
- 元suiteの旧testは推論/学習/Tokenizer学習に使わない。
- 新testは別の手続き生成family「計算を2/3回繰り返したprefixから保存・取得・終了」をまるごと予約する。add(31,18,reserved_sum)、subtract(46,11,reserved_difference)の全12判断を2groupへまとめ、内容とsuite digestを保存する。教師prefixは実ToolExecutorで検証するがモデルのscore/学習/Tokenizer学習に渡さない。全条件を決めてから公開判断する将来の評価用で、このPRではscored_cases=0。
- 新testも根底のcalculate_and_store課題familyは共通。手順prefixの予約であり、広いtask-family汎化を測る独立testではない。
- calibration fittingなし。test実行フラグや自動promotionもない。

## 測定と判定

train accuracy/NLL/margin、最終3epoch連続fit、reload一致、学習順/更新数を保存。
全学習が済んだcheckpointを再読込し、56 dev decisionと8 model-only episode（seen/numbers/key/joint各2）を固定評価。ruleも同じ8episodeで実行、fallbackなし。候補coverageとweights一致を検査する。評価中の更新をしない。
第一の比較対象は4条件間の変更課題episode成功率と元課題保持。decision accuracy/NLLは誤答段階の説明に使う。train fit不成立条件の失敗は汎化失敗だけに帰属しない。
遅延16件はteacher-prefixの測定のみで、遅延prefixから自力継続はまだ実装しない。候補生成器の計算引数/goal_verified供給も能力範囲に含めて記録。
改善がseed間で一致するかを見る。改善した条件が元課題や他sliceを壊した場合はそのまま報告する。test scoreを使って条件選びをしない。

## CLIと保存

```bash
.venv/bin/python scripts/run_decision_data_intervention.py --root . --threads 2 --seed 42
# 初回確認後、同じ条件の--seed 43と--seed 44を実行
```

新Runにstudy.json、split-manifest.json、training-suite.json、development-probes.json、sealed-test-suite.json、tokenizer、input-audit.jsonl、initial-model.pt、arms/*/checkpoint、learning-curve/epoch-predictions/epoch-order、dev-evaluation/diagnostics/predictions/episodesを保存する。
state_diagnosticsの測定部分を共用するが、元CLIのeight-fit checkpoint照合は維持する。fit manifestのdev=not_runは学習時点の記録で、評価は別dev-evaluation/Run eventへ保存する。Service/Webへの公開・マージ・導入は別段階。

## 初回実測

[3seed×4条件の報告](../reports/decision-data-intervention-v0.md)と[失敗/Tokenizer監査](../reports/decision-failure-audit-v0.md)を参照。小さなデータ拡張だけで安定した自力成功は確認できなかった。train fitの未成立条件とdev失敗を分け、次のTokenizer/入力比較を別条件として設計する。

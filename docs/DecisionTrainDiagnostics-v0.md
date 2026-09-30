# DecisionModel Train-Fit Diagnostics v0

## 目的と判断順序

PR #22の実験基盤に「学習終了後の同じ重みでtrain全件を評価する」観測を追加する。
`last_loss`は最後の更新前の1例のlossであり、最終train NLLではない。直近8 lossの平均も異なる重みで測った値なので代用しない。

1. 入力・候補・教師Actionの対応を確認する。
2. 最終train accuracy/NLLで、学習例を同時にfitできたか確認する。
3. trainが解けなければ1例→異なる判断を要求する2例→8例へ進み、必要なら固定順/shuffle/full-batchを比較する。
4. trainが解けてdevが低ければ、未知状態への汎化とshortcutを調べる。
5. teacher履歴上のdecision評価と、自分で行動するepisode評価を分ける。

train 8/8は小さな集合のargmax一致であり、低NLL、未知条件への汎化、広いAgent能力を保証しない。

## 実装範囲

- `training/decision_baseline.py`の既存 `_predict_cases` を再利用。eval/no_gradで測り、元のtraining modeを復元する。
- train前/後の診断はCPU RNG状態を保存・復元し、既存の固定順・単例AdamW更新を変更しない。
- `baseline.json`の`train.initial`/`train.final`、`training.json`の`initial_train`/`final_train`にexamples/correct/accuracy/nllを保存。
- `initial-train-predictions.jsonl`/`train-predictions.jsonl`に各例の候補、target/predicted Action、raw logits/probabilities、target probability、NLL、margin、step、leakage groupを保存。
- margin = 正解score - 最大誤答score。単一候補ならnull。同点時のargmax一致だけを強い学習の証拠にしない。
- train NLLはraw logitsの安定なlog-sum-expから計算し、極小確率のclipで大きな誤差を隠さない。既存dev metricsのclip仕様は変更しない。
- 評価時間と更新時間を分離。optimizer_updates/example_exposures/equivalent_epochsを記録する。現行単例方式ではupdates=exposures。
- 新規記録のresourcesにはPyTorch/Python/platformも残す。環境を跨ぐbit一致は保証しない。
- calibration/dev予測の従来ファイルは維持し、train予測を混ぜない。test推論は行わない。

## 実行

新しい学習run（従来と同じ条件）:

```bash
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 train
```

既存artifactのtrainだけを診断:

```bash
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 diagnose-train \
  --artifact runs/<development-run-id>/model-artifact
```

後者は新しいRunに`train-diagnostics.json`と`train-predictions.jsonl`を作る。
モデル/BPE更新・再calibration・calibration/dev/test推論は行わず、元artifactは書き換えない。
suite ID/digestとcandidate builderを照合する。suite全体のlineage hash確認はtestの推論ではない。
元の学習前weightsは最終artifactから復元できないため、`initial_train=null`と理由を明記する。
既存artifactを用意しない場合、このコマンドだけでは過去のユーザーPC runのtrain正解率は分からない。

## 次の実験の契約（未実装）

- 最初にユーザーPCで失敗した既存artifactを上記コマンドで測る。同一seed/2 threadsでも別CPU/PyTorchの新規runを代用しない。
- train未fitの場合、学習方法は別変更で比較し、共通初期重み・Tokenizer・データを固定する。
- 同じ25周なら単例200更新、8例full-batch25更新。全方式200更新に揃えるとfull-batchは1600例分を見るため、同じ学習量とは呼ばない。更新回数・処理例数・wall timeを記録する。
- train fitの場合、同じstepでも状態に応じて正解が変わる課題、異なるstepでも同じ状態なら正解が一致する課題、候補順変更、値/キー名の変更を調べる。
- 課題生成は実行器で正解を確認し、元課題/テンプレート派生を同じsplitにまとめる。新suiteは別版とし、旧testを学習へ移さない。
- thresholdを下げて強制実行させたり、観測だけで自動promotionしたりしない。

## 継続記録

今後この診断を変更する際も、仮説→変更条件→結果→言えること/未確定→次の実験をrepoに残す。
実測と設計案、ユーザーPCと別環境、ソフトウェアテスト成功と能力評価を区別する。
今回の結果は `reports/decision-train-diagnostics-v0.md` を参照。

# DecisionModel Fit Study v0

## 目的

Issue #23 の最初の実装。これは AsterDecision-v0 の「知能評価」ではなく、現行の教師あり Decision 学習が小さな train set を同時に fit できるかを切り分けるための train-only 実験である。

2026-09-29 のユーザーPC基準は次の通り。

- 元学習Run: `4841edd5d0d24ff5a976a1aad06963e1`
- 診断Run: `d100493770fb436baa0db2e8d65ed82a`
- artifact: `decision_model:5a57faaa98ba59ae84470c4f75ec4ac6e2b572debc8c1eebce03100ec5b00c8e`
- suite: `calculate-and-store-v0`
- train: 5/8、raw NLL 0.6999996805277703
- 対応する元Runのdev報告: 0/8
- test は封印

別環境の 8/8 / dev 3/8 は比較資料として残すが、ユーザーPCの結果と混ぜない。

## 仮説と判断順序

最初に「与えた例を fit できるか」と「未知状態へ generalize できるか」を分離する。Issue #23 は前者だけを扱い、後者は #24 へ送る。

1. serialized input / tokenized input / candidate順 / target の整合を監査する。
2. 1例だけを学習し、単一例を覚えられるか確認する。
3. 異なる判断を要求する2例を同じ初期値から学習し、同時保持できるか確認する。
4. 8例を固定順で学習する。
5. 同じ8例・同じ初期値・同じTokenizer・同じLR・同じexposureで epoch shuffle と比較する。
6. 必要な場合だけ full-batch を追加する。
7. 25epochで未fitなら、別Runとして学習予算を事前上限まで増やす。さらに必要な場合だけ learning rate を一変数として変える。

モデル拡大、データ追加、Tokenizer変更、RL、calibration調整、threshold変更、Service/Web公開はこの実験へ混ぜない。

## 事前登録するarm

`default_fit_arms()` は次を固定する。

| arm | case | mode |
|---|---|---|
| `one-fixed` | `train-add-small/step-0` | fixed single-example updates |
| `two-fixed` | step-0 + `train-add-small/step-2` | fixed single-example updates |
| `eight-fixed` | train 8件 | fixed single-example updates |
| `eight-shuffle` | train 8件 | seeded epoch shuffle |
| `eight-full-batch` | train 8件 | optional full-batch |

2例は calculator と memory.get という異なる教師Actionを要求する既知のtrain例として、実行前に固定する。成功したseedだけを後から選ばない。

全armは **train 8件全体から作った同じtrain-only Tokenizer** と **同じ初期weight** から開始する。初期stateのSHA-256を各armへ記録する。

## 学習量の比較

8件を25epoch見る場合、Wolframで算術を再確認した。

- fixed/shuffle: 200 optimizer updates / 200 example exposures
- full-batch 25epoch: 25 optimizer updates / 200 example exposures
- full-batch 200 updates: 200 optimizer updates / 1600 example exposures

したがって「等exposure比較」と「等update比較」を同一視しない。v0の既定は25epochで exposure を揃える。full-batch は任意armであり、固定順/shuffleの観測後に必要なら実行する。

## epoch末評価

epoch 0 と各epoch末に、**その時点の同じweight** で対象例を `eval/no_grad` 評価する。

保存する主指標:

- accuracy
- raw NLL
- target margin = 正解score - 最大誤答score
- 全target marginが正か
- optimizer updates
- example exposures
- online pre-update loss の平均（epoch末NLLとは別物）
- epoch内のcase順序
- update時間 / epoch末評価時間

`fit_target_met` は最後の `stability_window`（既定3）評価点すべてで accuracy=1.0 かつ全margin>0 のときだけ true とする。一瞬の8/8だけでは成立させない。

## input audit

学習前にtrainだけを使って次を保存する。

- case ID / leakage group / step
- candidate順 / target index / target Action
- candidateごとのserialized input SHA-256
- token長
- example signature

同一serialized decision inputsに矛盾した教師targetがある場合は学習を開始せず失敗させる。context length超過も事前に失敗させる。

## checkpoint

各armの最終weightを `aster-decision-fit-checkpoint-0` として保存し、直後に別load経路で再読込して最終train metricsの一致を確認する。

これは候補checkpointであり、本番用Decision artifactではない。

- calibration: `not_run`
- dev: `not_run`
- test: `sealed`
- `candidate_only: true`
- optimizer stateなし / resume非対応
- 自動promotionなし

既存 `aster-decision-model-artifact-0` を流用しない。既存schemaは calibration split で temperature をfitした inference-ready artifactを表すため、train-only診断checkpointへ偽のcalibration情報を書かない。

## 実行

最初の4arm（full-batchなし）:

```bash
.venv/bin/python scripts/run_decision_fit.py --root . --threads 2
```

full-batchも含める場合:

```bash
.venv/bin/python scripts/run_decision_fit.py --root . --threads 2 --include-full-batch
```

予算を変える場合は別Runにする。

```bash
.venv/bin/python scripts/run_decision_fit.py --root . --threads 2 --epochs 50
```

seedを変える場合も別Runにする。

```bash
.venv/bin/python scripts/run_decision_fit.py --root . --threads 2 --seed 43
```

最初はseed 42で配線と仮説を確認する。有望条件だけ、事前指定した同じseed集合（例: 42/43/44）でpaired比較する。

## Run出力

概念上:

```text
runs/<run-id>/
├── run.json
├── events.jsonl
├── study.json
├── input-audit.jsonl
├── initial-model.pt
├── tokenizer/
└── arms/
    ├── one-fixed/
    ├── two-fixed/
    ├── eight-fixed/
    ├── eight-shuffle/
    └── eight-full-batch/   # 指定時のみ
        ├── learning-curve.jsonl
        ├── epoch-predictions.jsonl
        ├── epoch-order.jsonl
        ├── summary.json
        └── checkpoint/
```

## 達成目標

能力目標は対象8例について最終的に 8/8、全target margin>0、かつ隣接する複数評価点で維持すること。NLLも必ず報告する。

ただし #23 の研究上の完了条件は 8/8 の達成そのものではない。達成できなかった場合でも、試した条件・予算・残る仮説を保存し、「次に何を一変数として調べるか」を絞れれば診断として成立する。

この実験だけから未知状態への汎化、calibration品質、Agent成功率、広い能力を主張しない。それらは #24 以降で別の物差しを使う。

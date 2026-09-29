# AsterDecision CPU Baseline v0

Issue #19の目的は、DecisionModelの最高scoreを作ることではなく、**学習済みAsterを再現・比較・監査できる実験対象にすること**です。

固定RuleBasedPolicyから一段進み、CPUだけで次の境界を成立させます。

```text
train
  ├─ Tokenizer training
  └─ DecisionModel weight update
          ↓
calibration
  └─ temperature 1 parameterだけをfit
          ↓
dev
  └─ 設計・失敗・routingを観測
          ↓
DecisionModel Artifact
          ↓
別processからreload
          ↓
設計を凍結してから test を明示的に開封
```

## なぜ二段階にするか

既存`calculate-and-store-v0`は`train / calibration / dev / test`を持ちますが、従来の統合fixtureにはTokenizer作成時にcalibration/test例まで含める経路があります。これは配線テストとしては使えても、能力比較の証拠には使いません。

Baseline v0では役割を固定します。

| split | 許可する用途 |
| --- | --- |
| train | BPE学習、DecisionModelの勾配更新 |
| calibration | 学習済みlogitに対するtemperature fittingだけ |
| dev | 設計判断、失敗観察、routing比較 |
| test | development runでは推論しない。保存済みartifactを使うfinal-testでだけ評価 |

`split-manifest.json`にはcase ID、件数、leakage group、suite SHA-256を保存します。`BenchmarkSuite`自身も同じleakage groupがsplitを跨ぐ構成を拒否します。

このsuiteは小さなsynthetic taskです。ここで高いscoreが出ても、広いAgent能力を示すとは扱いません。

## 学習するもの

初期baselineは事前学習済み言語backboneを必須にしません。

```text
scratch TinyLM backbone
        +
scalar Decision Head
```

をRuleBasedPolicy由来の`DecisionExample`でsupervised imitationします。

これは「言語能力がAgent routingへ効くか」を測る実験ではなく、**AsterのDecision学習・保存・再生・比較の経路そのもの**を固定する実験だからです。言語TinyLMとの接続は別条件として後で比較します。

## DecisionModel Artifact

Development runは`runs/<run-id>/model-artifact/`へ次を保存します。

```text
model-artifact/
├── manifest.json
├── model.pt
└── tokenizer/
    ├── manifest.json
    ├── vocab.json
    ├── merges.json
    └── special_tokens.json
```

`manifest.json`は少なくとも次を結び付けます。

- model ID / model config / Decision Head形式
- model state SHA-256
- Tokenizer SHA-256
- candidate builder ID
- suite ID / suite SHA-256
- training config / seed / initialization
- calibration temperature
- model/fallback/abstain threshold
- source Git SHA（取得できた場合）
- `resume_supported=false`

Optimizer stateは保存しないため、**推論用の再読込ができることと、学習途中からresumeできることを区別**します。

load時にはmodel stateとTokenizerのdigest、artifact identity、vocab整合を検査し、strictな`state_dict`読込を行います。

## 比較するもの

Decision-levelではaccuracyだけでなく、既存benchmarkのNLL、multiclass Brier score、ECE、risk/coverageを使います。

Agent-levelでは同じsplitから復元した初期taskについて、次を分けて実行します。

```text
RuleBasedPolicy
ModelPolicy
SelectivePolicy(Model + Rule fallback)
```

保存する主な値:

- episode success rate
- mean steps
- model / fallback / abstain route件数と率
- 個別episodeのtask_success / goal_verified / stop_reason

`model+fallback`のepisode成功をそのままモデル能力と表示しません。fallbackが救った割合を別に残します。

`candidate_set_shift`と`candidate_permutation`はDecision-levelの摂動です。Episode比較ではcanonical candidate builderを使うため、episode結果とcandidate摂動のscoreを同一視しません。

## Reload検証

Development run内で保存直後にloadし、dev caseのscore差を確認します。

さらにtest suiteでは**新しいPython process**を起動し、同じartifactをloadして、同じcaseのscoreとargmax Actionを比較します。これにより「同じprocessのPython objectが残っていたから再現した」だけではないことを確認します。

## CPU計測

実行時に次を保存します。

- `torch_num_threads`
- Tokenizer / train / calibration / dev decision / dev episodeのwall time
- 1 train stepあたり平均wall time
- process CPU time
- Linux `ru_maxrss`（KiB）

`ru_maxrss`はprocess lifetimeの最大値であり、学習区間だけの差分ではありません。

学習step予算は手元の実測から決めます。例えば30分を更新だけに全て使うという**上限計算**なら、平均step時間が0.25/0.5/1/2/5秒のとき、それぞれ7200/3600/1800/900/360 stepです。評価・保存時間を差し引いていないため、これは実測結果ではありません。

## 実行

WSL2/LinuxのAster rootで、まずtestを封印したdevelopment runを実行します。

```bash
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 train
```

出力されたRun directoryの`baseline.json`、`training.json`、`development-predictions.jsonl`、`model-artifact/manifest.json`を確認します。

設計条件を固定した後だけ、同じartifactを指定してtestを開封します。

```bash
.venv/bin/python scripts/run_decision_baseline.py --root . --threads 2 test \
  --artifact runs/<development-run-id>/model-artifact
```

final-testは保存済みweights、保存済みTokenizer、保存済みtemperature/routing設定を読み、モデル更新や再calibrationを行いません。

## 合格条件

Baseline v0の成功条件は高accuracyではありません。

1. **Leakage-safe**: BPEとgradient updateはtrainのみ、temperatureはcalibrationのみ、development時にtest推論をしない。
2. **Artifact-safe**: weightsだけでなくTokenizer・config・候補生成・split/suite lineageを保存する。
3. **Reload-safe**: 別processでも同じscore/Actionを再現できる。
4. **Evaluation-safe**: Decision accuracy、episode success、fallback、失敗種別を混同しない。
5. **Resource-known**: 実行したCPU環境のstep時間・評価時間・memoryを測って残す。

accuracyが低くても、この5条件が成立して原因を追えるならIssue #19の実験基盤としては有効です。

## この段階で含めないもの

- aster-webからのmodel選択・実行
- ServiceのDecisionModel artifact catalog
- raw upload
- Interventionによるcandidate再学習
- RL
- automatic model promotion
- optimizerを含むtraining resume

次は、このartifactとbaseline evidenceをServiceのallowlisted recipeへ公開し、aster-web #2からRun比較を観測する段階です。

## 2026-09-29: Train-Fit Diagnostics追加

学習前後のtrain全件raw評価と既存artifactの`diagnose-train`コマンドを追加した。
`last_loss`だけでfit成功を判断しない。学習順序・学習率・データ・thresholdは変更していない。
仕様・実行方法・次の実験条件は[DecisionTrainDiagnostics-v0](DecisionTrainDiagnostics-v0.md)、実測は[診断結果](../reports/decision-train-diagnostics-v0.md)を参照。

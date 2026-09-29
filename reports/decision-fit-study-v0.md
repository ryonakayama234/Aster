# DecisionModel Fit Study v0 — 実測記録

## 状態

2026-09-29時点: **実験runner実装済み、ユーザーPC実測は未実行**。

このファイルは Issue #23 の結果記録を、実装・テスト成功・ユーザー実機実測・能力解釈で混同しないための起点として置く。実測前に結果を埋めない。

## 比較基準

ユーザーPCで既に確認済みの基準:

- 元学習Run: `4841edd5d0d24ff5a976a1aad06963e1`
- 診断Run: `d100493770fb436baa0db2e8d65ed82a`
- artifact: `decision_model:5a57faaa98ba59ae84470c4f75ec4ac6e2b572debc8c1eebce03100ec5b00c8e`
- suite: `calculate-and-store-v0`
- train: 5/8、accuracy 0.625、raw NLL 0.6999996805277703
- 対応する元baselineのdev報告: 0/8
- test: sealed

別環境で得られた 8/8・NLL 0.669470・dev 3/8 は別証拠として扱い、ユーザーPC結果へ混ぜない。

## 最初に実行する条件

```bash
.venv/bin/python scripts/run_decision_fit.py --root . --threads 2
```

既定:

- seed: 42
- epochs: 25
- learning rate: 3e-3
- model / Tokenizer設定:現行baselineと同じ
- arms: one-fixed / two-fixed / eight-fixed / eight-shuffle
- full-batch: 未実行
- calibration/dev: 未実行
- test: sealed

## 実測後に記録する項目

- Run ID / source git SHA / Python / PyTorch / thread数
- input auditが通ったか。矛盾・context超過の有無
- 各armのepoch 0 / 最終 accuracy、raw NLL、min target margin
- `fit_target_met`
- optimizer updates / example exposures
- update時間 / epoch末評価時間 / 全経過時間 / max RSS
- checkpoint reload一致
- fixedとshuffleの学習曲線・個別誤答の差
- 観測から言えること / まだ確定できないこと

## 次条件の決定規則

1. 1例がfitしない: 配線、loss、gradient、optimizer、入力表現を先に診断する。
2. 1例fit・2例fail: 複数判断の同時保持/更新干渉を優先して調べる。
3. 2例fit・8例fail: fixed vs shuffleを比較し、順序依存・予算不足を切り分ける。
4. fixed/shuffleで原因が切れない場合だけ `--include-full-batch` を別Runで実行する。
5. 25epochで未fitなら、事前上限を決めた追加epochを別Runで行う。
6. さらに必要な場合だけlearning rateを一変数として変更する。
7. 8/8になっても未知状態への汎化とは解釈せず、#24へ進む。

## 未確認

- PR #25のrepository pytest / Pyright
- ユーザーPCでの新runner実行
- 既存元artifact manifestと手元保存先の再照合
- full-batch結果
- seed 42/43/44のpaired比較

上記が未確認の間、このreportは能力改善の証拠ではない。

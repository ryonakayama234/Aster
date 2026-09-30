# DecisionModel Train-Fit Diagnostics v0 実測

## 仮説と変更

2026-09-29、PR #22 / base `b837b79bee36bb4b3d3b0a544c5acf2714b7659b`。
最終train全件を測らずlast_lossで判断すると、fit失敗と汎化失敗を区別できない。
今回の変更は観測と既存artifactの読み取り診断。固定順・単例AdamW・200更新・seed42・2 threadsは変更していない。
実行時のHEADは上記baseで、診断追加は未コミット差分だった。実装の正本はPR #22の本報告を含むコミット。

## 実測環境と証拠

ChatGPT実行環境、Linux x86_64、Python 3.12、PyTorch 2.14.0+cpu、2 threads。
ユーザーのRyzen/WSL環境の再現結果ではない。環境詳細・config・digest・個別予測は[JSON証拠](decision-train-diagnostics-v0.json)に保存。

- 診断追加run: `2b18caaa52194b0aa3d590566821f595`
- 変更前コードの対照run: `7f75ec71404f4707af74b7885b200895`
- 既存artifact診断run: `04898423458c44a2bcc24cd3680d7142`
- testは両runとも封印。test正解率は未測定。

| 指標 | 初期train | 最終train | dev (raw) |
| --- | ---: | ---: | ---: |
| 正解数 | 3/8 | 8/8 | 3/8 |
| 平均NLL | 1.355008 | 0.669470 | 1.442730 |

最終online lossは0.542601で、最終train NLLとは異なる。
trainのmemory.get例は正解だがNLL約1.24〜1.26で、8/8を高確信と同義にはできない。
calibration temperatureは約1.563676。
dev episodeはrule 2/2成功、model 0/2、model+fallback 0/2。

## 観測の追加が学習を変えていないか

同じ環境で変更前のbase commitを別worktreeから実行した。
診断追加runと、artifact ID・model SHA256・全online loss列・temperatureが一致した。
既存artifactの`diagnose-train`も最終train指標と個別予測を再現した。
この対照は今回の環境/条件の証拠であり、全環境でのbit一致を保証するものではない。

## 検証

- 全pytest: **104 passed**。
- Pyright（実行venvを明示）: **0 errors / 0 warnings**。
- train JSONLと集計値の整合、既存`evaluate_decisions`とのNLL/accuracy一致を検証。
- diagnostics中のtraining/BPE/calibration呼び出しを禁止し、推論splitがtrainだけであることを検証。
- 元artifact全ファイルのSHA256不変、別Runへの出力、suite不一致の拒否を検証。
- 極端な誤答logitでもNLLをclipしないこと、mode/RNG/weightsを変えないことを検証。
- 新CLIの`diagnose-train`を実行し、保存済みartifactから8/8を再確認。

## 言えること・未確定

この環境では現行training protocolでも8例のargmax一致まで到達するため、ここでのdev失敗を「trainを覚えること自体が不可能」とは説明できない。
一方、NLLはまだ高く、最適化改善の余地がないとまでは言えない。状態/履歴の使い方やshortcutは未確認。
ユーザーPCのdev 0/8 runの最終trainは未測定。ユーザーPCの保存済みartifactを受け取っていないため、この別環境runからその原因を確定しない。

## 次の実験

まず失敗したユーザーPCのartifactに`diagnose-train`を実行する。
train未fitなら1例/2例/8例と更新方法比較へ、train fitなら状態とstepを切り分ける課題設計へ進む。
今回の環境では後者を支持するが、データ拡張・shuffle/full-batchの本番導入・閾値変更・test開封は行っていない。
設計と継続記録の契約は[DecisionTrainDiagnostics-v0](../docs/DecisionTrainDiagnostics-v0.md)。

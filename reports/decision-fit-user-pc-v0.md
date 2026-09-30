# ユーザーPC Decision fit実測（2026-09-30）

## 証拠と確認

ユーザー提供の5 Run ZIPを再検証。[機械可読要約](decision-fit-user-pc-v0.json) にRun/設定/初期重み/Tokenizer/checkpoint/ファイルhashを残す。ZIP原本・model.ptはGitへ同梱しない。
Python3.12.3、PyTorch2.14.0+cpu、WSL2、2threads、source SHA `0fa5736e30d38c6569a1c6419e774d16748235f7`。
全epochのaccuracy/NLLを個別予測から再計算。checkpoint model.ptのhashもmanifestと一致。reload結果はユーザーrunnerの記録。
後続の状態診断では8例の全6checkpointをChatGPT Linux CPU環境の別Python processで読み込み、既知8/8を再確認した。ユーザーPCの速度とその環境の速度を混ぜない。

## 条件の推移

| Run | seed | LR | epoch | fixed最終 | shuffle最終 |
|---|---:|---:|---:|---|---|
| f2f49aab1e524a469439b92a86096c7a |42|0.003|25|5/8、NLL0.700000|6/8、NLL0.554692|
| f4e435a8fe734ad2b54e24da26795cef |42|0.001|25|6/8、NLL0.432956|8/8、NLL0.328935、安定条件未達|
| 96d2e995bf15464c9733b618f7e7da69 |42|0.001|50|8/8、NLL0.010364|8/8、NLL0.006287|
| c947381d2b6341fea2ed5d5a60ccb175 |43|0.001|50|8/8、NLL0.287429|8/8、NLL0.143352|
| 058b98770b9e4f51a457e3fd5fbdb206 |44|0.001|50|8/8、NLL0.009343|8/8、NLL0.008006|

seed42のLR0.001・50epochは、25epochまでの26評価点を完全再現。8例×50epoch=400 updates / 400 exposures（Wolfram算術確認）。
50epochの12arm（3seed×one/two/eight-fixed/eight-shuffle）すべてfit_target_met=true。
ただし2例armは最小marginがseed42=0.002617、43=0.003105、44=0.055764と弱い。

| seed | fixed最小margin | fixed最終まで連続8/8 | shuffle最小margin | shuffle最終まで連続8/8 |
|---|---:|---:|---:|---:|
|42|5.067814|16評価点|5.258492|27評価点|
|43|0.459823|5評価点|1.021330|32評価点|
|44|5.273060|16評価点|4.656596|23評価点|

各Run wall timeは33.739 / 34.066 / 34.466秒、最大RSSは約302MiB。4armと評価/保存を含む実測。

## 解釈

現行のモデル/入力表現で、この2 trajectory由来8 decisionを3seedで安定fitできた。LR0.001・50epoch・shuffleを次の診断の暫定基準にする。seed43を除外しない。
固定順も50epochで成立するため、shuffleだけが唯一の解決策ではない。LR0.003×50epochは未測定で、元の失敗を単一原因へ確定しない。
24件の独立問題を解いたという主張ではない。同じ8decisionを3seedで学習した確認。言語能力、未知状態への汎化、Agent完了率、calibration品質とは分ける。
calibration/dev未実行、test sealed、checkpointはcandidate_only・resume非対応。旧artifact/Runを上書きしない。

## 次

#23の実測診断が成立したので、モデル拡大・RL・追加epoch調整をここでは進めない。
#24の[状態診断](decision-state-diagnostics-v0.md)へ進み、保存モデルを更新せずshortcut/新条件を測る。実装のpytest/PyrightはPRの検証記録で別に示す。Issueのclose・モデルpromotionは行っていない。

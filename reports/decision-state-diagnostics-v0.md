# Decision状態診断 初回実測（2026-09-30）

## 環境・範囲

ユーザーのLR0.001/50epoch、seed42/43/44のeight-fixed/eight-shuffle、計6checkpointを別Python processで読み込んだ。
実行はChatGPT Linux CPU / Python3.12 / PyTorch2.14.0+cpu / 2threads。ユーザーWSLでこの新CLIを実行した結果ではない。
全checkpointのidentity/model/Tokenizer hashをload時検査。評価前後に全weight tensorが一致。学習・calibration・test推論なし。
[機械可読報告](decision-state-diagnostics-v0.json) に診断Run ID・元checkpoint ID/hash・suite digest・slice・episode結果を記録。

## Teacher-prefixの判断結果

| seed | arm | seen /8 | 逆順 /8 | 数値変更 /8 | キー変更 /8 | 到達可能遅延 /16 |
|---|---|---:|---:|---:|---:|---:|
|42|fixed|8|8|2|1|4|
|42|shuffle|8|8|3|2|5|
|43|fixed|8|8|0|0|7|
|43|shuffle|8|8|2|4|7|
|44|fixed|8|8|0|1|6|
|44|shuffle|8|8|3|1|6|

正解candidate coverageは全48caseで100%。誤答を候補欠落へ帰属できない。
step-only対照はseen/numeric/key/permutationで各8/8、遅延で0/16。first-candidate対照は各8件sliceで2/8、遅延4/16。
元の4段階課題はstepだけで正答可能という構造を実測した。モデルがstepだけを使っているという証明ではない。遅延probeではモデルがstep-onlyを上回るが、state・history・noiseが同時に変わるため特定fieldの因果効果を確定しない。
候補逆順では選択Actionが全6モデルで一致。独立candidate scorerの構造から期待される結果であり、状態理解の証明ではない。

## 自力episode（fallbackなし）

全6モデルで、seen2taskは2/2成功、numbers2taskは0/2、key2taskは0/2。ruleは同じ6taskで6/6成功。
max_steps=8、terminalかつgoal_satisfiedを成功とし、stopped_without_goal/step_limitを分けて保存。
decision精度の分母48とepisode成功率の分母6を混ぜない。teacherが作った成功履歴での判断と、自分の行動で作った履歴での結果も分ける。
モデルの実際の訪問状態でteacher候補coverageも100%。成功episode/失敗episodeのtrajectoryは元Runに保存。
遅延prefixからの自力継続は未実装であり、上表の遅延精度をepisode成功率として扱わない。

## 言えること

学習例のfitと元課題の自力実行は成立した。一方、この小さな数値/キー/履歴変更には弱い。元trainの高確信だけで未知条件への汎化を保証できない。
同じ元課題由来の派生をgroup2件へまとめたdev診断で、独立holdout/広い能力のscoreではない。各seedを独立した新taskとして数えない。
candidate builderが計算値やgoal_verifiedを供給する範囲も能力主張から区別する。

## 次の最小研究

1. 保存predictions/trajectoryから、どのActionへ誤選択するかを分類。数値・キーの変化に伴うtoken分割/入力長の変化も点検する。
2. 少量のtrain拡張と新split manifestを事前定義する。診断devや旧testをそのままtrainへ移さない。新しいテンプレートfamilyのholdoutは別版として設計する。
3. 固定モデル/Tokenizer/学習予算で変更データの効果を測る。Tokenizerや入力表現の変更が必要なら独立条件にする。
4. train fit、dev decision、model-only episodeを別々に比較し、親子Artifact/Runを保持する。

## 検証

- pytest: 111 passed。新規4testは履歴の実行再生、step反例、read-only・保存出力、checkpoint範囲拒否を確認。
- Pyright: 0 errors / 0 warnings。repo設定と同じinclude/strict/pythonVersionを一時configへ反映し、検証venvとsrc探索pathを明示。
- PR #25由来の_stable_fitでobjectをintへ渡す型エラーも修正。epochがintであることを確認するようにし、既存のfit testsも通過。
- ユーザーPC上での新CLI実行、GitHub CI、Service/Web公開は未確認/未実装。stacked PRで既存main向けCIを自動実行済みとは扱わない。

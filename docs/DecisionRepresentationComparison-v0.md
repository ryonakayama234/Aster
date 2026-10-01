# Decision raw/compact comparison v0

2026-10-01 JST。Issue #24。PR #30 (065882e) と PR #31 (c5624af) の研究実装を統合して使用。両PRは未merge。実験はService/Webへ公開しない。

## 考えたこと・仮説

Byte入力では32判断のfitが3/3seedで成立したが、数値変更の自力episodeは0/2だった。次は観測可能な状態/履歴から必要情報を抽出する負担を切り分ける。仮説は「無関係な履歴と冗長な表現を除くと、限定的な状態判断が改善する」。raw/compactで差がない、悪化する、学習例だけ改善する結果も残す。情報の選別をプログラムへ外部化する比較であり、系列長だけの純粋な効果や汎用読解を証明する実験ではない。

## 方法・固定条件

- #27 numbers-and-keysの32判断を固定。新しい状態dev48件をtrainへ混ぜない。
- 2 arm: raw=aster-decision-input-0、compact=aster-decision-compact-input-0。compact契約はDecisionStateRepresentation-v0.md。教師/phase/評価器の正解を入力へ追加しない。
- Byte語彙258、BOS256/EOS257。width16/heads2/layers1/context2048、LR0.001、AdamW既存defaults、単例shuffle、50epoch、CPU2threads。
- seed42→43→44。seed内では全初期weight・提示順・候補順が同一。条件はdebugで配線確認後に変更しない。
- 32×50=1,600更新/提示/arm、6arm=9,600。epoch0を含むtrain予測9,792行。Wolframで算術照合。token量・時間は等しくない。
- tokenizer再学習・calibration・fallback・モデル採用なし。旧/予約testは構築・長さ確認・採点もしない。
- train/dev/既存56dev入力の復元、context、model-visible教師衝突を両arm学習前に検査。超過なら全比較を止める。model訪問状態でcontext超過ならそのepisodeをinvalidとし成功/失敗に混ぜない。切詰めなし。
- 各arm学習＋保存＋reload＋devを30分以内。timeoutはRun失敗として記録、片側予算延長なし。途中結果と最終50epoch結果を混ぜない。

## 評価・達成目標

1. epoch0/各epochのtrain accuracy/候補NLL/min target margin、最後3epochの全件正答＋正marginをfit成立とする。
2. 48devのteacher-prefix判断と、同step異target32組・異step同target32組の両方正答率。組はcaseを共有し、64独立課題とは数えない。
3. 48prefixから最大8追加Actionのmodel-only継続。最初のteacher不一致、全候補scores/入力/token IDs、訪問state、実trajectoryを保存。teacherと異なるActionでもgoal達成する可能性があるため最初の不一致と実際の失敗を同一視しない。
4. 元の56dev判断・8初期episode（seen/numeric/key/joint各2）も保持。旧課題の保持と変更課題を分ける。
5. Rule preflight、step-only16/48、first-candidate12/48、trainだけでfitする候補集合頻度lookupを対照とする。lookupはexact Action集合ごとの教師頻度で予測し、未知集合でabstain。tieはcanonical Action順。ニューラルcandidate-onlyの学習結果ではない。dev教師で得たoracle上限36/48は学習済み対照にしない。PR #31のニューラル候補のみ対照案を、この小さなlookup対照へ変更し本測定前に明記する。
6. weights/RNG不変、reload metrics一致、serializer mismatch拒否、入力衝突・候補coverage・順序同一を検証する。

dev48件は元train親2groupの診断probeであり独立holdoutではない。数値/キーだけを独立に変えるprobeは元56devで確認する。新48devはanchorとjointなのでその集計から数値/キー単独効果を推論しない。3seedを独立課題数へ加算しない。正答率改善を保証せず、全armの比較可能な証拠・CPU費用・次の判断が残れば完了。

## 次の分岐と役割

- compactが状態対照と継続で一貫改善: この限定課題で外部化された情報整理が役立つ。新しい到達可能状態をtrainへ入れる別protocolと構造family分割へ進む。
- 両方fitするがdev失敗: 学習課題のshortcut、状態対照組をtrainへ導入する仮説を次に検証。入力比較だけで状態学習の十分性は確認できない。
- teacher-prefix成功でも継続失敗: 訪問状態と誤り伝播を診断し、#20の訂正→親子比較へ渡す。
- train未fit: データを固定した最適化診断へ戻る。モデル拡大とデータ追加を同時にしない。

エージェントが実装・検証・測定・記録を担当。今回ユーザーの追加素材/長時間PC学習は不要。ユーザー側接続や本人の選定が必要な段階では目的・手順・返してほしい成果物を提示する。観測#29→aster-web#2、言語#21は別完成点で進め、今回両者の実装完了とは扱わない。

# Aster Decision：問題探求と反証可能な問い v0

2026-10-01。研究メモ。実装済み能力・学習結果・次の提案を区別する。
参照する実装はPR #33のsource `6b851a06c777464a9f601753359462c0864f7413`、
記録込みhead `3121cbecc31308a3e16a727de5c6349b9fb2d400`。
本メモは追加学習の事前登録でも、新しい独立holdoutの報告でもない。

## 着手前の考え方・作業方法・達成目標

考え方：悪化したデータ置換実験を「状態学習は不可能」と一般化せず、
成功の意味、学習例の除去、表現への依存、自力実行の失敗を競合仮説として分ける。
Aster全体の到達点は、道具の観測から次の実行可能な行動を選び、検証して完了すること。
一般言語Corpusの能力とDecision課題の能力は別に扱う。

作業：既存実測とコードを読む → 一次研究の診断方法を照合する →
実行器・教師・評価器の意味を小さな実履歴で確認する → 問いと反証条件を記録する。
学習条件、既存教師、採点、公開UIは変更しない。

達成目標：次の測定で「どの説明を弱められるか」が明確な優先順を作る。
停止条件：それぞれに対照・限界・次の分岐が書けたら、研究メモを閉じる。
数値改善やモデル拡大を今回の達成条件にはしない。

## 観測されたこと／まだ言えないこと

- 短縮入力はraw入力より既存の状態判断を改善した。
  しかし対象task・対象キー・関連履歴を外部コードが抽出する課題専用表現であり、
  モデル自身が汎用の履歴読解を獲得した証拠ではない。
- PR #33では正常calculator/put例16slotをwrong-write状態例へ置換した。
  state条件の新状態判断・原状態判断は3seedともbaselineより低下した。
  正常例の除去、誤値変更、履歴長変更を分離できていない。
- fresh trainingであり、逐次学習の意味での「破滅的忘却」と呼ばない。
  seed43のstate条件は安定fitしなかったので、全失敗を汎化だけに帰属させない。
- 数値変更の初期episodeは全6条件0/2。2課題から一般的な成功率を推定しない。
  診断のseed数・派生数を独立task数へ加算しない。
- Decision比較のTinyLMはランダム初期化から学習するcandidate scorer。
  一般Corpusで獲得した言語能力を継承する比較ではない。
  行動引数はcandidate builderが作り、モデルが算術や自由引数生成を担う評価でもない。

実測詳細：[状態train結果](../reports/decision-state-training-v0.md)、
[入力表現比較](../docs/DecisionRepresentationComparison-v0.md)。

## 問い0：完了とは「過去に達成」か「停止時点でも正しい」か

追加学習より先に確認する仕様の問い。現在の教師は履歴中の正しいput/getの存在を調べ、
評価器は一度立ったgoal_satisfiedを後続へ保持する。

実行器で次を実行する意味確認を行った。モデル学習・モデル採点は行っていない。

```text
task: add(2, 3), store_as="total"
calculator → memory.put(total, 5) → memory.get(total)
→ memory.put(total, -999) → 教師が選んだstop("goal_verified")
停止時memory: {"total": -999}
goal_satisfied: true
terminal: true
```

再現（リポジトリroot、Python標準環境、PYTHONPATH=src）：

```python
from aster.benchmark.state_representation import make_example, replay, execute
from aster.agent.policy import RuleBasedPolicy
from aster.records.transition import Action

example = make_example(
    {"kind": "calculate_and_store", "operation": "add",
     "left": 2, "right": 3, "store_as": "total"},
    phase=3, delay=0,
)
context, recorder, tools = replay(example)
execute(context, recorder, tools, Action.tool("memory.put", key="total", value=-999))
history = recorder.trajectory()
state = context.snapshot(len(history.transitions))
action = RuleBasedPolicy().decide(
    state, history, tools.registry.names() + ("stop",)
)
last = execute(context, recorder, tools, action)
print(context.snapshot(len(recorder.trajectory().transitions)).memory)
print(action)
print(last.evaluation)
```

これは達成を履歴上の出来事とする現契約なら整合している。
将来の復帰課題で停止時点の正しさを要求するなら別契約が必要。
PR #33にはこの達成後上書きprefixがなく、既存結果を無効とする根拠ではない。
推奨する次版の案は「停止時点で正しい保存値があり、最後の対象キー変更後に
正しい取得が確認されたこと」。ユーザーの意図との一致を確認してから別版で扱う。
本メモでは教師・評価器を修正せず、旧結果を新契約で再採点しない。

## 優先する競合仮説と見分け方

|順|問うこと|最小の対照・観測|その説明を弱める結果／次の分岐|
|---|---|---|---|
|1|正常経路を学習から外した影響が主か？|まず保存済みモデルで、既知taskの空履歴・正常計算直後とwrong-write状態を別集計する|正常経路は保たれ、新状態だけ崩れるなら除去だけでは説明できない。除去が疑わしい場合のみ、元例を保持する混合学習を別protocol化する|
|2|誤値の文字列か、履歴の回数か、両方か？|同じtask・段階で誤値-999/-777と誤保存1/2回の2×2を交差する|値のみで崩れる／回数のみで崩れる／交互作用／全部崩れるを分ける。差がなければ今回の2因子説明を弱める|
|3|数値・キーの名前を覚えたのか、役割と関係を使ったのか？|task・実観測・候補を整合的に変換する。対象キーの一貫した改名、計算済み/未計算の状態対を別評価する|同じ役割で名前だけ変えた場合に保てれば名前依存説を弱める。数字変更では算術結果と候補も変わるので単純な同一意味変換とは呼ばない|
|4|必要な事実の抽出が難しいのか、その事実からの行動選択が難しいのか？|未実装の診断案：実行時に観測可能な事実の明示表現とcompactを対照する|明示表現だけ成功なら表現を疑う。どちらも失敗なら読解だけでは説明できない。正解Actionを入力する対照は能力評価にしない|
|5|一判断の誤りか、自力で訪れる状態の偏りか？|初期状態・teacher-prefix・model-only継続を分け、最初の誤りとその後の訪問状態を記録する|初期判断から失敗するなら自力実行の分布変化だけでは説明できない。初手は正しく後で崩れる場合、復帰例を採る根拠になる|
|6|入力の位置や長さに依存しているか？|意味・候補を保つserializer変換を別版で定義し、入力長と候補ごとのEOS位置を記録する|意味不変でも予測が変わるなら表現への依存を支持。ただし長さ・書式・位置が同時に変われば位置符号化だけの原因とは断定しない|

問い4の明示表現は、候補名ではなく「対象キーが存在する」「観測した計算結果と一致する」
「どの書込みより後の取得か」などの事実を想定する。教師Actionから逆算して作らない。
計算結果を外部道具で得ることはAsterの設計に沿うが、
その事実抽出を外部へ移した範囲を能力の帰属として明記する。

## 一次研究から借りる考え方

1. Geirhos et al., *Shortcut Learning in Deep Neural Networks* (2020).
   [論文](https://arxiv.org/abs/2004.07780)
   学習内の正答だけでは意図した手掛かりを使うと確定しない。
   Asterへの適用は、名前や経路を変えた対照と簡単なbaselineで依存を調べる方法。
2. Lake & Baroni, *Generalization without Systematicity* (2018).
   [論文](https://proceedings.mlr.press/v80/lake18a.html)
   学習例に近い成功と要素を組み替える汎化を区別する。
   SCANのRNNの結果がTinyLMの失敗原因を証明するわけではない。
3. Battaglia et al., *Relational inductive biases, deep learning, and graph networks* (2018).
   [論文](https://arxiv.org/abs/1806.01261)
   対象・関係を表す構造が学習の前提を変えるという設計視点。
   Asterへの提案は事実表現の診断であり、graph network導入の決定ではない。
4. Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction
   to No-Regret Online Learning* (2011).
   [論文](https://proceedings.mlr.press/v15/ross11a.html)
   行動が次の入力を変えるため、教師の状態だけの学習と自力実行を区別する。
   DAggerは後続候補であり、初期判断の失敗を直ちに解決する保証ではない。
5. Press, Smith & Lewis, *Train Short, Test Long: Attention with Linear Biases Enables
   Input Length Extrapolation* (2022).
   [論文](https://arxiv.org/abs/2108.12409)
   長さの外挿と位置表現を診断対象に含める根拠。
   TinyLMの学習済み絶対位置embeddingとEOS読出しへの原因帰属は未確認。
   大型言語モデルの結果をAsterへ直接転用しない。

## 次の一段：学習せず失敗の境界を測る案

問い0を仕様確認として扱いつつ、現契約の範囲で問い1・2を先に測る。
新しい復帰仕様を既存測定へ混ぜない。

- 保存済み6checkpointを固定し、更新回数0。
- 既知8task × calculator/putの2段階 × 誤値2種 × 誤保存回数2種
  = 64判断/model、合計384判断。正常経路対照は別枠で件数を登録する。
- Wolframで因子の件数を確認した。計算の一致はモデル能力の証明ではない。
- 各入力を6armのtrain入力unionと照合し、同一入力の保持確認と新入力を分ける。
- 同じ生成familyの診断であり独立holdoutとは呼ばない。
- 主効果・組合せ別の誤り・正常経路保持を件数で示す。
  6checkpointの結果を独立task384件と扱わない。
- 冗長計算prefixの予約test familyには触れない。
- 結果を見てから条件・除外を変更しない。変更が必要なら別版にする。

これは未実装・未測定の後続案。着手時にprotocolへ完全な生成条件、
候補・入力署名・測定指標・停止条件・結果別の分岐を固定する。
問い3以降を同じ測定へ詰め込まず、問い1・2で説明できなかった範囲から進める。

## ユーザーの役割とAster全体との接続

今必要な本人の判断は「一度達成すれば成功か、停止時点にも保存値の正しさが必要か」。
推奨は後者だが未合意の設計案として残す。今はユーザーPCでの追加学習や資料収集を依頼しない。

実装・実行・記録・反証対照の準備はこちらで担当する。
モデル側の目的・教師・Evaluatorの意味を先に揃える。
aster-webはそれらの実測とTraceを観測する側として捉え、
今回UI・API・公開recipe・採用モデルの変更は行っていない。
言語pilotと行動診断は別の成果として続ける。

## 2026-10-01 18:44 JST：問い0へのユーザー合意

「停止時点でも正しく保存され、最後の変更後に取得確認できている」を採用する。
上の未合意という記述はメモ作成時点の履歴。[完了契約v1](CurrentStateGoal-v1.md)を
新しい正本とし、旧契約・旧結果と版を分ける。ユーザー側の追加作業は不要。

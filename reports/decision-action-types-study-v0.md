# 観測特徴→行動種類：全3seed診断 v0

2026-10-02 JST。[事前条件](../docs/DecisionActionTypes-v0.md)、[正本JSON](decision-action-types-study-v0.json)。

## 結論・能力の境界

全3seedでtrain stable fit、128状態のAction種類/結合Action一致、全32対照組の両方正答、
全128prefix自力継続、各variant初期2episodeと追加4初期task完了が成立した。
**外部抽出済みの観測事実から、学習器が行動種類を選ぶ足場は成立。**
学習器は数値/キー/履歴文字列/candidate引数を読まない。実行・算術・equality・時系列抽出・
引数供給は外部実装。特徴空間では評価128/128と追加4/4がtrainと重複する。
従って未知の状態規則を発見した、生履歴を読めた、一般会話/コード生成ができたとは扱わない。

## 固定条件・全結果

48raw train/64slotを前比較から継承。C/M/F3特徴→4Actionのnn.Linear、16parameter、
ランダム初期化、AdamW lr0.01、25epoch=1,600単例更新/seed、CPU2threads、seed42/43/44。
Action種類だけを教師にし、選んだ種類を既存candidateへ結合。rule fallbackなし。
前の文字列candidate scorerとは入力/モデル/出力分解/LRが異なる複合診断で、純粋比較ではない。

|seed|stable fit|train /48|種類/結合Action /128|対照両方 /32|prefix完了 /128|数値/キー/同時初期 各/2|新初期 /4|
|---|---|---:|---:|---:|---:|---|---:|
|42|成立|48|128|32|128|2/2、2/2、2/2|4|
|43|成立|48|128|32|128|2/2、2/2、2/2|4|
|44|成立|48|128|32|128|2/2、2/2、2/2|4|

既知初期も各seed2/2。binding failure/範囲外Tool failure invalidは全seed0。
最終train最小正解marginは2.7641/2.7960/2.6065。最終3epoch全正答+margin>0をstable fitとした。
追加taskは学習前にadd(123,-45)/add(-7,0)/subtract(100,123)/subtract(-4,-9)と新キーを固定。
同じtask family内の数値/名前移送で独立holdoutではない。

## 特徴の重複が意味すること

raw48判断は4種類のC/M/Fへ写る。128devと追加初期4taskもその4種類へ写り、
features→teacher typeの衝突0。入力の整理で同じ意味の判断に変換する責務を外部化した。
全32対照組はraw状態が異なっても、行動に必要な違いが特徴へ残ることの確認。
前実験の「非重複24組」と同じ能力/分母の測定ではない。高scoreを広い汎化の証拠へ使わない。
stop理由のgoal_verified候補がある場合の優先、計算済み値のput引数は既存builder/binder側。
モデルがその条件や値を生成したわけではない。candidate欠落時は別種別へ置換しない。

## 検証・source・実行資源

- 全pytest175 passed（72.47秒）、新規関連5 passed、Pyright0 errors/0 warnings、diff check。
- feature metadata非依存、候補逆順/欠落/同種類複数の拒否、Tool失敗範囲外、checkpoint contract拒否を確認。
- 手設定の線形weightで128ケースを表現できるテストは仕様確認で、学習モデル測定と別。
- 1epoch debug dbe83c5ca974468e9671edce081f8624はstable-fit判定に必要な3epoch未満。
  dev/prefixは全成功でも正式3seedへ加算しない。debug保存全記録を独立監査。
- 正式Run 212b32fd029640ce8efc89930b5c4403、clean local0817a1d341bf88ca595f62fd36b4fdc3132cba9d、dirty=false。
  API measurement source d6f4bc6ec18216320219d01fe44b5fb23ab097f9はtree d1fcdffec270464f4ced251bef712520ae101f68がlocalと一致。
- モデル保存/reloadでtrain/dev全score一致、評価中weights/RNG不変を全seed確認。
  初期weight/全epoch順/全予測/最終weight/契約とsource/protocol hashを保存。optimizer未保存でresume非対応。
- 独立cached audit: train3744/dev384/episode396/transition2088/訪問1152行を照合。
  全Actionを実Tool再実行し、停止時memoryと最終対象put後getを独立計算。スコア/特徴/分類/NLL/marginも検査。
- 正式測定後に測定時全Python file hashを照合し変更0。
- ChatGPT Linux/Python3.12.14/torch2.14.1+cpu/CPU2threads、正式wall4.464秒。
  全pytestが同時実行中だったため精密な速度ベンチマークではない。ユーザーRyzen/WSL測定ではない。
- Wolframで16parameter、1,600更新/seed・合計4,800、train3744行/prefix384episode/新初期12episodeを照合。
- 旧/予約test生成/点検/採点0。seedや派生episodeを独立task件数へ加算しない。

## 次の境界・公開範囲

次は文字列→事実/Taskのreader。[後続設計](../docs/LanguageToActionRoadmap-v0.md)へ
構造化Tool履歴→C/M/F、短い依頼文→Task、次token事前学習と課題用学習、
ランダム/事前学習初期化の同条件比較、モデル/Tokenizer互換性と素材準備を記録。
今回言語モデルの追加事前学習・reader実装・自由会話・コード生成評価は未実施。
既存pilotの私的原文/training viewは本checkoutにないため、採用/分割/入力長を確認してから開始する。

PR #40に依存するstacked Draft。main/Service/API/aster-web/公開モデル/promotionは変更しない。
16parameterモデルは診断Artifactとして保存し、生履歴DecisionModelの置換/自動採用をしない。
全モデル/正式/debug証拠は別bundleに保持。今回ユーザー側追加作業不要。

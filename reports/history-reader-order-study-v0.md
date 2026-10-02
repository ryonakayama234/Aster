# 履歴readerの順序学習介入 v0: 正式結果

2026-10-02 JST。正本[protocol](../docs/HistoryReaderOrderIntervention-v0.md)、
[登録データ](history-reader-order-preflight-v0.json)、[全条件JSON](history-reader-order-study-v0.json)。

## 結論・得られたこと

順序対照の追加はseed42では元80例保持と追加32例fitを両立した。
seed43は追加31/32、seed44は16/32で、同じ予算でも安定成立しない。
追加した対照両側は16/15/0（各16組）。同更新量/同ラベル頻度の元例反復は全seed0/16。
この予算内で、提示内容の変更が既知順序への適合を一部改善した証拠はある。
ただしtrain fit、未学習shape、未知task、自力完了を同一の改善としない。

既知taskの未学習shape両側は順序追加5/5/1（各24組）、
既存dev taskでは0/1/0（各12組）。後者はseed43のtail_getで1/4のみ。
全て完全入力非重複だが、観察済みdevと同じ12task由来であり、独立testではない。
元devの自力完了は順序追加5/17/4（各40件）。seed43の改善とseed42の悪化を併記する。
反復条件13/10/12に対し安定優位はない。今回のモデルを公開採用しない。

## 全9条件

|seed|条件|元80事実|追加32事実|追加対照 /16|既知未学習shape /24|dev未学習shape /12|元dev事実 /40|元dev自力完了 /40|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|42|frozen|80|16|0|0|0|21|6|
|42|repeat|80|16|0|0|0|19|13|
|42|order|80|32|16|5|0|21|5|
|43|frozen|80|16|0|0|0|24|12|
|43|repeat|80|16|0|0|0|24|10|
|43|order|80|31|15|5|1|18|17|
|44|frozen|80|16|0|0|0|20|3|
|44|repeat|80|16|0|0|0|16|12|
|44|order|80|16|0|1|0|16|4|

全条件の元80例事実正答80/80、元正常初期8/8完了を保持。
ただし元80prefixからの継続完了は全件とは限らない（全値はJSON）。
学習後epoch40だけを正式採点。seed42のorderはepoch30=110/112→40=112/112、
seed43は96/112→111/112、seed44は10/20/30/40全て96/112。
したがって全3seed安定fitや容量不足の確定を主張しない。

## 未学習shape: 両側事実正答

|seed|条件|既知middle_put /8|既知tail_get /8|既知repeat_get /8|dev middle_put /4|dev tail_get /4|dev repeat_get /4|
|---|---|---:|---:|---:|---:|---:|---:|
|42|frozen|0|0|0|0|0|0|
|42|repeat|0|0|0|0|0|0|
|42|order|1|2|2|0|0|0|
|43|frozen|0|0|0|0|0|0|
|43|repeat|0|0|0|0|0|0|
|43|order|0|1|4|0|0|1|
|44|frozen|0|0|0|0|0|0|
|44|repeat|0|0|0|0|0|0|
|44|order|0|0|1|0|0|0|

## データ・外部実装との境界

元80train/40dev、追加32、未学習shape72、前回固定対照40を全条件で同じ評価。
264評価slot/236unique入力/12task、76対照。重複を独立課題へ数えない。
追加trainにはplain/tail_putのみ。未学習shapeはmiddle_put/tail_get/repeat_get。
両側は同長/同memory/同event multiset、tail_put/tail_getは共通末尾も同一。
未学習72入力は比較全armのtrain入力unionと重複0。旧/予約test生成/採点0。
最大1169Byte tokens、context2048以内。自力継続でのcontext超過はinvalidとして残す。
原正常例、旧train重複や前回対照の保持はmanifestと全pairのboth_nontrainへ保存。
特にcontrast_train/orderは追加trainのplainを含み、独立汎化とは呼ばない。
前回値対照両側は全条件/全seed0/4のままで、値一致能力の改善根拠はない。

実Toolの観測から入力を作り、JSON oracle/既存relations/教師を照合。
oracle264/264完了は生成・分類器・候補結合の対照で、readerの能力ではない。
予測を真値へ差し替えず、3事実class/Action/8step自力継続を別記録。
failed Tool/矛盾abstain/候補結合失敗/誤stopを全開始分母へ残す。
学習器はまだTask生成/算術/引数生成/自由会話を担当していない。

## 比較と費用

各seedは保存親の全tensorが同じ。frozen更新0、repeat/orderは各4480更新。
40epoch×112slot、元80例は各epoch1回。追加32slotは同task/同C,M,Fをslot単位照合。
B/Cのslot順・optimizer再初期化・AdamW lr0.001/default・clip1.0を一致。
元例個別の総exposureは追加枠で異なる。介入内容と原例反復を比較した。
入力はByte258/幅16/2head/1layer、末尾読出しの同architecture。
系列長とCPU費用は一致しないので純粋な計算量一定効果とは呼ばない。
repeat各2,756,840tokens、order各2,865,320tokens（約3.9%多い）。
optimizer stateを証拠として保存するが、runnerの学習再開APIは未実装。

正式Run `41954b36546245649d40e810683261a9`、clean local source `2fa38409468dacbacb8ed36f029f9b89121f6391`、
GitHub同一source tree `56b26d7c85f3e2c2e41dd84b51127362b1cd70b4` / commit `07f08ad09e06395cb7304cf69896422983b82b0d`。
正式240.536秒、Linux/Python3.12.14/torch2.14.1+cpu/CPU2threads。
ユーザーRyzen/WSLの測定ではない。

## 保存・監査・検証

親weight files/固定分類器は不変。親旧120入力×3seedのcached scoreへ絶対差1e-5以内/
class変更0を要求。全9条件の264予測は同process checkpoint再読込で全score完全一致。
評価weights/RNG不変、測定後Python hash変更0。元データ/初期親/optimizer/slot順/
curve/全logits/訪問履歴/Tooltrajectoryを保存。
193pytest passed、Pyright0 errors/0 warnings、compile/diff check成功。
保存監査は全2376予測を再読込/別算式NLL/margin再採点、訓練3360slot予測と提示順/
更新量/ラベル頻度を照合。2376自力episodeの20,652transition/11,957visitと
264oracle episodeを実Toolで再生し、停止時の保存値と最後の対象put後getを独立確認。
監査自体は学習更新せず、集計と能力測定を区別する。

## 次に問うこと

順序例追加は一部の既知履歴への適合を改善したが、3seed安定fitも構造転用も未成立。
単に予算延長・複数問題追加・一般Corpusへ移行する根拠にはしない。
次の候補は「確認Fだけの最小読解課題で、順序の安定fitと未学習shape転用が成立するか」。
現在はraw JSON構文、対象対応、順序、C/M/F同時学習を一緒に扱っている。
Fだけでもfitできないなら学習/位置表現/読出し構造を一軸比較し、fitするが転用なしなら
見た目の手掛かりが通用しない分割と表現を調べる。Fだけ成功しても完全Agent能力と呼ばない。
現時点で特定の内部shortcut、architecture不可能性、事前学習の必要性を証明していない。
自力訪問への訂正や値一致の学習を同時導入せず、次protocolは別に登録する。

PR #44依存のstacked Draft。main/Service/API/aster-web/公開モデル/promotion変更なし。
ユーザー追加作業不要。将来のWeb表示は学習例への適合/未知shape/未知task/自力完了を分ける。

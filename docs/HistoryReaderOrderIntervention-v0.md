# 履歴readerの順序学習介入 v0

2026-10-02 JST。PR #44依存。考え方: 固定3readerは既知taskの値/順序対照を
両側正答0/4。元80例fitを順序規則の転用と呼べない。今回は値不一致を追加せず、
同じmemory/event multisetで確認Fだけが変わる順序を課題用学習する。

## 仮説・反証・達成目標

仮説: 関係を要求する順序対照を提示すると、更新量/ラベル頻度を揃えた元例反復より
順序両側正答が改善し、元80例を保持する。未学習shapeにも転用できるかは別問い。
train対照にfitしないなら学習/表現/容量、fitするがshape転用なしなら関係汎化へ戻す。
自力完了だけ崩れるなら自力訪問履歴訂正を次候補にする。成功保証/モデル採用なし。

## 固定条件・変更点・対照

- 親Run1cc57fcd7b6e41819bdd95d35c272003のseed42/43/44を同一親として使う。
  architecture/Byte/入力schema/教師/候補/goal-v1/固定分類器を変更しない。
- A frozen: 更新0。B repeat: 元80例＋元例から選ぶ32反復slot。
  C order: 元80例＋train8task×2shape×F両側=32slot。
- B/Cは40epoch×112=4480更新/seed、AdamW lr0.001/default、clip1.0、CPU2threads。
  保存optimizerがないため両方とも新規optimizer。学習再開ではなくcheckpoint起点介入。
  同seedの初期全tensor/epochのslot順/元80例の各epoch提示1回が一致。
  B追加slotはC追加slotと同task/同C,M,Fの元例を循環選択する。
  元例個別の総提示回数はB/Cで異なる。この違いは介入内容そのもの。
  更新/ラベル頻度は一致、系列長/総token量/CPU時間は一致せず全て記録する。
- train shape: plain(計算→put→get→put 対 計算→put→put→get)、
  tail_put(両側へ同じ無関係keyへのputを追加)。同じ末尾道具でもFが違う後者を含む。
- shape評価: middle_put(最初の対象put後に無関係put)、tail_get(無関係keyを先に
  作り、両側末尾でそのkeyをget)、repeat_get(対象getを2回にする)。
  これらのshapeは追加trainへ含めない。元80例unionとの完全入力重複を検査する。
  train8task/既存dev4taskを分けて報告。既存devは研究者が観察済みで独立holdoutではない。
- 元80train/40dev、今回追加train32/未学習shape72、前回固定対照40も全条件同じ評価。
  比較armのtrain入力union重複を各入力/対照両側へ注記する。評価slot数とunique数を区別。
  旧/予約testは生成/読込/採点しない。未学習shapeの診断を封印testと呼ばない。

## preflight・停止条件・測定

実Toolで生成/再生。独立JSON oracle/既存relations/教師を照合し、各対照同長/
同memory/event multiset/期待C=true,M=true,F両側を検証。切詰め/正解差替えなし。
全登録input context2048以内、oracle完了必須。suite/protocol/source/親weights hash固定。
親120cached logitsは絶対差1e-5/class変更0で確認。同process reloadは完全一致。
登録後testと型検査を通しclean source commitから正式1回。debugは正式へ加算しない。
各学習arm900秒上限、非有限loss/hash/正解/配線不一致なら停止し失敗を記録する。
trainはepoch0/10/20/30/40の全112slotを保存。選択/早期停止なし、最終epoch採点。
三事実完全一致/事実別混同行列/対照両側/Action一致/自力8step継続を別保存。
正常80例保持と正常初期8例、追加train両側16組、未学習shapeのtrain24/dev12組を別集計。
全開始分母へinvalid/abstain/誤stopを残す。初手正答と完了を同一視しない。
全重み・曲線・提示slot・logits・訪問入力・実Tooltrajectoryを保存、別監査で再採点/再生。
測定後に予算・条件を調整しない。悪化でも報告と次の判断を残す。

## 結果別分岐・全体への接続

追加train fit不成立: 容量不足と断定せず、学習/位置/読出しを一軸比較する。
fit成立かつ元例忘却: 保持と追加fitを両立する学習配分を別実験にする。
fit/保持成立かつshape失敗: shapeへの依存を探る。大量Corpusへの移行根拠にはしない。
shape転用成立かつ自力失敗: train taskのみの自力訪問訂正を別登録する。
正常保持/転用/完了改善: 順序を足場候補にし、値一致を次の独立介入にする。
本実験はCLI研究。Service/API/aster-web/公開モデル/promotionは変更しない。
観測UIへ将来渡す基準は元例保持/学習shape/未学習shape/既知と既存dev/更新量を分ける。
ユーザー追加作業不要。文献背景: CheckList (ACL2020)、shortcut learning (Geirhos2020)、
SCAN (Lake/Baroni2018)。これらをAsterでの原因証明と扱わない。

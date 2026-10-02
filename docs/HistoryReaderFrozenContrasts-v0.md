# 固定履歴reader：名前・一致・順序の対照診断 v0

2026-10-02 JST。基準PR #43 head e30a7c1cb22eca4bb86c4eb2877d051f4e09599a。
モデルは正式Run1cc57fcd7b6e41819bdd95d35c272003のseed42/43/44を復元し、更新0。

## 考え方・仮説・反証条件

前回はtrain80/80をfitしたが、dev40事実同時21/24/20、自力完了6/12/3。
未知taskで数値・キーの文字列・入力長が同時に変わるので原因を断定できない。
今回は同じ長さ・到達可能性を維持する対照で、名前への依存、値の一致、最終書込みと取得順を分ける。
成功を保証せず、固定モデルの関係読解に反例があるかを確かめる。

- 名前：対象キーの全出現を同じUTF-8 byte長の一意な別名へ変えても、正しい事実/行動を保持するか。
- 一致：最後の書込みだけを同じ文字数の誤値へ変え、Mだけが変わる組を選び分けるか。
- 順序：同じAction/Observation multiset、同じmemory、同じbyte長でも、最後のput前後のgetでFだけを変えられるか。
- 正しい側も誤った側も同じ予測である場合、単なる予測不変を名前対照の成功と呼ばない。
- 診断は必要条件の反例探索であり、内部アルゴリズムや失敗の唯一原因を証明しない。

## 測定前に固定する生成条件

元reader train taskのindex0/1/4/5と元dev全4taskを採用。既知/既存devの各4task。
名前対照はnormal3(計算→保存→取得)と全対象キーのASCII文字を1文字ずつ循環改名した同じ経路。
一致対照は計算→保存→取得→同値再保存と、最後だけ同じ符号・十進表記長の違う値を再保存した経路。
誤値は観測値+1、長さ/符号が変われば観測値-1。対象taskについてどちらも不可能なら生成を停止する。
順序対照は計算→保存→取得→同値再保存と、計算→保存→同値再保存→取得。
順序ではM=trueを保ち、F=false/trueに変える。v1の教師はそれぞれget/stop。

8task×3軸=24組、48参照slot。一致の正側と順序の未確認側は同一入力を共有するため40unique入力。
全入力を実ToolExecutorで生成し再生する。Task/current memory/全Action・Observationの旧schemaを維持。
step/candidate/teacher/evaluation/正解事実をreader入力へ追加しない。切り詰め/正解差替え/fallbackなし。
旧trainとの入力完全重複、前回devとの重複を明示し、完全非重複の対照両側件数も別集計。
これは研究者が観察済みの既存devを使う追加診断。独立test/構造holdout/新しい成功率の推定ではない。
旧/予約testを生成・読込み・採点しない。3seed/派生/重複slotを独立taskとして加算しない。

## preflight・固定モデル・停止条件

各組のbyte長、イベント数、期待C/M/F、教師種類、入力衝突0、40uniqueを確認。
名前はtaskのキー/全events/current memoryの一貫改名と旧キー消失を確認。
一致はtaskと前三eventsが同一、現在memoryと最後put値だけが変わり、C/Fは不変と確認。
順序はtask/memory/イベントmultisetが同一、イベント順だけが違うことを確認。
独立JSON oracle/既存observable_relations/v1教師と手続き上の期待ラベルを照合する。
全oracle継続は停止時memoryと最後対象put後getを独立チェック。失敗なら測定前に停止/改版。
suite/pair/protocol/Python/source/元manifestとweight hashを保存し、生成registration一致を要求。
readerは元3model digest、分類器は元de12ec2…を要求。元120cached logitsとreload scoreも照合。
同一processの再読込は全score完全一致を要求する。
元環境とのscore比較は下記の測定前改版に従い、絶対差1e-5以下/相対許容0かつ全事実class一致を要求。
範囲超過やclass変更なら正式測定を停止する。完全一致したと報告しない。

### 正式測定前の数値再現確認・改版（v0 registration revision 1）

新対照はまだモデル採点せず、元120入力だけで復元検証を行った。
weight file/model digestは一致、PyTorch2.14.1+cpu/2threads。
seed42/43/44の元logit最大絶対差は1.9073486328125e-6 / 1.430511474609375e-6 / 1.9073486328125e-6、
事実class変更は全て0/120。CPU実行環境間の浮動小数点差に整合するが、特定kernelを原因確定しない。
旧入力での数値確認だけを根拠に、上記許容を正式対照測定前に固定。モデル/対照/正解/予算は変更しない。
元score/復元score/最大差/許容/予測一致件数を正式Runに残し、同一process reloadは厳密一致のまま。

## 測定・達成目標

CPU2threads、各seed300秒上限、正式1回。全40uniqueの事実/事実別混同行列/NLL/margin、
教師Action一致/結合失敗/abstain、24組の両方事実正答と両方Action正答を記録。
予測不変と正しく不変、正しく変化を分ける。各組の元Task group/旧train重複を保持。
全40prefixのoracle継続と3readerの自力継続を記録し、invalidも全開始分母に残す。
weights/RNG不変、別reload score一致、入力への教師混入なしを検証。学習/温度校正0。
保存logitsを別算式で再採点し、選択・結合・実Tool再生・停止時成功・集計を独立監査する。
達成は正本cases/pairs/全予測/trajectory/reportから三つの反例と範囲を追えること。数値改善は条件にしない。

## 結果別の次の分岐と全体への接続

改名だけで崩れる→同長の名前/役割対応をtrain内で変える学習介入を別登録。
一致だけで崩れる→値比較を分離した課題用学習/入力表現の対照。
順序で崩れる→同値再保存と取得順の関係学習を別登録。
prefix初手が正しく後で崩れる→train task内の自力訪問履歴に訂正を付け、更新量対照と比べる候補(#20)。
複数軸で崩れる→複数原因候補を保持し、一軸の診断を原因確定としない。
事前学習比較(#21)はこの診断を物差しにして素材/architecture整合後。Task readerは後続。
Service/API/aster-web/公開モデル/promotionを変更しない。観測#29→aster-web#2へ将来渡す保存証拠を用意する。
今回ユーザーの追加作業不要。

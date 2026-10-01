# Decision：観測関係を明示した入力の比較 v0

2026-10-01。基準PR #39、commit 66de75cd67b55e473928321e76b51435a82dca0b。

## 考え方・仮説・反証

正常例を保持した混合trainでは全seed fitが成立したが、同候補・異targetの
未知状態対照は安定改善しなかった。次は「事実の関係抽出」と「Action選択」を分ける。
観測関係を外部抽出した入力で改善するなら、現在の文字列読解/表現が障害という
説明を支持する。両入力でfitしても未知状態が同様に崩れれば、読解だけでは説明できない。
train未fitなら、そのarmの汎化失敗を読解だけへ帰属しない。

## 変更するもの・固定するもの

- 2arm: compact-v1 / relations-v1。両方PR #39のnormal-plus-recovery-v1の48unique/64slot。
- 同じByte、width16/heads2/layers1/context2048、全初期weight、slot permutation。
- seed42/43/44、LR0.001、AdamW既存default、25epoch=1,600更新/arm、CPU2threads。
- 教師/候補/実行/評価はcalculate-and-store-current-v1、fallbackなし。
- relationは成功した現task calculator観測、現在対象memory、成功put/getの順序、
  最新getの出力、最後Toolの失敗からのみ導出。未観測はnull/存在fieldで区別。
- 正解Action、Evaluator goal、独立計算した期待解、phase/stepを入力しない。
- 実task/候補/観測値は残すが、生event列と回数は取り除く。
  関係明示と履歴正規化・系列長変更を同時に行う**診断介入**であり、
  一つの特徴の純粋な因果効果とは呼ばない。外部化した能力をモデルへ帰属しない。
- relation extractorはteacherのcurrent_calculation/has_current_confirmationを呼ばず、
  raw Action/Observationを独立scanする。calculationの算術自体は実行しない。

## 学習前検証と正式登録

既存preflight正本を再生成しdigest一致を要求する。その後48train/128devを実Toolで
再生・独立終状態確認。各serializerでinput/target衝突0、Byte可逆、全固定入力2048内、
教師/evaluation metadata非依存を検査する。新入力のtrain重複は各armで再計算する。
主対照は両serializerのtrainとの重複を除いた共通caseに限定し、全32組も別記録する。
関係が同じ誤put反復で入力が同一、get後の同値putでfreshnessが失効、他キーputで
維持することを実Toolで確認する。生成preflight成功をモデル能力と扱わない。
登録JSONのdigestを正式runner開始前に要求し、変更があれば測定を拒否する。

## 指標・保存・停止

全epoch train予測/NLL/margin、正常保持、serializer別重複と共通非重複主対照、
128teacher-prefix判断、同prefixから最大8行動の自力継続、初期各variant2episode、
invalid、最初の誤り、入力/token/候補/実trajectory、CPU/token量を保存。
両arm共通有効prefix数を示し、context超過を通常失敗へ合算しない。
reload/全初期weight/slot順/候補逆順Action/評価中weightsとRNG不変を検査。
各arm30分deadline。失敗を保存し、条件を変えて成功armだけ報告しない。
独立cached auditでtrain指標、dev選択、全trajectoryの実Tool再生と停止時状態を照合。

達成目標は6armと監査を完了し、原因候補を狭めること。改善は完了条件ではない。
主指標は共通非重複対照組の両方正答。全seedで改善し既知正常保持が成立したら、
表現介入の限定的な支持とする。seed混在なら安定改善未成立とする。
初期自力完了は別の能力指標で、prefix成功で代用しない。
事前定義した3seedのみで有意差や広い成功率を主張しない。

## 結果別の次の分岐・公開範囲

relationだけ改善: 外部事実表現を足場にし、後続で正規化/関係追加を別比較する。
両方fitして崩れる: 学習内の関係組合せ支持・名前依存を監査し、それから選択モデルを比較。
一判断は改善し自力だけ失敗: 訪問stateの訂正比較へ進む。どの分岐も別protocol。
devは既存2親group由来で独立holdoutではない。旧/予約testは生成/点検/採点しない。
main/Service/aster-web/公開モデル/自動promotionは変更しない。PR #39へstackする。
ユーザー側追加作業は不要。観測#29→aster-web#2、言語#21、育成#20は別経路。

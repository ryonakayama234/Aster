# Decision状態対照train比較 v0

2026-10-01 JST。親 #24、基準 PR #32 head ee61711483dd876d35728a1999f6def6966b328f。

## 考えたこと・反証可能な仮説

compactは学習fitを安定させたが数値変更の自力完了は0/2。固定4段階から作るtrainが実行経路に偏る仮説を検証する。無関係なdelayはcompactから消え、新しいモデル入力を作らない。再計算prefix系列は既存sealed testへ予約済みなのでtrain/devへ移さない。

実Toolで対象キーへ誤値を保存したprefixを作る。誤値はtrain=-999、dev=-777とし、実計算出力と異なることをpreflightで確認する。値だけ変えたdevは同じ生成family由来の診断であり独立holdoutではない。

仮説: 入力方式・モデル・更新予算を固定して実状態の対照を学習させると、学習と入力が重複しない診断caseの対照組正答が複数seedで改善し、元課題を保持する。fitが成立しても改善しない場合、この介入による改善仮説は支持されない。fit不成立の場合は汎化の結論を保留する。

## 固定条件と変更点

- baseline: PR #32と同じnumbers-and-keys 32判断。state: 同じ8 task/数値/キーと4行動の教師件数8ずつを保つ32判断。
- stateのcalculator教師: 計算前に対象キーへ誤値を保存。put教師: 計算後に対象キーへ誤値を保存。get/stop教師は元の正常prefixを維持。異なるtargetを要求するstep2のput/getは同じ候補集合。
- compact serializerのコードとID、Byte、width16/heads2/layers1/context2048、LR0.001、50epoch、単例shuffle、2threads、seed42/43/44を固定。各arm1600更新。seed内の全初期weightとslot permutationを共有する。同じslotの実内容は介入で変わる。
- baselineも新環境で再測定。PR #32との環境差を記録し、以前の数値との純粋な比較をしない。fresh initializationで、旧モデルからの継続学習ではない。
- 各arm学習/reload/devに30分deadline。例外・timeout・context超過を保存。片側のみ延長しない。最大3seed/6armで終了し、途中の成績を理由に条件やseedを変更しない。

## 作業方法・学習前検査

1. 実Tool replay、rule継続、教師coverage、同step/同候補対照、short inputの差・衝突・長さを検査する。
2. trainとdevの全候補compact token列の一致をunionで監査。重複caseはretentionとして別集計し、新しい対応力へ加算しない。候補順を正規化した署名も使う。
3. 同じ手順番号のcalculator/put/get対照を新devに作る。calculator教師には誤保存を二回実行し、put/getとstep2を揃える。candidate集合が一致するput/getを主対照、calculatorを含む組は補助対照とする。
4. 原48state dev/56old dev/8初期episodeを保持して再評価。原dev内の入力重複も明示する。sealed old/reserved testは構築・長さ点検・採点すべて0。
5. protocol・suite・候補/入力・split・初期weight・source hashを学習前に保存。debugと本測定を分離。

## 評価・達成目標

- train安定fit: 最後3epochの全32正答・全margin正。accuracy/NLL/min margin/全予測を保存。
- 主指標: 新devの両caseともunion-train入力非重複で、同step・同候補集合・異targetのput/get対照組を両方正答した件数。全件と重複を除く件数を保存。複数seedで差の方向を示し、独立反復数へ加算しない。
- 副指標: calculatorを含む状態判断、rule/step-only/first-candidate/train-only候補lookup、prefixからのmodel-only継続、元既知2episode保持、数値/キー/joint初期episode、最初の誤り、coverage、invalid。
- teacher-prefix判断とmodel-only継続、最初からの自力完了を区別。teacher不一致でもgoalへ到達できる場合は成功。fallbackなし。引数はcandidate builderが作り、モデルによる生成能力へ算入しない。
- 開始状態のprefixを与えた継続は自力初期episodeと分ける。既存devと同じparent2group/同じ計算保存taskで、独立family汎化とは主張しない。

## 結果別の次の分岐・役割

判断だけ改善: 失敗が伝播するmodel-visited状態を調べる。判断と完了が改善: 別の生成経路を新protocolで設計。fitするが改善なし: 介入範囲の限界を報告し表現/最適化を別実験で検討。fit不成立: 同データで学習成立を診断。数値変更失敗は状態対照と別の表現仮説として残す。

こちらが実装・学習・監査・docs/reports・Draft PRを担当。ユーザーWSL再現は環境再現が必要になった時だけ、目的と手順/必要な出力を提示する。今回私的素材や本人の選定は不要。

AsterのAgent Loop→Trace→Evaluatorの段階に位置する。#20の訂正親子比較、#21の言語pilot、#29→aster-web#2の保存再生は別経路。Service/Web/モデルpromotionは変更しない。改善なしでも比較可能な証拠と次の判断が残れば実験を完了できる。

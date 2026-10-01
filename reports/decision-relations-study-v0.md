# 観測関係入力の比較：全6条件実測 v0

2026-10-01。[事前条件](../docs/DecisionRelationsComparison-v0.md)、[正本JSON](decision-relations-study-v0.json)。

## 結論

compact/relationsとも全3seedで48trainのstable fitと正常32判断/既知初期2episodeを保持。
関係入力の共通非重複110判断の正答は増えたが、主対照24組の両方正答は4/2/2→0/0/0。
共通有効prefix完了も全seedで低下し、数値変更の初期完了はrelations全0/2。
**この関係明示＋履歴正規化介入の改善仮説は支持されなかった。**
一判断の精度だけでは、状態に応じた選び分けや自力完了を代用できない。
train未fitだけ/生履歴の長さだけで失敗を説明する根拠は弱まるが、容量不足は断定しない。

## 条件・全結果

同じnormal-plus-recovery-v1の48raw判断/64slot、Byte、width16/heads2/layers1/context2048、
LR0.001、AdamW、25epoch=1,600単例更新/arm、CPU2threads。seed内の全初期weightとslot index順一致。
calculator/put/get/stopは各400提示。教師/候補/実行/評価はcurrent-v1、fallbackなし。
relationsは観測された計算値/現在memory/成功した対象putとgetの前後/最新get一致/最後失敗を外部scan。
教師Action、Evaluator goal、算術で求めた期待解、段階/stepは入力しない。
実task/候補/観測値は残すが生event列と回数を除くので、関係追加だけの純粋な効果ではない。

|seed|入力|stable fit|正常train /32|共通非重複判断 /110|主対照両方 /24|既知初期 /2|数値初期 /2|
|---|---|---|---:|---:|---:|---:|---:|
|42|compact|成立|32|50|4|2|0|
|42|relations|成立|32|57|0|2|0|
|43|compact|成立|32|51|2|2|0|
|43|relations|成立|32|58|0|2|0|
|44|compact|成立|32|50|2|2|1|
|44|relations|成立|32|54|0|2|0|

キー変更・同時変更初期は全6arm0/2。正常train保持は学習内で未知正常devとは別。
主対照は同じ候補集合で異targetを要求する。24組はcaseを共有し独立24taskではない。

## 入力・重複・自力継続

生成は48train＋128devを実Tool再生/独立終状態で検査。両serializer衝突0、48unique train入力。
固定最大系列長1190→519（56.4%減）、固定候補token総量445428→327506（26.5%減）。
学習token量3789600→3092100（18.4%減）、padding込み3888300→3190800。parameterは両方44353。
同じ更新回数でもtoken量/系列長/CPU費用は一致しない。

新入力のdev train重複はcompact14/128、relations18/128。両表現のいずれかに重複する18を
保持確認へ分離し、共通非重複110判断と24対照組を主集計とした。旧114分母を流用しない。
128prefixから最大8行動の自力継続を保存。以下は両arm共通有効開始に限定。

|seed|共通有効 /128|compact完了|relations完了|差|
|---|---:|---:|---:|---:|
|42|121|51|37|-14|
|43|120|58|39|-19|
|44|123|69|39|-30|

context超過invalidはcompact7/8/5、relations全0。invalidを通常失敗に加算しない。
prefix完了は初期状態からの完了と別。教師Action一致と終状態成功も区別する。

## 測定後の探索的支持監査

測定後に追加したscripts/analyze_decision_relation_support.pyは事前登録した主指標ではない。
観測関係から具体的な計算値/保存値を除いたboolean/nullパターンとAction種類の対応を、
**train48だけ**から表にした。6パターン、ラベル衝突0。devに同じパターンが104/128存在し、
その表からAction種類を選び既存candidateへ結び付けると104/104正答、未知24はabstain。
正常32/32、factorial64/64、completion8/32をカバーする。dev正解で表を作っていない。
この表は外部抽出済みの事実と候補引数を使う非ニューラル対照であり、学習済みモデルの成績ではない。

同じ104件のモデル正答はcompact55/56/57、relations64/65/61。
「少なくともこれらの状態はtrainで見た抽象的関係の組合せだけで選べる」ことが確認できた。
関係入力でも数値/キー/候補引数の文字列が残り、モデルがその不変な対応を獲得したとは言えない。
この探索結果から、次は**行動種類の選択と、候補引数への結び付けを分けた小さな学習器**を
診断対照として事前登録する案が妥当。未実装・未測定。元の生履歴読解を学んだとは扱わない。

## 検証・source・保存・限界

- 全pytest170 passed（75.21秒）、Pyright --pythonpath .venv/bin/pythonで0 errors/0 warnings、diff check成功。
  最初のPyrightはvenv未指定で既存torch/docutils importを解決できず、環境指定を修正して確認。
- 1epoch debug Run bdf6fd0c82d84e1482967df851586564は正式結果へ算入しない。
  debug独立監査: train192/dev256/episode256/transition1715/訪問1091行を照合。
- 正式Run 7d4660729cf74cf3a4aab123ff23d05c。clean local source3538a25、dirty=false。
  API公開source18c4e2df38d54b52bf81eeb84b9315fb5ec0ae1cはcommit identityが異なるがtree
  696467ba5ed9b527b6628b0bec182594cfa1d53aがlocalと一致。Python/protocol digestはRunに保存。
- checkpoint serializer不一致拒否、train/dev全score reload一致、候補逆順のAction一致/score許容差1e-5、
  評価中weights/RNG不変、初期weight/提示順一致を全arm確認。
- 独立cached auditでtrain7488/dev768/episode768/transition6071/訪問4199行の指標/入力/選択を照合し、
  全Actionを実Toolで再実行。停止時memoryと最終対象put後getを別計算した。
- 測定完了後に測定時の全Python file hashを再照合し変更0。追加は探索分析scriptとreport/docsのみ。
- ChatGPT Linux/Python3.12.14/torch2.14.1+cpu/2threads、正式wall259.895秒。
  ユーザーRyzen/WSL測定ではない。全arm30分deadline内。CPU費用と全arm結果はJSONへ保存。
- 旧/予約testは生成/点検/採点0。2親group由来devで独立holdoutではない。
  seedやepisode派生件数を独立task件数へ加算せず、広い成功率/有意差を主張しない。
- PR #39をbaseとするstacked Draft。main/Service/API/aster-web/公開モデル/promotionへ未導入。
  全checkpoint/入力/予測/順序/trajectory/debugは別の実測bundleに保持。

次の問いは、値や名前に依存しないAction種類の判断を小さい学習器が獲得できるか。
今回の関係入力モデルを採用せず、別protocolで反証条件を定める。ユーザー側追加作業不要。

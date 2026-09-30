# Decisionデータ4条件 初回実測（2026-09-30）

## 条件と由来

[測定前に定めた条件](../docs/DecisionDataIntervention-v0.md)でseed42/43/44を測定。ChatGPT Linux CPU、Python3.12.14、PyTorch2.14.0+cpu、2threads。ユーザーWSLでの新CLI実測ではない。
元8 train判断の専用BPEを固定。各seed内で同じ初期weight・shuffle index。LR0.001、50epoch、各arm32slot×50=1,600 updates/exposures。元50epoch/400 exposuresとの比較には新しいrepeat-controlを使う。
[機械可読報告](decision-data-intervention-v0.json)にRun/suite/Tokenizer/checkpoint/model/元predictionのdigestと全条件を記録。sourceはPR #26上のworking treeであり、base SHAだけのコードで実行したとは扱わない。
seed42の起動後、重複検査のsignatureからteacherラベルを除いてmodel-visible入力を比較する安全検査を強化した。学習入力・初期化・順序・損失・更新・採点は変更していない。unique判断数も全条件で一致することを検証。

## train fitと自力episode

|seed|arm|train /32|最後3epoch fit|seen /2|数値 /2|キー /2|同時変更 /2|
|---|---|---:|---|---:|---:|---:|---:|
|42|repeat-control|32|成立|2|0|0|0|
|42|numbers-only|32|成立|2|0|0|0|
|42|keys-only|24|未成立|0|0|0|0|
|42|numbers-and-keys|30|未成立|2|0|0|0|
|43|repeat-control|32|成立|2|0|0|0|
|43|numbers-only|28|未成立|0|0|0|1|
|43|keys-only|28|未成立|0|0|0|0|
|43|numbers-and-keys|32|成立|2|0|0|1|
|44|repeat-control|32|成立|2|0|0|1|
|44|numbers-only|32|成立|2|0|0|0|
|44|keys-only|32|成立|2|0|0|0|
|44|numbers-and-keys|29|未成立|1|0|0|1|

## Teacher-prefixの診断

|seed|arm|seen /8|数値 /8|キー /8|同時変更 /8|遅延 /16|
|---|---|---:|---:|---:|---:|---:|
|42|repeat-control|8|4|2|1|6|
|42|numbers-only|8|6|3|4|2|
|42|keys-only|6|4|2|4|9|
|42|numbers-and-keys|8|1|0|5|1|
|43|repeat-control|8|3|1|2|5|
|43|numbers-only|6|4|3|5|8|
|43|keys-only|6|3|4|5|6|
|43|numbers-and-keys|8|2|1|5|6|
|44|repeat-control|8|2|0|5|7|
|44|numbers-only|8|4|3|2|4|
|44|keys-only|8|3|3|3|4|
|44|numbers-and-keys|7|3|1|4|7|

## 解釈

- 元課題の反復だけの対照は全3seedでtrain fitと既知2episode完了を維持した。
- 数値/キー/同時の拡張には、train fit自体が未成立のseed・条件がある。dev失敗を汎化だけに帰属できない。
- 判断精度の改善が自力episode完了へそのまま移らない。途中で誤ると、自分の誤行動で次の入力も変わる。
- 同時変更episodeの成功はrepeat-controlもseed44で1/2、numbers-onlyはseed43で1/2、numbers-and-keysはseed43/44で各1/2だった。拡張だけが成功の必要条件とは言えない。
- 同時変更episodeに部分的成功があっても、3seedで元課題と数値/キーの保持を伴う一貫した改善があるとは扱わない。
- teacher候補coverageは全56判断・モデル訪問状態で100%、rule対照は全条件8/8成功。モデル評価前後のweight一致、checkpoint reload一致を確認。
- 候補逆順の選択は各条件内で一致。独立candidate scorerの構造から期待されることで、能力向上の根拠にはしない。
- 56判断はtrainと親groupを共有するdev probe。独立task-family holdoutではない。新test12判断は教師の実行検証とhash保存だけでscoreなし。旧testも未評価。

## 今回わかったこと・次

元の8判断はfitできるが、今回の小さなデータ拡張だけで変化に安定対応する段階には達していない。単純にtrainを増やし続ける根拠にはしない。
[Tokenizer監査](decision-failure-audit-v0.md)では、元task全体を含む長いBPE tokenがあり、変更後に分割とtrain未出token IDの利用が変化した。次はデータ・予算を固定し、Tokenizer/入力表現を別条件として比較するのが候補。train fitの不安定さも先に追う。
Tokenizerを変える場合、語彙数/系列長/計算量/初期parameter数も変わるため「Tokenizerだけの純粋な因果効果」と誇張せず各条件を保存する。3seedだけの小さな課題群から有意差・広い知能を主張しない。

## 検証・公開状態

pytest115 passed、Pyright0 errors/0 warnings、diff whitespace検査成功。全12条件×51評価epochの19,584 predictionからaccuracy/NLL/min marginを独立再計算し学習曲線との一致を確認。Tokenizer digestは元ユーザーfitと同じd68bd292…でもある。新規4testは実行再生・equal budget/shared initialization/order・test隔離・Tokenizer anchor・dev更新禁止・cached監査tamper検出を確認。
実測はLinux CPU側。各seed約4分の観測だが、検証/監査の併走があり精密な速度ベンチマークではない。ユーザーWSLの所要時間は未測定。
モデル採用/promotion、Service/Web recipe、公開bundle、UI比較は未実装。PR #26に依存するDraft PRとしてレビューする。既存GitHub CIはmain向けで、このstacked PRのCI実行済みとは扱わない。

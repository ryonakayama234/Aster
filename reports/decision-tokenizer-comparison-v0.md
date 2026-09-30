# Decision入力比較 v0 実測（2026-09-30）

[測定前の条件](../docs/DecisionTokenizerComparison-v0.md)。PR #30の意味対応byte/BOS/EOS行を共有する初期化で、32判断・50epoch・LR0.001・width16/heads2/layers1/context2048・CPU2threads・seed42/43/44を固定した。
ChatGPT Linux CPU / Python3.12 / PyTorch2.14.0+cpu。ユーザーWSL測定ではない。BPEは元8trainから決定的に復元したartifactの公表digest完全一致を確認し、比較harnessへ保存済みファイルとして入力した。
語彙embeddingを共有しない別条件の試運転・途中実行は、この事前条件と違うため結果へ含めない。その実行を見てPR条件を変更していない。

|seed|方式|train /32|最後3epoch fit|元 /2|数値 /2|キー /2|同時変更 /2|
|---|---|---:|---|---:|---:|---:|---:|
|42|bpe|30|未成立|2|0|0|0|
|42|byte|32|成立|2|0|1|0|
|43|bpe|32|成立|2|0|0|1|
|43|byte|32|成立|2|0|0|0|
|44|bpe|29|未成立|1|0|0|1|
|44|byte|32|成立|2|0|0|0|

## 解釈と次の分岐

Byteでtrain fitの安定性は改善したが、数値・キー変更の自力episodeが3seedで揃って改善したとは言えない。専用BPEの長い結合を、変更対応の主因として採用する根拠にはしない。Tokenizerの影響がないと証明した結果でもない。
単にデータやモデルを増やす段階へ進めず、次は短い構造化入力を、到達可能な状態に対する判断で比較する。未計算/計算済み/保存済みの状態と手順番号を分離し、同じstepで異なる正解・異なるstepで同じ正解を含む事前protocolを設計する（Issue #24）。言語TinyLMの評価（#21）は独立した実験とする。

## 計算費用・検証・限界

全6armを完了。各arm1,600更新、全9,600更新。保存された9,792 train予測からaccuracy/NLL/min marginを独立再計算し全epochの学習曲線と一致。336 dev行で有限logits・候補順・argmaxとAction・NLL/確率/marginを照合した。候補coverage、rule対照各8/8、reload、評価前後weights不変、同一提示順を確認。旧test/予約testは未評価。
関連回帰64testと、新しいtimeout testを含む比較5testを実行。テスト成功はモデル能力の証拠とは別。過去の元cached auditログの再照合は今回未実施であり、今回照合したのは新規6armのdev predictionである。
系列長/語彙数/parameter数も変わるため純粋な一要因の因果効果とは扱わない。devは親trainと2課題groupを共有する診断probeであり、独立holdoutではない。3seedを独立課題数に足さない。
CPU時間は検証が一部併走した環境の観測値。両armを順次実行するprocessの最大RSSはarm別のメモリ比較に使えない。30分deadline超過はなし。
[機械可読報告](decision-tokenizer-comparison-v0.json)に全条件・学習曲線・Run/checkpoint/Tokenizer/入力/初期化digest・資源値を保存。測定時sourceはPR #30 head 64b0870に監査/deadline/事前条件を加えたworking treeで、全source snapshotのhashを明記した。
PR #30はレビュー用Draft。モデル採用、Service recipe、公開bundle、aster-webへの比較画面は今回未実装。私的Corpusの提供やユーザー側の追加実験は今回の完了条件ではない。

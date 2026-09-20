# 開発の現在地

2026-09-20。WSL `/home/zhong/src/Aster` で確認。
Git HEADは `f953dd6`。ユーザーのrepo骨格・資料と今回までの追加実装には未追跡ファイルが多くある。
それらを保護して作業しており、整理のための削除・コミットは行っていない。

## 実装の地図

| 場所 | 現在の役割 |
| --- | --- |
| src/aster/corpus/pipeline.py | 原本→canonical候補 |
| scripts/inventory_corpus.py | 原本一覧と出典台帳から採用調査を再生成 |
| src/aster/training/extract.py | document・RST・conversationから本文を抽出 |
| src/aster/training/view.py | pilotの採用・分割・配合・対応表・観測bundle |
| src/aster/records/runlog.py | pipelineのrunと順序付きイベントを保存 |
| src/aster/training/tokenizer_run.py | 固定viewのtrainだけでBPE学習、全splitの指標と往復一致 |
| src/aster/tokenizer/ | 既存のbyte/BPEと保存・読込を再利用 |
| src/aster/model, agent, runtime, tools, coding, evaluator | 主にユーザーの設計骨格。TinyLM・Agent実行は未実装 |
| docs/UI/ | ユーザーの意図・参考画像・Sites実装計画 |
| ui/ | 独立Git管理のSites画面。保存結果の本文比較・BPE・考察保存を実装 |

空の既存ファイルを推測で埋めず、最初の実装を小さな追加モジュールとして接続した。
将来役割が固まったら整理する。既存の旧canonical CLIは互換のため保持。

## 次に進める順序

1. **完了**：canonical→pilot training view→BPEと実データの観測bundle。
2. **実装済み**：Sitesの素材・Tokenizer画面。保存bundleをブラウザ内で読み、比較・考察を残す。
3. **次**：ローカル実行の接続方式を検証。画面から許可した処理を起動・監視する。
4. TinyLMの最小学習ループと学習前後の観測。その後に小さなtool runtimeへ接続する。

正式Corpus-v0凍結は、自作tool枠・分割・配合を確定してから行う。pilotの実行成功と区別する。

## ユーザーとエージェントの分担

- エージェント：抽出・検査・記録・UI・実行接続・学習処理を構築する。技術的な選択は推奨案で進める。
- ユーザー：原稿や対話の好み、Asterに真似してほしい応答、結果を見て次に学ばせたいことを判断する。
- 今すぐ必須のユーザー作業はない。原典の追加情報は分かるときに台帳へ反映できる。
- 私的対話の口調をSFTのお手本にする判断は、実際の候補を画面で見てから行う。

実測の詳細は [最初のpilot結果](../reports/training-pilot-v0.md)。

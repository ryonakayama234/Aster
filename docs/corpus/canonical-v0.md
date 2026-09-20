# rawからcanonicalへの自動変換

## 考え方

rawは収集した原本のままでよい。原本の種類ごとに読み取り方（adapter）を定義し、共通形式へ変換する。
ファイルごとに手でJSONを書き直す必要はない。新しいサイトや形式が増えたときに読み取り方を追加する。
人が与えるのは出典、用途、曖昧な発言境界など、原本だけから決められない情報。

canonicalはモデルに使うデータの共通の保管形式。変換できたことと学習に採用したことは区別する。
候補も共通形式で保存できるため、原典や用途の確認が終わるまで変換の自動化を止める必要はない。

```text
raw/pool + 出典台帳 + 変換recipe
  ↓ 一覧化・原本hash確認・種類ごとの変換
canonical/builds/<内容で決まるID>/records.jsonl
  ↓ 採用・重複確認・元文書groupで分割・配合（今後）
training/<実験名>/：学習方式ごとの入力（今後）
  ↓ trainだけでBPEを学習、本文をtoken化（今後）
TinyLM
```

raw/currentは、必要な場合に今回の採用原本を集めて眺めるための場所として残す。
変換の必須入力にはせず、台帳からpoolを直接読む。currentへの手作業コピーが自動化の前提にならない構成。
今回の実装はcanonical候補の生成まで。current、train/dev/test、学習用入力はまだ作らない。

## 一つのコマンドで更新

WSLのリポジトリルートで実行する。

```sh
.venv/bin/python scripts/build_canonical.py
```

1. poolの一覧と採用調査を更新する。
2. `configs/canonical-v0.json` のrecipeを読む。
3. 対応する原本を変換し、処理しないファイルにも理由を記録する。
4. 出力先の絶対パスを表示する。

出典情報は `docs/corpus/source-notes-v0.json`、外部調達の固定版は `external-seed-manifest.json`。
自動生成の `pool-inventory-v0.json` と `selection-review-v0.md` を手編集しても再実行で置き換わるため、出典は出典台帳へ追記する。
現在の一覧化は全poolを走査する。大規模化時の差分処理・ストリーミングは未実装。

## 共通schemaの最初の二種類

`data/canonical/AGENTS.md` の案に沿い、今回はdocumentとconversationを実装した。
他のkindは将来の拡張先として残す。新しいschemaは `aster-canonical-0`、Corpus-v0の凍結を意味しない。
既存の `id/text/source` 変換器はそのまま残してあり、新しいpipelineとは別コマンド。

| 項目 | 意味 |
| --- | --- |
| schema_version | レコード構造の版 |
| id | 内容と出典・変換情報から計算した識別子 |
| kind | document / conversation |
| text, text_format | documentの本文と形式 |
| messages | conversationの発言列。role/content/source_linesを持つ |
| domain, language | 領域と主言語 |
| group_id | 元文書・問題・会話のグループ。派生を分割で離さないために使う |
| split | 未設定はnull。変換だけではtrain/testを決めない |
| provenance | 原本パス・hash・出典・固定版・生成元・確認状態 |
| transform | 読み取り方、版、変更内容 |
| review | 学習への採用状態、応答模倣対象か、内容確認状態、注意点 |

現在は全件 `training_status: not_selected`、`content_verified: false`。
ChatGPTの回答はassistantの発言として保管し、その中のコードや結果を実行済みtool観測へ変換しない。
`source_lines` はBOM除去・LF改行統一後の原本文の1始まり・両端を含む行番号。発言の見出し行は含めない。

## 対応する読み取り方

| adapter | 入力 | 出力と変更 |
| --- | --- | --- |
| text_document | 台帳に登録した外部記事 | document。BOM・改行以外は本文保持 |
| headed_conversation | `# 質問` / `# 解答` / `# 回答`で区切ったAI対話 | conversation。見出しからroleを決める。コードfence内の見出しは本文として保持 |
| gsm8k | 固定版GSM8Kのtrain JSONL | 問題ごとにdocument。計算注釈を除去しQuestion/Answer形式へ変換 |
| rst_document | 固定版Python教材 | RST本文をdocumentとして保持。教材の表示用記法の除去・節抽出は未実装 |

曖昧な見出し順、空の発言、閉じていないコードfence、不正UTF-8などは理由付きで保留する。
回答内に引用として見出しが単独で出ると、自動判定が誤る場合はある。書式規則の範囲でのみ解析し、意味の正しさは保証しない。
日本語・Pythonの空白やインデント、全角半角、Unicode文字を勝手に置き換えない。
教材を説明文とコードへ切り出す処理はこのadapterを育てる次の段階。

## 出力と再現性

各buildに次を保存する。

- `records.jsonl`：一行一レコードの共通データ。
- `report.json`：全原本の変換件数・保留理由。
- `manifest.json`：入力台帳・recipe・変換コード・出力のhashと件数。

全ファイルを一時ディレクトリへ書いてから完成版を公開する。同じ入力とコードでの再実行は同じ出力先になる。
既存の出力が変更されていたら上書きせず停止する。原本や台帳の変更を検知した場合、古い一覧からは変換しない。
一コマンド版は先に一覧を更新するので、新しい原本は新しいbuildに反映される。

JSONLはファイルを一行ずつ読んで `json.loads(line)` で解析する。
`read_text().splitlines()` は本文中のUnicode行区切り文字まで分割するため使わない。

## 検証範囲

テストでは改行・インデント保持、見出しとコードの区別、曖昧な対話の拒否、GSM8Kの変換、
Unicode行区切りの扱い、再実行の一致、原本・台帳・出力の変更検知を確認する。
実データの初回変換は、日本語記事5件・ChatGPT対話5件・Python教材4章・GSM8K 7,473問を対象にした。
数学的正しさ、利用条件、対話の好ましさ、学習効果を証明するものではない。

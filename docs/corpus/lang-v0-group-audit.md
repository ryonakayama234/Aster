# LANG v0 — Group audit and split freeze

## Decision

LANG v0のactive corpus 22件は、現時点で**10個の保守的なleakage group**として扱う。
未解決batchをファイル単位へ細分化する根拠はまだ無いため、同一batch内をtrain/dev/testへ跨がせない。

splitは `configs/lang-v0-split.json` をsource of truthとする。
provenance台帳 `source-notes-v0.json` 自体にはsplitを書き込まない。

## なぜhash splitを使わないか

10 groupしかなく、かつ最大groupが大きいため、単純hash splitではdev/testの領域構成が偶然に強く左右される。
一方、現在有効な6 groupについては、LANG v0以前の `configs/training-pilot-v0.json` に既にsplit overrideが存在する。

今回は**現在のbytesや将来の学習結果を見て再最適化せず、その既存overrideを継承**する。

## Frozen assignments

| group | files | bytes | disposition | rationale |
| --- | ---: | ---: | --- | --- |
| article-calculemus | 1 | 20,861 | train | 既存overrideを継承 |
| article-bsd | 1 | 16,063 | dev | 既存overrideを継承 |
| article-lean-trust | 1 | 25,072 | train | 既存overrideを継承 |
| article-tex-lean | 1 | 16,606 | train | 既存overrideを継承 |
| article-adverbs | 1 | 23,347 | test | 既存overrideを継承。sealed |
| unresolved-chatgpt-sessions | 5 | 95,203 | train | 既存overrideを継承。独立session未確認なので1 group維持 |
| unresolved-memes-batch | 5 | 20,864 | pending_excluded | 原典未特定 |
| unresolved-manga-batch | 3 | 69,594 | pending_excluded | 作品・版・話の特定待ち |
| marshmallow-qa-batch | 3 | 12,745 | pending_excluded | 個別ページ特定待ち |
| unresolved-x-dialogue | 1 | 876 | pending_excluded | 元投稿特定待ち |

## Byte accounting

active全体:

- 22 files
- 301,231 bytes

今回のsplitへ入るeligible subset:

- 10 files
- 197,152 bytes
- train: 157,742 bytes (**80.01%**)
- dev: 16,063 bytes (**8.15%**)
- test: 23,347 bytes (**11.84%**)

provenance未解決で保留:

- 12 files
- 104,079 bytes

この `pending_excluded` は削除ではない。
raw/canonical候補として保持するが、LANG v0のBPE fitting・training・dev・testへは入れない。

## Group audit

### 独立groupのまま扱えるもの

5件のexternal articleはそれぞれ異なるsource URLを持つため、現状の1記事=1 groupを維持する。

### 細分化しないもの

- `unresolved-memes-batch`: 原典が未特定。別出典と断定できない。
- `unresolved-manga-batch`: 作品・版・話のidentityが未確認。
- `marshmallow-qa-batch`: collectionは分かるが個別ページidentityが未確認。
- `unresolved-chatgpt-sessions`: 5ファイルが独立sessionである証拠をまだ固定していない。
- `unresolved-x-dialogue`: 1 recordなので現状のgroupで十分。

漏洩を減らす方向では「不明なら広くまとめる」を採用する。

## Limitation

LANG v0のdev/testは**proseのみ**になる。
ChatGPT dialogueを独立sessionだと仮定して分割することはしない。

したがってLANG v0で主張できるのは、group-separated held-out proseに対するnext-token signalまで。
**held-out dialogue generalizationはLANG v0のclaim対象外**とする。

Dialogue評価は、session identityを確認した後のLANG v1または別のcorpus revisionで扱う。

## Freeze rule

このsplitを学習結果を見て変更しない。

変更が必要になった場合はLANG v0の同一runを上書きせず、
理由を記録した別corpus/split revisionとして扱う。

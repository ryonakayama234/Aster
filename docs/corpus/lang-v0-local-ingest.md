# LANG v0 private corpus — local ingest

Google Drive上のprivate corpus本文はGitHubへcommitしない。
LANG v0では、Driveからローカルへ取得した22件を一度だけWSLのraw poolへ配置し、
以後のhash・canonical・split・Tokenizer処理はローカルの実bytesを基準にする。

## 使い方

Driveから対象テキストを**リポジトリの `data/raw/pool` と重ならない別フォルダ**へダウンロードする。
フォルダが複数階層でもよい。対象ファイル名は `proseNNN.txt` /
`dialogueNNN.txt` のままにする。`data/raw/pool` 自体、その配下、またはそれを含む親ディレクトリを
sourceとして指定すると、元ファイルと正規化コピーが同じpoolに混在するため拒否する。

最初に書き込みなしで確認する。

```bash
.venv/bin/python scripts/ingest_lang_v0.py /path/to/downloaded-folder --dry-run
```

問題がなければ同じフォルダを本投入する。

```bash
.venv/bin/python scripts/ingest_lang_v0.py /path/to/downloaded-folder
```

これだけで、

1. `configs/lang-v0-corpus.json` の22件がすべて存在するか確認する。
2. manifest外の `proseNNN.txt` / `dialogueNNN.txt` が混じっていないか確認する。
3. 各ファイルのbytesがDrive snapshotと一致するか確認する。
4. 全件が通るまでraw poolへ何も書かない。
5. `prose001.txt -> data/raw/pool/prose/ja/001.txt` の形式で配置する。
6. 既存rawと完全一致ならno-op、異なるbytesなら上書きせず停止する。
7. 配置後に `scripts/inventory_corpus.py` を実行し、raw SHA-256等を記録する。

スクリプト自体はGoogle Drive APIへ接続しないため、Drive ID・OAuth token・private URLを
GitHubへ保存しない。

## この段階でまだしないこと

- train / dev / test splitの固定
- corpus balancing
- BPE学習
- TinyLM学習
- private本文のGitHub commit

inventory結果を確認し、provenanceとgroup identityを監査してからsplitを固定する。

## 旧版で raw pool 自体をsourceにした場合

旧版helperで `data/raw/pool` 自体をsourceにすると、Drive名の原本
（`prose001.txt` / `dialogue003.txt` 等）と正規化後の
`prose/ja/001.txt` / `dialogue/ja/003.txt` が同居する。
Windows由来の `:Zone.Identifier` がある場合もinventory対象になる。

まず削除候補だけ表示する。

```bash
find data/raw/pool -type f \( \
  -name 'prose[0-9][0-9][0-9].txt' -o \
  -name 'dialogue[0-9][0-9][0-9].txt' -o \
  -name '*:Zone.Identifier' \
\) -print
```

表示内容がDrive名の原本とZone metadataだけであることを確認してから削除し、
`scripts/inventory_corpus.py` を再実行する。
正規化後の `prose/ja/NNN.txt` / `dialogue/ja/NNN.txt` は残す。

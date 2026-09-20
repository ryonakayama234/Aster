# 学習用本文＋BPE：最初の実行結果

2026-09-20。正式Corpus-v0の凍結前に行ったpilot。TinyLMの学習・Agent実行ではない。

## 結果

| 項目 | 実測 |
| --- | ---: |
| canonical入力 | 7,487レコード |
| 採用 | 886レコード |
| train | 110レコード・188,429 bytes |
| dev | 774レコード |
| test | 2レコード |
| 保留 | 2レコード |
| trainの領域別予算により不採用 | 6,599レコード |
| BPE語彙 | 512（特殊tokenのBOS/EOSは別途2） |
| encode→decodeの往復一致 | 886 / 886 |

## 学習に使用した本文

| 領域 | train件数 | bytes | BPE tokens | bytes/token |
| --- | ---: | ---: | ---: | ---: |
| 日本語prose | 3 | 62,539 | 32,304 | 1.936 |
| 日本語dialogue | 2 | 22,371 | 12,538 | 1.784 |
| Python | 2 | 53,569 | 29,711 | 1.803 |
| Math | 103 | 49,950 | 27,641 | 1.807 |

bytes/tokenは「1 tokenあたりにまとめられた元のUTF-8 bytes数」。大きいほどtoken数は少ない。
圧縮の指標であり、知能や学習結果の良さを表さない。token数にBOS/EOSは含めていない。

Python教材は文書構造を読み、説明・コード・入出力を取り出した。
検索／実行表示が混在するdialogue/ja/014・015は、本文範囲を勝手に決めず保留。
dialogue/ja/013は抽出できたが、今回のtrain予算を超えたため非採用。理由はdecisionsに保存。
元会話の独立性が不明なためChatGPT対話は同一groupを維持し、対話のdev/testはない。

## 追跡用ID

- canonical：`09242f8214673bb847af513d86936c8be001fad78579dda7f0dab8ca6d202d6b`
- training view：`e27055fbe666b2f8915414a2d565074c2e50225376aeadde7b85acd0065c1981`
- 本文抽出run：`b22e1cfb780c4acdb698839356c9d52b`
- 再生成確認run：`241c64943c614b83af6213603f1164ad`
- Tokenizer artifact：`0f84d39c8d7726383155e5a4f9e3dd7e0bb7782b0a3ba48b2ed1b0c28d4b160c`
- Tokenizer run：`07032cd0a170411dad43117c9b779c81`

本文抽出run内の `observation-bundle.json` とTokenizer run内の `tokenizer-bundle.json` を、今後Sitesでブラウザ内読込する。
これらは原文・対話を含むローカル成果物。サイトのソースや公開sampleへ自動的に含めない。

## 確認したこと

- テスト45件成功。RSTのコード保持、不明directive・include/rawの保留、対話整形、重複groupの分割矛盾を検査。
- 入力や出力のhash不一致を検知。失敗runも理由付きで記録。
- 同一条件で同じtraining viewを再生成でき、実行ログは別runとして残る。
- BPEにdev/testを渡さないテストと、保存viewの全ファイル検査を実施。
- 保存結果を読み、全886件のtoken往復一致を確認。

## 限界と次の一歩

- devの大部分はGSM8K公式trainからの保留問題。独立した数学testではない。
- proseとPythonのtestは各1文書のみ。広い汎化性能の評価には足りない。
- 重複の検査範囲は全文一致と空白のみが違う全文。意味的な近似・部分引用は未検査。
- 空白だけが違うものは同じ分割へまとめるが、コードの意味が変わりうるため削除はしない。
- toolデータは含まず、初回Corpus-v0の予定配合ではない。正式v0は自作枠などを揃えてから凍結する。
- 次はSitesの「素材」と「ことばを分ける」で、この実データを読み込んで観察できるようにする。

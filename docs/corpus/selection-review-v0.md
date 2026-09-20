# AsterCorpus-v0 採用表（初回調査）

この表は自動生成。出典の変更は `source-notes-v0.json` に記録する。

**候補は採用確定ではない。currentへのコピー・分割・学習はまだ行っていない。**

記事のURLはユーザー指定。閲覧できたことは全文一致や利用条件の確認完了を意味しない。
本文内リンクは参考リンクとしてJSON索引に保存し、原典URLと区別する。
ChatGPTの質問はユーザー、回答はChatGPT。モデル版と元会話の境界は未確認。
用途はまず読む素材という提案。標準応答の口調としての採用は未決定。

## 集計

| 領域 | ファイル数 | 文字数（改行・記号含む） | bytes |
| --- | ---: | ---: | ---: |
| prose/ja/ | 18 | 59,425 | 142,681 |
| dialogue/ja/ | 15 | 81,286 | 179,964 |
| math/ | 5 | 2,136 | 3,289 |
| technical/ | 1 | 1,705 | 4,647 |
| structured/ | 14 | 11,198 | 12,010 |
| external-seed/ | 10 | 4,351,963 | 4,351,978 |
| image/ | 10 | 0 | 3,812,112 |

全73ファイル。LF改行統一後の完全一致重複：0組。近似重複と意味内容の正しさは未検査。

## 状態

| 状態 | 意味 | 件数 |
| --- | --- | ---: |
| candidate_needs_review | 候補：内容・用途・利用条件を確認 | 10 |
| candidate_needs_transform | 候補：出典固定済み、整形とレコード単位の分割が必要 | 5 |
| design_reference_needs_execution | 設計資料：実行・検証済み経験としては未採用 | 14 |
| excluded_os_metadata | OS付加情報：学習対象外 | 1 |
| held_outside_text_v0 | 画像：テキスト版v0では保留 | 9 |
| metadata_only | 出典資料として保存、学習本文に入れない | 5 |
| needs_original_source | 確認待ち：原典・作品・個別ページが未特定 | 23 |
| needs_source | 確認待ち：作り手・出典が未確認 | 6 |

## ファイルごとの採用案

原典グループは未確定のものを広めに仮まとめしている。同じ著者・同じ会話という断定ではない。

| pool内のファイル | 文字数 | 由来 | 原典・出典 | グループ | 状態 |
| --- | ---: | --- | --- | --- | --- |
| `dialogue/ja/001.txt` | 124 | manga_transcription | 未特定／対象外 | unresolved-manga-batch | needs_original_source |
| `dialogue/ja/002.txt` | 80 | manga_transcription | 未特定／対象外 | unresolved-manga-batch | needs_original_source |
| `dialogue/ja/003.txt` | 178 | manga_transcription | 未特定／対象外 | unresolved-manga-batch | needs_original_source |
| `dialogue/ja/004.txt` | 537 | manga_transcription | 未特定／対象外 | unresolved-manga-batch | needs_original_source |
| `dialogue/ja/005.txt` | 23223 | manga_transcription | 未特定／対象外 | unresolved-manga-batch | needs_original_source |
| `dialogue/ja/006.txt` | 1476 | external_qa | [出典](https://scrapbox.io/marshmallow-rm/) | marshmallow-qa-batch | needs_original_source |
| `dialogue/ja/007.txt` | 1874 | external_qa | [出典](https://scrapbox.io/marshmallow-rm/) | marshmallow-qa-batch | needs_original_source |
| `dialogue/ja/008.txt` | 306 | external_qa | [出典](https://scrapbox.io/marshmallow-rm/) | marshmallow-qa-batch | needs_original_source |
| `dialogue/ja/009.txt` | 1297 | external_qa | [出典](https://scrapbox.io/marshmallow-rm/) | marshmallow-qa-batch | needs_original_source |
| `dialogue/ja/010.txt` | 4355 | user_ai_conversation | 未特定／対象外 | unresolved-chatgpt-sessions | candidate_needs_review |
| `dialogue/ja/011.txt` | 6023 | user_ai_conversation | 未特定／対象外 | unresolved-chatgpt-sessions | candidate_needs_review |
| `dialogue/ja/012.txt` | 300 | social_transcription | 未特定／対象外 | unresolved-x-dialogue | needs_original_source |
| `dialogue/ja/013.txt` | 8234 | user_ai_conversation | 未特定／対象外 | unresolved-chatgpt-sessions | candidate_needs_review |
| `dialogue/ja/014.txt` | 5491 | user_ai_conversation | 未特定／対象外 | unresolved-chatgpt-sessions | candidate_needs_review |
| `dialogue/ja/015.txt` | 27788 | user_ai_conversation | 未特定／対象外 | unresolved-chatgpt-sessions | candidate_needs_review |
| `external-seed/gsm8k/LICENSE` | 1062 | external_dataset | [出典](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE) | 未設定 | metadata_only |
| `external-seed/gsm8k/README.md` | 6112 | external_dataset | [出典](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md) | 未設定 | metadata_only |
| `external-seed/gsm8k/grade_school_math/data/train.jsonl` | 4166206 | external_dataset | [出典](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl) | derive_problem_ids_before_split | candidate_needs_transform |
| `external-seed/python-tutorial/Doc/copyright.rst` | 447 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Doc/copyright.rst) | 未設定 | metadata_only |
| `external-seed/python-tutorial/Doc/license.rst` | 55511 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Doc/license.rst) | 未設定 | metadata_only |
| `external-seed/python-tutorial/Doc/tutorial/controlflow.rst` | 40752 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Doc/tutorial/controlflow.rst) | python-chapter:controlflow | candidate_needs_transform |
| `external-seed/python-tutorial/Doc/tutorial/datastructures.rst` | 24846 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Doc/tutorial/datastructures.rst) | python-chapter:datastructures | candidate_needs_transform |
| `external-seed/python-tutorial/Doc/tutorial/errors.rst` | 24234 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Doc/tutorial/errors.rst) | python-chapter:errors | candidate_needs_transform |
| `external-seed/python-tutorial/Doc/tutorial/introduction.rst` | 18984 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Doc/tutorial/introduction.rst) | python-chapter:introduction | candidate_needs_transform |
| `external-seed/python-tutorial/LICENSE` | 13809 | external_documentation | [出典](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/LICENSE) | 未設定 | metadata_only |
| `image/BSD予想の標語.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/EllipticCurve.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/L関数の絶対値.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/Rumor P=NP.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/UsainBoltObservedSplitSpeed.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/UsainBoltObservedSplitSpeed.png:Zone.Identifier` | — | unknown | 未特定／対象外 | 未設定 | excluded_os_metadata |
| `image/ギャル.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/ギャル2.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/ギャル3.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `image/ギャル4.png` | — | unknown | 未特定／対象外 | 未設定 | held_outside_text_v0 |
| `math/en/001.txt` | 194 | unknown | 未特定／対象外 | 未設定 | needs_source |
| `math/ja/001.txt` | 64 | unknown | 未特定／対象外 | 未設定 | needs_source |
| `math/ja/002.txt` | 1250 | unknown | 未特定／対象外 | 未設定 | needs_source |
| `math/ja/003.txt` | 71 | unknown | 未特定／対象外 | 未設定 | needs_source |
| `math/ja/004.txt` | 557 | unknown | 未特定／対象外 | 未設定 | needs_source |
| `prose/ja/001.txt` | 9119 | external_article | [出典](https://theseus.hatenablog.com/entry/2026/09/12/221731) | article-calculemus | candidate_needs_review |
| `prose/ja/002.txt` | 6204 | external_article | [出典](https://tsujimotter.hatenablog.com/entry/BSD-conjecture-part1) | article-bsd | candidate_needs_review |
| `prose/ja/003.txt` | 9529 | external_article | [出典](https://qiita.com/kyamaz/items/7b38e5977dc97747d2d9) | article-lean-trust | candidate_needs_review |
| `prose/ja/004.txt` | 2683 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/005.txt` | 2615 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/006.txt` | 1709 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/007.txt` | 1332 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/008.txt` | 2009 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/009.txt` | 8520 | external_article | [出典](https://zenn.dev/k16/articles/f468f28a08a764) | article-tex-lean | candidate_needs_review |
| `prose/ja/010.txt` | 12107 | external_article | [出典](https://hsjoihs.hatenablog.com/entry/2026/09/17/101650) | article-adverbs | candidate_needs_review |
| `prose/ja/011.txt` | 633 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/012.txt` | 833 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/013.txt` | 368 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/014.txt` | 360 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/015.txt` | 341 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/016.txt` | 292 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/017.txt` | 359 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `prose/ja/018.txt` | 412 | internet_meme | 未特定／対象外 | unresolved-memes-batch | needs_original_source |
| `structured/tool_json/001.jsonl` | 2372 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/002.jsonl` | 3275 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/003.jsonl` | 725 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/004.jsonl` | 495 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/005.jsonl` | 478 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/006.jsonl` | 432 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/007.jsonl` | 266 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/008.jsonl` | 272 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/009.jsonl` | 413 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/010.jsonl` | 996 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/011.jsonl` | 294 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/012.jsonl` | 488 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/013.jsonl` | 389 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `structured/tool_json/014.jsonl` | 303 | unknown | 未特定／対象外 | 未設定 | design_reference_needs_execution |
| `technical/ja/001.txt` | 1705 | unknown | 未特定／対象外 | 未設定 | needs_source |

## 次に確定すること

1. 漫画の作品名と巻・話、Scrapboxの個別ページ、Xの元投稿。原典不明のミームは原本を保管して保留する。
2. ChatGPT対話の口調を応答のお手本にするか。元会話が共通なら同一groupへまとめる。
3. math/ja・math/en・technical/jaの出典または作り手。structured資料の生成元・実行履歴。
4. 記事の抽出範囲と利用条件、対話の発言境界、画像参照、実行表示を点検する。
5. 採用ファイル・採用範囲を固定し、原典group単位で分割してからcurrentへコピーする。

構造化資料はJSON解析成功だけでは実行済みと扱わない。形式結果と注意点は `pool-inventory-v0.json` に保存。
この調査では原本を書き換えず、外部コードも実行していない。

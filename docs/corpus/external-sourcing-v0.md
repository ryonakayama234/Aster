# AsterCorpus-v0 外部調達仕様

2026-09-15の初回提案。原本の収集と学習用採用を区別する。

## 採用候補

| 領域 | v0の第一候補 | 採る内容 | 適用範囲 |
| --- | --- | --- | --- |
| Math | GSM8K mainの公式train | 問題＋解答を一組として選択 | 英語の算数文章題。数学全般の教材ではない |
| Python | CPython v3.13.0の英語公式tutorial | introduction / controlflow / datastructures / errors | 文法、データ構造、例外、短い入出力。コード課題集ではない |

GSM8Kは[公式README](https://github.com/openai/grade-school-math)に問題・解答形式と公式分割が記載されている。
[MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE)と著作権表示を原本とともに保存する。
Socratic版とモデル生成解答は初回に混ぜない。

Python教材は[公式license](https://docs.python.org/3/license.html)と固定版の権利表示を保存する。
本文のPSFライセンスと、例示コードに追加適用されるZero-Clause BSDを区別する。
本文込みの教材全体を0BSDとして扱わない。配布時は対応する表示・条件と変換内容を添付する。
最新追従より再現性を優先し、取得したcommitをmanifestで固定する。

## 変換契約（次の実装対象）

### GSM8K

- `question` と `answer` が空でない文字列であることを検査。
- 一レコードを `Question:\n…\n\nAnswer:\n…` として出力し、原本の行番号をupstream IDにする。
- 計算注釈 `<<…>>` は除去する。原本は保存し、注釈外の式・数値と最終答 `####` は保持する。
- 理由：Math枠を問題・解答に限定し、toolプロトコルはAster側で統一するため。
- 最終答の抽出可否を検査する。これは解答の数学的正しさを保証しない。採用候補を目視で点検する。
- 公式testは今回未取得。評価段階で別の保留領域へ取得する。

### Python tutorial

- RST原本を保管し、説明段落＋対応する例を節ごとに取り出す。章を分割groupとする。
- navigation、index、参照用directiveを本文へ流し込まない。意味のある説明と入出力は残す。
- `>>>` / `...` の対話例と独立Pythonコードを区別する。例外を示す例も正しい教材として保持。
- コード例があることを、その節全体が実行可能なPythonであることと混同しない。
- 外部コードは収集時には実行しない。実行評価用への変換は別工程とする。
- 初期の外部領域は英語。日本語の数学・Python説明への転移は保証されない。翻訳はv0に黙って追加しない。

## 今回採らないもの

- 大規模Web/GitHub丸ごと取得：出典整理・難易度・重複が初回の学習を複雑にする。
- 競技数学、大量の合成推論：初歩の比較から始めるため、v0.1以降の候補。
- 外部JSON/tool会話：ユーザーの方針に従いAster内部で作る。
- MBPP：将来の短いPython課題候補。[公式説明](https://github.com/google-research/google-research/tree/master/mbpp)では問題・コード・テストとID別分割が示されている。採用前にデータ固有の権利条件を確認する。今回は取得しない。

## 収集の再現と確認

`scripts/fetch_external_seed.py` は許可した二つのリポジトリの選定ファイルだけを取得する。
初回にcommitを解決し、二回目以降はmanifestのcommitとハッシュを使用する。
収集先は `data/raw/pool/external-seed/`。原本本文はGit管理外、manifestと収集器はGitで追跡可能。
収集途中に失敗した原本は学習に使わない。未登録ファイルがある状態での初回再実行は停止するため、必要なら別ディレクトリへ退避してやり直す。

取得検査はUTF-8、ファイルhash、GSM8KのJSONと必須フィールド。
この段階では本文の品質、重複、train/dev分割、教材抽出、TinyLM性能を検証していない。
学習用への採用条件は [コーパス仕様](AsterCorpus-v0.md) を参照する。

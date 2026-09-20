# canonicalから学習用本文を作る設計

2026-09-20。pretrain pilotの本文抽出・分割・配合・training出力とBPE計測を実装済み。
SFT、意味的な近似重複検査、正式Corpus-v0凍結、TinyLM学習は未実装。

## 実行方法

WSLのリポジトリルートで実行する。新規環境では先に `python -m pip install -e .` で依存を入れる。

```sh
.venv/bin/python scripts/build_training.py
.venv/bin/python -m aster.training.tokenizer_run --view data/training/<出力されたview ID>
```

recipeは `configs/training-pilot-v0.json`。canonical buildを固定し、元データを勝手に最新版へ切り替えない。
原典groupごとのsplit指定とtrainの領域別bytes上限を保存している。
最初の実測は [reports/training-pilot-v0.md](../../reports/training-pilot-v0.md) を参照。

RSTの構造読取には固定版Docutils 0.22.2を使用する。
[publish_doctree](https://docutils.sourceforge.io/docs/api/publisher.html#publish-doctree)で文書構造を作り、
[設定](https://docutils.sourceforge.io/docs/user/config.html)で外部includeとrawを無効化している。
CPython教材で使われるmethod・seealso・参照role等に対応し、未知の記法は理由付きで保留する。
これはSphinx全体を実装するものではない。

## 目的

共通形式のcanonicalから「この実験で何を予測させるか」に合わせた入力を作る。
本文だけを保存するだけでなく、原本からどの部分を残し、何を除いたかをUIで辿れるようにする。
canonicalを直接書き換えず、元のbuildを固定したtraining viewを生成する。

## 処理の順序

1. **入力を固定**：canonical build IDとmanifest・records hashを確認する。latestの自動選択をしない。
2. **採用を指定**：採用台帳でrecord IDまたは原典groupを選ぶ。用途はpretrain/sft/evalを区別する。
3. **本文を抽出**：kindとtext_formatに応じた抽出器を使う。除去・保留理由と対応範囲を記録する。
4. **重複と派生を確認**：元文書group、本文の完全一致、近似一致候補を使って分割してはいけない単位を確定する。
5. **分割**：その単位ごとにtrain/dev/testへ割り当て、割当表を固定する。
6. **配合**：trainだけから予算に合わせて本文を選ぶ。実際のbytesと件数を記録する。
7. **出力**：本文、対応表、保留一覧、実行記録を一つの版として保存する。
8. **token化**：trainで学習したTokenizerを、同じ版のtrain/dev/testへ適用する。

長い原稿の節への分割やtoken数による学習window化は、元文書の分割先を固定してから行う。
文書の前半をtrain、後半をtestへ分けない。dev/testを配合の都合で選び直さない。
GSM8Kの原典testは別保管。今回ある公式trainの派生を、新規の独立testと呼ばない。

## 二つの学習用途

### pretrain：次のtokenを予測する

文章・会話を連続した本文として学ぶ。会話の表現に触れることもここに含まれる。
会話を読むことは学習に影響するため、pretrainなら口調へ影響しないとは言わない。
初回はこのviewを実装した。

### sft：応答のお手本から学ぶ

会話の質問を入力条件とし、選定したassistantの応答を予測対象にする。
roleとloss mask（どのtokenを採点して学ぶかの指定）が必要。
ChatGPT対話の全回答を自動的にお手本としない。pretrain採用とsft採用は独立して指定する。
初回のpretrain用テキストをそのままsft入力に流用しない。

## 種類ごとの抽出

| canonical | 本文として残す | 別記録へ移す／保留 |
| --- | --- | --- |
| document / plain | 本文・必要な見出し・式・コード | 独立した管理情報、表示用の付加情報。判定できないものは保留 |
| document / rst | 説明、対応するコード・入出力、意味のある見出し | navigation/index等の表示用指示。RST構造を読んで抽出する |
| conversation | 確定した話者ラベルと発言本文 | 実行表示・取得ログなど。発言と混在して分離不能ならそのレコードを保留 |

Python教材は正規表現で記号を一括削除しない。構造を読むパーサーで段落・コード・出力を識別する。
未対応directiveや外部includeは黙って展開せず、明示した対応範囲外として記録する。外部アクセスやコード実行はしない。
既知の小さな対応形式から始め、実例で抽出前後を確かめて対応範囲を広げる。

GSM8Kはcanonicalですでに計算注釈を除去しているため、二重に処理しない。
plain文書でも、ファイル内のURLをすべて削除するような処理は行わない。
AIが説明のために書いたPythonコードは教材として残せる。コードだから削除するのではなく、発言・実行表示の由来で区別する。
検索表示が発言に混入した既存ログは、原典ごとの範囲指定を例外設定として保存し、毎回手で編集し直さない。
例外設定は対象canonicalのhashに結び、元文書が変わったときは再確認する。

## 対話のテキスト化

pretrainでは、最初は普通の文字列の話者ラベルを使う案とする。
たとえば説明用の例として `User:\n…\nAssistant:\n…` と並べる。
これは専用tokenを新設することではない。ラベル、空行、終端改行をrecipeに固定する。
会話全体を一つのsampleとして保持し、source message番号から本文の対応範囲を辿れるようにする。
文書境界には既存TokenizerのBOS/EOSのIDをデータセット側で付与する。単に文字として書き込まない。

## 出力案

```text
data/training/<view_id>/
  manifest.json         # 元build・recipe・割当・出力hash
  samples.jsonl         # 採用ID・原典group・分割・本文パス
  decisions.jsonl       # 採用／除外／保留と理由
  transformations.jsonl # 原文から本文への対応・変換内容
  train/*.txt           # 既存BPEビルダーへ渡す本文
  dev/*.txt
  test/*.txt

runs/<run_id>/
  run.json              # 実行条件、入力版、出力版、成否
  events.jsonl          # 進行と結果。UIでもCLIでも同じ記録を読む
```

本文の除去前後を後から再計算して説明するだけでなく、実際に使用した抽出器版・範囲・理由を記録する。
初回は文書・段落・発言単位の追跡で十分。逐文字の万能な差分エンジンは先に作らない。
generatedなtraining本文と私的ログはGitへ誤追加しない運用を実装時に確認する。

## 配合と凍結

現在のcanonical全量はGSM8Kが大部分を占める。全件の連結を学習recipeにしない。
既存Corpus-v0の目標比率と実際に採れる量を比較し、採用可能な本文bytesで予算を決める。
自作structured/toolの枠はまだ実行検証済みデータがない。外部データで黙って埋めない。
先行して本文抽出やBPEを見る場合はpilot viewと明記し、正式なCorpus-v0凍結と区別する。
正式v0は自作枠・分割・配合を確定してから凍結する。

## 最初の実装の完了条件

- 一つのコマンドで固定canonicalからpilot viewを再生成できる。
- 正常なテキスト、対話、Python教材の代表例で期待する本文・インデント・数式を確認する。
- 未対応入力を理由なく落とさず、保留として一覧に残す。
- 異なるsplitに同一groupと完全一致本文が混ざらない。近似重複検査の範囲を明記する。
- 同じ入力・設定で同じ本文とmanifestになる。日時等の運用イベントとは分離する。
- 原本→canonical→学習本文の対応をUI用データとして辿れる。
- 既存Tokenizerにはtrainディレクトリだけを渡す。損失計算やモデル学習を完了扱いにしない。

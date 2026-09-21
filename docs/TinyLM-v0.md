# TinyLM v0：最初の事前学習

目的は、固定した本文とBPEを使って「次tokenを予測する → 誤差を測る → 重みを更新する」を一周させることです。
ランダムな重みから始める小さな事前学習です。BPEのルール作成に続き、今度はembeddingやTransformerの数値を学習します。

## 触る場所と読む順序

| ファイル（src/aster以下） | 役割 |
| --- | --- |
| training/dataset.py | 既存viewの検証、token化、入力と正解を1個ずらして作る |
| model/transformer.py | 未来を見ないAttention、MLP、残差接続 |
| model/lm_head.py | 各tokenの予測スコアへ変換 |
| model/tiny_lm.py | embedding・位置・Transformer・出力を接続 |
| training/pretrain.py | 誤差、逆伝播、AdamWによる更新、評価と記録 |
| model/checkpoint.py | 重み・設定・Tokenizerをまとめて保存／再読込 |
| inference/generate.py | 重みを更新せず次token候補と続きを観察 |
| records/runlog.py | 開始、step、評価、完了／失敗を記録 |

例えばtoken列が `[BOS, A, B, EOS]` なら、入力は `[BOS, A, B]`、正解は `[A, B, EOS]` です。
正解を先に見せず、各位置で次の一つを当てます。短い文書の余白は誤差に含めません。
文書同士は連結せず、長い文書はcontext長ごとに分けます。隣接ペアは一度ずつ使い、窓の境界では過去文脈と位置をリセットします。

## WSLで実行する

PRを取り込んだリポジトリのルートで、既存の仮想環境を使います。まずCPU版で試します。

```sh
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[training]' pytest
.venv/bin/python -m pytest -q
```

`--view` はBPEを作った学習用viewのディレクトリ、`--tokenizer` はそのBPEの出力ディレクトリです。
後者には `evaluation.json` とTokenizerの保存ファイルが必要です。Siteの「BPE処理が完了」の `output` が該当します。
以下の `<...>` は手元の実際の値に置き換えます。

```sh
.venv/bin/python scripts/train_tinylm.py \
  --view 'data/training/<view ID>' \
  --tokenizer 'runs/workbench/<BPE実行ID>/artifacts/tokenizers/<Tokenizer実験ID>' \
  --config configs/tinylm-overfit-v0.json
```

最初は **overfit** です。record ID順の先頭1窓を200step繰り返し、覚えられるかを確認します。
学習前のstep 0と50stepごとに、損失、次token上位候補、同じ入力からの生成を記録します。
この段階の損失低下は暗記の確認で、未知の文章への強さを意味しません。

次に同じコマンドの設定を `configs/tinylm-pilot-v0.json` に変えます。
**pilot** はtrain全体から更新し、全train／全devの損失をtoken数で重み付けして比較します。
分野別の損失も保存します。testは整合性検証のためファイルを確認しますが、token化・学習・予測評価には使いません。
200stepは動作確認の初期値で、十分な学習量を保証しません。

設定は2層、幅64、4 heads、context 128、batch 8、seed 42です。
語彙数は保存済みTokenizerから決まり、通常語彙512ならBOS/EOSを含め514です。

## 実行後に見るもの

出力先は `runs/<新しい実行ID>/` です。

| ファイル | 観察する内容 |
| --- | --- |
| run.json / events.jsonl | 実行状態と時系列、処理した正解token数 |
| experiment.json | 設定、view／Tokenizer ID、実装のhash、Python／torch版 |
| training-windows.json | 実際に更新へ使った文書IDと窓の位置 |
| training-bundle.json | 学習前後の損失、分野別評価、候補確率、生成結果 |
| checkpoint-000200.pt | 重み、モデル設定、Tokenizer、実験情報 |

checkpointは単独で生成に使えます。学習完了時には、再読込した重みで予測が完全一致することも確認します。

```sh
.venv/bin/python -m aster.inference.generate \
  --checkpoint 'runs/<新しい実行ID>/checkpoint-000200.pt' \
  --prompt '観察したい書き出し' --max-new-tokens 32
```

生成は最大確率を選ぶgreedy方式です。BOSを候補から除外し、EOSで終了します。
上位候補の表示は加工前の確率です。文脈が長くなった場合は直近context個を使います。
byte列がUTF-8として成立しない場合もあり、表示用文字列とともに `utf8_valid`・元のtoken ID・bytesを残します。

## 確認できたことと残ること

2026-09-21、Python 3.12／PyTorch 2.14.0+cpuで人工の小さなJSON・コードfixtureを使って確認しました。
幅32・2層・100stepの暗記試験ではtrain lossが **5.5402 → 0.0252** に下がり、
`{"result` から `{"result":42}` と改行まで復元しEOSで終了しました。
別Pythonプロセスでcheckpointを読み直して同じ続きを生成しました。
未来tokenの遮断、余白の除外、文書境界、test除外、dev評価、入力改変時の失敗記録もテストします。

これは処理の接続確認です。手元の私的Corpusでの学習・汎化評価はまだ実行していません。
checkpointにはoptimizer状態を含めないため、学習の途中再開には未対応です。再実行は新しい実験になります。
現在はCPUのみで、dropout・混合精度・分散学習はありません。

`aster-training-bundle-0` は今回追加した学習結果形式です。既存SiteのBPE結果読込にはまだ接続していません。
次の段階で「学習を観察する」に損失曲線と同じ書き出しの学習前後比較を接続します。
Agentの道具呼び出しや自律実行の学習は、その先の段階です。

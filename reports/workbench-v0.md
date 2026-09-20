# Sites workbench 初回確認

2026-09-20。保存結果の読込と観察が対象。Python実行・モデル学習・Agent実行は対象外。

Sitesの本人限定公開が成功： https://aster-learning-lab-zhong.rynaka0112.chatgpt.site
UIのsource commit：`70a2a0d15f57993b5a472041cdc8d801bbe01b9c`。

## 実装済み

- observation-bundle.jsonをブラウザ内で読込。領域・split別に素材を選び、canonicalと学習用本文を比較。
- Tokenizer bundleを読込み、入力文を実際のBPEで分割。適用規則数を変え、ID・bytes・復号を表示。
- tokenが単独でUTF-8文字として復号できない場合、bytesの16進表示を使う。
- 考察をview・画面・素材に紐づけてブラウザ内へ保存。JSONへ書き出せる。
- TinyLMとAgentの画面は準備中として示す。架空の学習値は表示しない。

## 確認

ローカルHTTPの画面をブラウザで操作し、実際の二つのbundleを読み込んだ。
本文選択、Python教材の抽出表示、考察保存、BPEの段階変更、token選択を確認。
日本語・Python・絵文字・結合文字・空文字の入力でPython版とJavaScript版のtoken ID列が一致。
スマートフォン幅390pxで横方向のはみ出しなし。ブラウザ実行エラーなし。
ファイルを読んだ操作でサーバーへのPOST等のアップロードなし。
スクリーンショットは `reports/generated/workbench/` に保存し、表示を確認した。

WebMCPはテストregistryで登録・正常入力・不正入力を確認。実ブラウザのネイティブWebMCP対応は未確認。
サイト公開時の私有アクセス判定はSitesが担当する。原本や対話本文はSiteの配信物に含めない。

## 利用ファイル

`artifacts/workbench-inputs/e27055fbe666b2f8915414a2d565074c2e50225376aeadde7b85acd0065c1981/`
に同じviewのobservation-bundle.jsonとtokenizer-bundle.jsonをまとめた。
画面の「実験ファイルをひらく」で両方選ぶ。ファイル読込はブラウザ内だけで、リロード後は再度選択する。
考察はブラウザに残るが、端末間同期はない。

サイトを使うために原典不明素材の調査や追加執筆を終える必要はない。
次は実行接続の方式を小さく検証し、画面から実験を開始・監視できるようにする。

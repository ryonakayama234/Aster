# Decision失敗・Tokenizer監査 v0（2026-09-30）

PR #26の保存済み6モデル（seed42/43/44 × fixed/shuffle）の48 dev predictionを再集計した。
新しい推論・学習は行わず、checkpoint identity/model/Tokenizer hash、suite/case/input digest、token長、target/candidate/正誤を照合する。
元ユーザーWSLのfitと、ChatGPT Linux上のdev測定の由来を維持する。
[機械可読監査](decision-failure-audit-v0.json)にはsource predictionのSHA-256、元checkpointと診断Run、新しいaudit Runを記録した。

## どこで崩れたか

- キー変更2課題×6モデルのphase0計12判断はすべて誤答。正解calculatorに対しmemory.getが8回、policy_stopが4回。まだmemory操作をする前から変化に弱い。
- 数値変更のphase0計12判断は9回誤答し、すべてmemory.getを選んだ。
- この12は同じ2派生課題への6モデルの観測で、独立した12課題ではない。
- 遅延probeではcalculator→policy_stopが14回、put→getが10回など、多段階に誤答が分布する。元stepだけで説明できると断定しない。

## 入力の分割

全6 checkpointでTokenizer digestは同一。正解候補calculatorのphase0は次のように変わった。

| 入力 | BOS/EOS込みtoken数 | 元trainの全候補入力に現れないtoken IDの出現回数 |
|---|---:|---:|
| 元add/subtract | 10 | 0 / 0 |
| 数値変更add/subtract | 23 | 11 / 11 |
| キー変更add/subtract | 24 | 6 / 7 |

元train専用BPEの最長tokenは129bytes。例えば元addの入力では、left=2・operation=add・right=3を含む複数fieldが一つに結合され、taskの大きな部分も一つに結合されている。数字や保存先が変わるとその結合が解ける。
全phaseの正解入力token長はseen10〜31、数値23〜62、キー24〜107、遅延70〜137。いずれもcontext2048内で、encode処理も超過を例外として拒否するため、この測定は入力切捨てによる結果ではない。
ここでの「trainに現れないtoken」はUNKや復元不能を意味しない。byte-level BPEは入力を表現できるが、元trainの入力としてそのIDが使われなかったという観測である。

## 仮説と次の測定

数値/キーを含むtrain固有の長いtokenと、変化後の入力分布への弱さが共存している。Tokenizerが唯一の原因だと証明したわけではなく、モデル容量・入力表現・最適化・学習課題の狭さも残る。
先にTokenizerを固定した[データ4条件比較](../docs/DecisionDataIntervention-v0.md)を測る。train fitが崩れる条件のdev失敗を、汎化だけの問題へ帰属しない。
その後必要なら、同じデータ・予算で、現在の専用BPEと数値/キーの分割を保ちやすいTokenizerを別条件として比較する。ByteTokenizerは系列が長くなるため、同じ提示回数が同じ計算量ではないことも記録する。

## 再現

```bash
.venv/bin/python scripts/audit_decision_failures.py --root . \
  --diagnostic-run runs/<diagnostic-run-id> \
  --checkpoint runs/<original-fit-run-id>/arms/eight-shuffle/checkpoint
```

新Runのaudit.jsonとtoken-cases.jsonlに分割片・token ID・誤Actionを保存する。既存診断の内容と指定checkpointが一致しなければ停止する。sealed testは入力に使わない。

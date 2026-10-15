# READ v0 — 作業・検証ログ
2026-10-15 JST。作業指示書: [TODO](../TODO/ReadEvidence-v0.md)。研究Gate: [#113](https://github.com/ryonakayama234/Aster/issues/113) / [#115](https://github.com/ryonakayama234/Aster/issues/115) / [Draft PR #116](https://github.com/ryonakayama234/Aster/pull/116)。

## 2026-10-15: 実験を開始する前の計画確定（design-only）
- **実施済み:** GitHubに存在する設計/Run契約/現行Gateを照合し、実装の分割・検収・停止条件を定義。Wolframで6×8=48 pair、2×48=96 world、2×96=192 candidate exposures、固定位置対照24/48、暫定目安36/48=75%を算術検算。
- **未実施:** #113-Aのprivate reportの再検査、#113-Bの新資料選定、READ model/serializer固定、fixture generator、G0/G1、モデル評価G2。**READ結果=NOT_RUN**。
- **変更しない:** 既存のv0結果/Artifact、sealed test、private資料、Model weights、Tokenizer、学習量、Agent runtime、GitHub Actions workflow。
- **次の一問:** 保存済みLANG v0.3の6つのprefix介入は、元の先行文脈の方が正しいtokenに高い確率を与えたか（step0/200別）？
- **開始条件:** ユーザーWSL上の既存private Runとハッシュ、古いspecに適合したread-only集計器の確認。結果が得られない場合はBLOCKEDで記録し、推測しない。

### Gateごとの事実記録（後で実測値に置き換える。推測で埋めない）
| Gate | Status | Evidence / 最初の入力 | 未完了 |
| --- | --- | --- | --- |
| LANG #113-A | NOT_RUN | private reportのdigest・6-case signed benefit | user WSL private Run読取 |
| LANG #113-B | PROPOSED | 未使用prose/出典group設計 | user選定後にmanifest固定 |
| READ DESIGN | PROPOSED | 一候補選択能力・model比較の仕様 | artifact/serializer/閾値未凍結 |
| READ G0/G1 | NOT_RUN | 48 pair proposal・独立Verifier | generator/検査未実装 |
| READ G2 | NOT_RUN | 0 scored / 0 trained | 独立した固定protocolが必要 |

## 最終報告に必須の項目
cycle_id、source_git_sha、manifest_sha256、model/Tokenizer/serializer ID、同一family由来、対象全件/適格/blocked、worldごとの選択と根拠、primary、各対照、全失敗、CPU時間/RSS、Trialの限界、次の一問。外部提供するのは許された安全なaggregateのみ。

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

## 2026-10-15: 文献と優先順位の再点検（design-only）
- **実施済み:** GitHub #113/#115、Draft PR #116、当該architecture/TODO/実験台帳を照合。Wolframで6×8=48 pairs /96 worlds/192 candidate exposures、二候補の独立ランダム両world正答確率1/4を算術確認（後者はworld間独立を仮定した理想対照で、固定位置やfamily相関がある実験の帰無分布ではない）。
- **設計上の追加:** [Liu et al., TACL 2024](https://aclanthology.org/2024.tacl-1.9/) の根拠位置依存を踏まえ、既存の位置バランスと順序交換probeを保持する根拠を[READ設計](../EvidenceUse-v0-plan.md)へ追記。RAG原著/Pointer Sentinelも転用限界を明記。
- **進行Gateは変更なし:** #113-A private Runのread-only同一性確認・signed benefit集計が最初。私的Runへのアクセスなし。#113-Bの素材選定も未開始。
- **実験結果:** READ G0/G1/G2は**NOT_RUN**、学習/追加forward/新しい実測値は0。既存検証状態を改ざんせず、旧Runの成功claimを拡張しない。


## 2026-10-15 実装単位の具体化（設計文書の追記のみ）
- 既存GitHubの`DecisionHead.forward`/`DecisionModel.forward`、`CandidateBuilder`、Agent `run_loop`を照合。DecisionModelは候補スカラー採点用だが現在の訓練対象はAction候補であり、READ候補への互換性は未確定と判断。
- READを`schema / candidate enumeration / serializer / frozen selection adapter / byte copier + independent verifier / trial runner`へ分解し、[実装TODO](../TODO/ReadEvidence-v0.md)へ各責任と検収条件を追記。[architecture](../architecture.md)にもoffline評価の境界図を記録。
- Wolfram計算による設計値の整合: 6×8=48 pairs、×2=96 worlds、×2=192 candidate scores。value-change pairで**正解候補位置を両worldで固定**し、位置バランスが各family 4/8なら固定位置対照は24/48。36/48=75%。この条件が崩れれば対照期待値を再設計。
- 実行状態は以前と同じ: #113-Aのprivate report集計**NOT_RUN**、READ model eligibility未凍結、READ G0/G1/G2 **NOT_RUN**、学習0、追加forward0。合格値を先取りしていない。機能成功とCI PASSを混同しない。

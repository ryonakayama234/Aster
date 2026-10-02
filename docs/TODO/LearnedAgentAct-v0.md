# TODO: Learned Agent ACT v0

Issue #48。各項目は実装・検証・実測を分けて更新する。

- [x] SEE実機browser確認をPR #47へ記録しReady化
- [x] ACT v0 Issue #48を作成
- [x] ACT v0 Specを固定
- [x] `decision_model` をArtifact kindへ追加
- [x] DecisionModel Service catalogのlist / resolveを実装
- [x] 既存DecisionModel artifactをimmutable catalogへ登録するローカルCLIを実装
- [x] decision_fit checkpoint → uncalibrated DecisionModel Artifact promotionを実装
- [x] promotion後に calibration=not_run / routing=not_configured を保持
- [x] promotion score一致/lineageのunit test
- [x] catalog登録/改ざん/path拒否のunit test
- [x] model-only learned Agent runtime runnerを実装
- [x] 固定 `act-calculate-store-dev-v0` taskをrunnerへ固定
- [x] `decision-traces.jsonl` を保存
- [x] weights unchanged監査
- [x] DecisionTrace.selected == Transition.action監査
- [x] Wiring Gate / Capability Gateをagent bundleへ保存
- [x] `agent-decision-model-v0` Service recipeをallowlist
- [x] Service requestはlogical DecisionModel IDだけを受ける
- [x] Service contract / AGENTS / changelog / development statusを同期
- [x] pytest / Pyright
- [ ] ユーザーWSL2でfit checkpoint本体の有無を確認
- [ ] checkpointがある場合はpromotionしてDecision artifactを登録
- [ ] checkpointが無い場合はarchive復元またはfit study再実行
- [ ] ユーザーWSL2でACT Runを実行
- [ ] Wiring PASS/FAILを記録
- [ ] Capability PASS/FAILを記録
- [ ] aster-webへDecision evidenceを接続
- [ ] 実browserでModel → scores → selected → executed → Observationを確認

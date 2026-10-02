# TODO: Learned Agent ACT v0

Issue #48。各項目は実装・検証・実測を分けて更新する。

- [x] SEE実機browser確認をPR #47へ記録しReady化
- [x] ACT v0 Issue #48を作成
- [x] ACT v0 Specを固定
- [ ] `decision_model` をArtifact kindへ追加
- [ ] DecisionModel Service catalogのlist / resolveを実装
- [ ] 既存DecisionModel artifactをimmutable catalogへ登録するローカルCLIを実装
- [ ] catalog登録/改ざん/path拒否のunit test
- [ ] model-only learned Agent runtime runnerを実装
- [ ] 固定 `act-calculate-store-dev-v0` taskをrunnerへ固定
- [ ] `decision-traces.jsonl` を保存
- [ ] weights unchanged監査
- [ ] DecisionTrace.selected == Transition.action監査
- [ ] Wiring Gate / Capability Gateをagent bundleへ保存
- [ ] `agent-decision-model-v0` Service recipeをallowlist
- [ ] Service requestはlogical DecisionModel IDだけを受ける
- [ ] Service contract / AGENTS / changelog / development statusを同期
- [ ] pytest / Pyright
- [ ] ユーザーWSL2でDecision artifactを登録
- [ ] ユーザーWSL2でACT Runを実行
- [ ] Wiring PASS/FAILを記録
- [ ] Capability PASS/FAILを記録
- [ ] aster-webへDecision evidenceを接続
- [ ] 実browserでModel → scores → selected → executed → Observationを確認

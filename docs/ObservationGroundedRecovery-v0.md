# OBS v0a — Observation-Grounded Read Recovery Protocol

Status: **design freeze candidate; no implementation/run/learning authorized by this document**. Date: 2026-10-11 JST. Parent: [Issue #103](https://github.com/ryonakayama234/Aster/issues/103). Baseline: main `3b19dca7b1f8111f9d587a42ce952c927cb52ca9`.

## Exactly one research question

> When the action set, task, and environment state are held constant, does the existing frozen Aster DecisionModel change its **post-observation** action between a successful memory read and a deterministic, recoverable read failure, and can that choice lead to separately verified completion?

This tests **within-episode observation conditioning**. It does **not** test weight update, learned correction transfer, arbitrary fault recovery, planning from scratch, or fluent Japanese.

The code already serializes `{"action", "observation"}` history into each `aster-decision-input-0` candidate. The current `CalculateAndStoreCandidates` returns `stop("tool_failure")` alone after any failed tool; the model cannot choose to retry. Original builder, ARTIFACT, serializer, frozen runs and ACT recipe must remain unchanged.

## Exact first development fixture (one task family)

- Task: `{"kind":"calculate_and_store","operation":"add","left":23,"right":19,"store_as":"answer"}`.
- Initial runtime memory: `{"answer":42}`. This is a **fixture**, not a value calculated or stored by the model. The experiment is verification/retry, not calculation.
- First executed action (**scripted prefix, not attributed to the model**): `memory.get(key="answer")`.
- Branch **S** (success): real `memory.get` returns `Observation(accepted=True, ok=True, output={"key":"answer","value":42})`. The existing `TaskEvaluator` may verify the goal at that step. Desired *next* model action: `Action.stop("policy_stop")`.
- Branch **F** (one-shot transient failure): a test-only wrapper for the same registered `memory.get` handler, on **its first call only**, raises `ToolExecutionError("transient_read_error", "temporary read failure")` **before any environment mutation**. The next invocation delegates to unchanged `memory_get`. Initial and post-prefix memory stay `{"answer":42}` in both branches. Desired *next* model action: retry `Action.tool("memory.get", key="answer")`; after the successful retry, select `stop("policy_stop")`.
- The non-mutating wrapper and fault schedule must be recorded in evidence, never installed in normal service or silently activated for production. No external systems, private corpus or persistent tool writes.

## Decision isolation (non-negotiable)

After the prefix, both arms offer **exactly the same ordered candidate tuple**, including arguments and stop reason:

1. `Action.tool("memory.get", key="answer")`
2. `Action.stop("policy_stop")`

Do not branch the candidate set, candidate order, candidate arguments, task, current memory, or step according to whether the read succeeded; do not insert `goal_verified`-specific stop candidates, oracle labels, fallback actions or state annotations. At the paired first decision, the only decision-visible difference must be the last **Observation** in the trajectory. `Transition.evaluation` must remain correct in each arm but is not serialized to model input by the existing `serialize_decision_input`; assert this property in a regression test. The later history naturally diverges as a consequence of the agent's own actions; do not call it a same-state counterfactual after that point.

The observer-independent policy is a fixed candidate choice across S and F, so it can satisfy **at most one of two desired post-prefix decisions** in this development pair. This is a deterministic sanity bound for this constructed pair, not an estimate of chance-level performance on arbitrary tasks.

Because the preceding step is scripted, label model results **model-only continuation after scripted prefix**. They are **not** fully autonomous initial-action agent success. Model-selected Action must exactly equal executed Transition.action for every non-scripted step.

## Versioned compatibility and preflight gates

1. Before model inference, identify and record locally the immutable DecisionModel artifact ID, tokenizer/model config digests, `candidate_builder_id`, source Git commit, expected `context_length`, and the exact test-only builder/runner version. The artifact's original `calculate-and-store-v0` lineage may not be changed or falsely relabeled. Injecting the new two-candidate builder is an **explicit out-of-distribution experimental adapter**, not a production-compatible ACT recipe.
2. Preserve the `aster-decision-input-0` serializer. Encode **each exact candidate** under both completed-prefix histories (and the second choice after the retry). Verify `max(encoded_length) <= model.backbone.config.context_length` before any model call. The LANG model's context 128 is *not automatically* the DecisionModel's context; load the actual artifact config.
3. On overflow, classify **BLOCKED_CONTEXT**; stop without reporting model failure/success. Do not silently truncate the history, change context/model/tokenizer, or introduce a compact serializer in this issue. Create a separately reviewed versioned input protocol if needed.
4. The existing `run_loop` starts from `step=0`. A scripted prefix must be captured through a new **test-only orchestrator** with correct consecutive `Transition.step`, replayable `Trajectory`, and correct cumulative `EvaluationResult`. Do not stuff synthetic history into `run_loop` while resetting step numbering.
5. Stable `ToolExecutor`, `RuntimeState`, `TaskEvaluator`, `TrajectoryRecorder` and `evaluate_episode` are reused. Reject impossible success/failure records and any mismatch among environment facts, selected/executed actions or the saved trajectory.
6. Before comparing models, prove the counterfactual prefix has the same task, memory and step; prove candidate equality and observation-only difference in the serialized input. Check both branches under fixed reference policy; reference labels must not enter the model input. Never read old sealed test.

## Controls, budget and stopping

- **Reference/oracle policy:** selects stop on S; get then stop on F. Expected: two goal-verified successful terminals. Its success is *not* credited to the learned model.
- **Observation-blind policy:** same state and candidates; mask last observation or use a pre-fixed single action in both arms. Report both arms, not the favorable one. The observational intervention must be documented so it doesn't accidentally alter unrelated information.
- **Frozen ModelPolicy:** no training, gradient steps, calibration, best-seed search, rule fallback or selection among checkpoints. Score identical candidates in each arm, record scores/probabilities/selected index and all subsequent real-tool transitions. The artifact may not have been trained on recovery decisions; failure is an expected and informative outcome.
- A **single pair** of S/F is a **development diagnostic only**. Ceiling: one fault-free reference pair, one blind pair, one frozen-model pair, each at most **three** executed tool/stop actions after including the prefix (S: 2, F: 3), plus minimal pure unit fixtures. No repeated trial until a favorable result; no training runs, GPU, CI paid resources or external services.
- Fault case: prefix read failure → retry success → stop, **3 total actions**. Success case: prefix read success → stop, **2 total actions**. If step budget is exhausted, it is failure. Preserve explicit environment-error vs model-choice labels.
- Any future independent task-family replication requires a **separate preregistered design** with meaningful independent error/causal structures; different keys/integers/seeds of this one failure rule are **not** independent families. No confirmatory p-value or broad generalization statement follows from this pair.

## Three gates and exact interpretation

| Gate | Pass condition | What it proves |
| --- | --- | --- |
| G0 Structural | Both branches have valid real prefix transitions, identical state/candidates, different truthful Observations, immutable frozen artifact, correct verifier, replayable trace; measured tokenized inputs all fit | Evaluation/wiring feasibility; **not capability** |
| G1 Reference | Reference success **2/2**; blind fixed policy's initial post-prefix choice correct **at most 1/2**, with no model involvement | Task discriminates observation conditioning; **not learned skill** |
| G2 Frozen-model diagnostic | The model chooses **stop on S** and **retry on F**, then **stop after successful retry**; both persisted trajectories have `goal_verified=true`, `task_success=true`, and 0 unplanned rejection/fallback/model updates | Supported **only for this scripted-prefix development pair**; never general capability |

If G0/G1 fail, classify **BLOCKED_INFRASTRUCTURE** and do not run G2. If G0/G1 pass but G2 fails, classify **NOT_SUPPORTED_ON_DEV_PAIR**, record details and do not silently add training. If G2 passes, classify **DEV_PAIR_SUCCESS_ONLY**.

Failure taxonomy: absent recovery candidate, action set leakage, incorrect observation serialization, invalid counterfactual state, context overflow, reference policy/evaluator mismatch, model misranking, inappropriate retry/stop, executor rejection vs execution error, ineffective retry, unverified stop, trace mismatch, and action budget exhaustion.

## Subsequent decision, not part of v0a

If G0 is blocked by input length, version a compact-observation serializer and separately verify information sufficiency before trying again. If G0/G1 work but model fails G2, only then consider a **new separately frozen supervised DecisionExample** trial with matched baselines and independent families; this is not permission to retrain in the current experiment. If G2 works, design independent fault-family replication (and eventually missing-key reconstruction vs irrecoverable divide-by-zero), **without** recycling this observed dev pair into a holdout. No changes to Drive Corpus / LANG v0 tokenizer / RL / KLPO.

## Required evidence to persist when a later implementation/run is approved

- `protocol_id`, Git SHA, immutable artifact manifest and readback, test-fixture version, exact candidate tuple, serialized-input hashes and token lengths (no private text), branch/fault schedule.
- Step-indexed `state_before`, model candidate score trace, selected/executed Action, real `Observation`, `state_after`, independent `EvaluationResult`, terminal and `goal_verified`.
- Protocol gate statuses and explicit separation of scripted prefix, rule/reference/observation-blind controls, and frozen-model continuation.
- A deterministic source/evidence audit asserting zero updates to weights, corpus, tokenizer and old Runs.

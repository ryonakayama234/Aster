# OBS v0c — Truthful Missing-Key Reconstruction (development Trial)

Status: **implementation on review branch; G0/G1 fixtures and optional local G2 runner; real frozen-model evidence NOT YET MEASURED**. Parent [#106](https://github.com/ryonakayama234/Aster/issues/106), preceding OBS v0a #104 / v0b #105. Baseline main `5fc236d20e9416d324b72cc8095670c4d96c0d8f`.

## One falsifiable question

After the unchanged real `memory.get("answer")` fails with `memory_key_not_found` on an absent key, can the exact *previously pinned* frozen DecisionModel select a productive **four-action model-only continuation**:

```text
Task: calculate_and_store(add, 23, 19, store_as="answer")
Initial RuntimeContext.memory = {}
                         |
       SCRIPTED memory.get("answer")
                         |
  ToolExecutor -> memory_key_not_found, memory stays {}
                         |
           [MODEL decisions START]
              calculator(add, 23, 19)
                         |
               real Observation: 42
                         |
               memory.put("answer", 42)
                         |
             real memory write, then
                         |
                memory.get("answer")
                         |
               real Observation: 42
                         |
                 stop("policy_stop")
                         |
           independent TaskEvaluator
        goal_verified AND task_success
```

No model action before the missing-key prefix is counted. This is a **different state/fault mechanism** from the same-state OBS v0a success/transient pair. It does not isolate the semantic effect of one observation field: the model sees the task, evolving history and evolving memory state.

## Frozen protocol and candidates

- `protocol_id=obs-missing-key-reconstruction-v0c`; `experiment_mode=trial`; `evidence_scope=development`.
- Test-only builder `obs-missing-key-reconstruction-v0c-experimental` is injected explicitly. The production `calculate-and-store-v0` builder is not modified; its original fail-stop behavior remains.
- At every decision, the ordered options are: `calculator`; *only once a real successful calculator Observation exists*, `memory.put` whose value is copied from that Observation; `memory.get`; `stop("policy_stop")`. A wrong action is still available and can be selected. No observation-conditioned removal of erroneous choices, goal-specific stop labels, reference-target hints, Evaluator labels, or precomputed oracle values are fed to the model.
- Exact first choices are `[calculator, memory.get, stop]`. After successful calculator execution they are `[calculator, memory.put(observed_value), memory.get, stop]`. `memory.put` may not exist before observing a calculator result: that is an unavoidable *causal/tool-argument availability constraint*, not evidence that the DecisionModel knows how to construct the value.
- Only `memory.get`, `calculator`, `memory.put`, `stop` may execute. The reference policy may use tool outcomes but is never permitted to influence frozen-model actions. The blind baseline always stops immediately.
- Four **model-controlled actions maximum**; one separate scripted prefix; therefore five total `Transition` objects at most. An early stop or budget exhaustion without verified terminal success is a failure, not an implicit success. No retries or post-hoc checkpoint selection.
- Artifact identity: `decision_model:0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca`. Reload strictly verifies the existing manifest/model/tokenizer SHA-256 and builder lineage. The historical training `source_git_sha` may be null: preserve this uncertainty. No trained weights or tokenizer artifacts are changed.

## Architectural contract

```text
Task / RuntimeState / Trajectory
                 |
    MissingKeyCandidates (experimental only)
                 | concrete ordered Actions
                 v
      Frozen ModelPolicy / DecisionModel
           | scores -> selected Action -> DecisionTrace
           v
     ToolExecutor -> real Observation -> RuntimeContext mutation
           |                              |
           +----------> Transition <------+
                           |
               TrajectoryRecorder
                           |
                  TaskEvaluator (independent)
                           |
         saved trajectory.jsonl -> evaluation.json
```

Evaluator recomputes expected arithmetic **only at the verifier boundary**. It is not available in the candidate builder or `serialize_decision_input`. The model's input history contains previous Action and Observation, not `EvaluationResult`, reference labels, or teacher targets. Candidate Action parameters for `memory.put` are supplied by the **successful actual calculator output**, so this Trial measures *candidate selection*, **not spontaneous argument/code generation**.

## G0/G1/G2 gate criteria

| Gate | Pass condition | Claim allowed |
| --- | --- | --- |
| G0 truthful structure | Empty initial memory; genuine `memory_key_not_found` via unchanged ToolExecutor; unmodified memory after prefix; stable Action/Transition/Evaluator invariants; exact serializer omits evaluation; correct versioned candidate builder | Plumbing/contract only |
| G0 context + artifact | Tokenize *every candidate at every reachable nonterminal decision node* under the 4-action budget, **including possible wrong actions**, with the actual artifact tokenizer; worst tokenized candidate fits its actual model context length; identity and immutable provenance verified | Safe-to-score within this bounded fixture, not competence |
| G1 reference & negative control | Real calculator -> put -> get -> stop, 5 transitions including scripted failure, `goal_verified=true`, `task_success=true`, exactly one accepted execution failure from the scripted prefix; blind stop fails | Valid experiment fixture, not learned competence |
| G2 frozen-model diagnostic | Model chooses precisely the four continuation actions, records one DecisionTrace per selected Action matching executed transition, tool read-back verifies state, terminal task success and unchanged weights | **DEV_CASE_SUCCESS_ONLY**, one already-designed development case |
| G3 independent generalization | Distinct future fault/task families with prospectively frozen Batch protocol/holdout | **Not included or claimed** |

If context fails, mark `BLOCKED_CONTEXT` **before the first model call**. If wiring/reference fails, do not call the model. Otherwise a wrong first choice, wrong continuation, timeout/budget exhaustion or unverified stop is `NOT_SUPPORTED_ON_DEV_CASE`. Preserve negative results. A synthetic fake ModelPolicy fixture that passes is **not** a frozen-model run.

The frozen prefix means the recovered trajectory is **model-only continuation**, never a fully autonomous initial plan. Success also does not isolate the observation semantic mechanism: the task instruction alone may cue calculate-and-store.

## Evidence and budget

The optional G2 runner writes a new append-only `runs/<id>/` containing `trajectory.jsonl` (canonical evidence), `evaluation.json` derived from a read-back with a SHA-256 link, `decision-traces.jsonl` with all candidates/scores/probabilities, `obs-v0c-summary.json` with exact model input SHA-256 per candidate and action, and `run.json` with protocol, source Git revision, model/tokenizer hashes, source builder, experimental builder, and no-update/no-fallback metadata. Do not upload private artifact weights or local runs. The small fixture is CPU-only, without training, SFT/RL/KLPO, service changes, extra CI runners, or sealed test access.

The 4-action decision budget yields 3 candidates on the first decision and up to 4 on the three intended subsequent decisions. A *hypothetical uniform independent random selector*, if it sees exactly those candidate counts on the correct path, would pick the entire correct path with probability `1/(3*4^3) = 1/192 ≈ 0.00521` (computed with Wolfram); this is **not** a calibrated chance-level benchmark for the trained model, since candidate scores, action history and model biases are nonuniform. One Trial gives no population generalization estimate.

## How to run (after reviewed source is available locally)

```bash
# Safe structure/reference checks: G0/G1 only, no private model loaded.
.venv/bin/python scripts/run_obs_v0c.py --root .

# Explicit one-time user-side frozen-model development diagnostic (G2).
.venv/bin/python scripts/run_obs_v0c.py --root . \
  --artifact artifacts/decision_models/0056272101eb7fd5bb34e6fef7a07a52e202a9bac9b0bc3db65361332feefeca
```

Run `pytest tests/runtime/test_observation_reconstruction.py` and the repo's usual `pyright` check before any G2 handoff. If the pinned artifact is not locally present, G2 remains **NOT_RUN** and no alternative checkpoint is admissible.

## Research outcome decision

Record: **observed / not observed / blocked** on this development Trial, plus the *first erroneous decision* (repeat missing get, premature stop, duplicate calculator, premature get, wrong put, context block, etc.), then freeze exactly one next experiment. Do not promote a worked development case to unseen confirmatory evidence; do not silently train the same model after seeing G2. Follows `docs/ExperimentCycle-v0.md` and `docs/EvaluatorContract-v0.md`.

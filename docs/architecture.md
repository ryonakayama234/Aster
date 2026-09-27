                 ┌───────────────┐
                 │   RAW DATA    │
                 └───────┬───────┘
                         │
                         ▼
                    corpus/
                         │
                         ▼
                data/canonical/
                         │
           ┌─────────────┴────────────┐
           │                          │
           ▼                          ▼
      tokenizer/                  training/
           │                          │
           ▼                          ▼
       token IDs  ───────────────→  model/
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                      LM Head               Decision Head
                         │                         │
                     language                structured
                    generation               decision
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                                  runtime/
                                      │
                                      ▼
                                    tools/
                                      │
                                      ▼
                                 ENVIRONMENT
                                      │
                                      ▼
                                  records/
                                      │
                      State → Action → Observation
                                      │
                                      ▼
                                 evaluator/
                                      │
                              Evaluation vector
                                      │
                                      ▼
                                   reward/
                                      │
                         RewardSpec → RewardResult
                                      │
                                      ▼
                                   credit/
                                      │
                       assigned + unassigned credit
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                      runs/              counterfactual analysis
                                                   │
                                      forced candidate branches
                                                   │
                                        reward / advantage estimate
                                                   │
                         └────────────┬────────────┘
                                      ▼
                         training data生成
                                      │
                                      └────────→ model v+1

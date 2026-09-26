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
                     success / score / verifier
                                      │
                                      ▼
                                  runs/
                                      │
                                      ▼
                         training data生成
                                      │
                                      └────────→ model v+1
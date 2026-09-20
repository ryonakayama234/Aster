# 最終イメージ仮案

Shared representation
        │
        ├── LM Head
        │
        └── Decision Head

## LM Head
context
↓
50,000 vocab logits
↓
next token

## Decision Head
state + candidates
↓
candidate logits
↓
action probability
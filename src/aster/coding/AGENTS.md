# coding専用environment
*Code Generation→構文的に許されたActionの連続選択*
へ変換する。

## 学習対象仮案
before.py
    ↓
AST before

「変数名を変更」
    ↓

edit action
    ↓

after.py
    ↓
AST after

pytest
    ↓
成功 / 失敗

P(edit action∣code state,goal)
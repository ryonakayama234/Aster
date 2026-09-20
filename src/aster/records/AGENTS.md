# Asterが経験を記録する

## 中心schema仮案
*Transition*
### 概念
Transition(
    state=...,
    available_actions=...,
    chosen_action=...,
    observation=...,
    evaluation=...
)

Trajectory = list[Transition]

1.Computer Use
screen
→ click
→ new screen

2.Coding Agent
AST
→ edit
→ new AST

3.Shell Agent
filesystem state
→ command
→ stdout/stderr

などであっても、**全部同じTransitionへ落とす**。

## 「Asterを使うこと自体」をデータ収集にする
将来Aster Runtimeが動く、その各段階にRecorderを噛ませる

## Computer Use仮案
偽物のUI環境で
screen = {
    "elements": [
        {"id": 1, "type": "button", "text": "Open"},
        {"id": 2, "type": "button", "text": "Delete"},
        {"id": 3, "type": "textbox", "text": ""},
    ]
}

Action Space：
actions = [
    ("click", 1),
    ("click", 2),
    ("type", 3),
]

Goal：
"Openを押してください"

正解：
("click", 1)

*state+goal→action*
datasetをつくる。

### その後、

Fake UI
↓
HTML DOM
↓
実Browser
↓
Screenshot + DOM
↓
Computer Use

とする。
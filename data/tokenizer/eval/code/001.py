from dataclasses import dataclass


@dataclass
class Task:
    name: str
    priority: int
    done: bool = False


def pending_tasks(tasks: list[Task], minimum_priority: int = 1) -> list[Task]:
    return [
        task
        for task in tasks
        if not task.done and task.priority >= minimum_priority
    ]


def summarize(tasks: list[Task]) -> str:
    pending = pending_tasks(tasks)
    names = ", ".join(task.name for task in pending)
    return f"pending={len(pending)} [{names}]"


sample = [
    Task("write tests", 3),
    Task("update docs", 1, done=True),
    Task("check tokenizer", 2),
]

print(summarize(sample))

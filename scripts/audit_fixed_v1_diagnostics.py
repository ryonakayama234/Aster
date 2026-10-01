"""Recompute frozen diagnosis counts and stop-time truth from saved evidence only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_prediction(row):
    if row["invalid"]:
        return
    values, actions = row["scores"], row["candidates"]
    selected = actions[max(range(len(values)),key=values.__getitem__)]
    assert selected == row["selected"]
    assert row["correct"] == (selected == row["target"])
    target_index = actions.index(row["target"])
    margin = (values[target_index]-max(v for i,v in enumerate(values) if i != target_index)
              if len(values)>1 else None)
    assert margin == row["margin"]
    for text, ids in zip(row["inputs"],row["token_ids"],strict=True):
        # Source artifacts use UTF-8 byte IDs 0..255, BOS=256, EOS=257.
        assert ids == [256,*text.encode("utf-8"),257]


def independent_success(transitions):
    last=transitions[-1]
    task=last["state_after"]["task"]
    expected=(task["left"]+task["right"] if task["operation"]=="add" else task["left"]-task["right"])
    key=task["store_as"]
    memory=last["state_after"]["memory"]
    confirmed=False
    for t in reversed(transitions):
        a,o=t["action"],t["observation"]
        if not o["ok"] or a["arguments"].get("key")!=key:
            continue
        if a["name"]=="memory.put":
            break
        if a["name"]=="memory.get":
            output=o["output"]
            confirmed=(isinstance(output,dict) and output.get("key")==key
                       and not isinstance(output.get("value"),bool) and output.get("value")==expected)
            break
    return (last["action"]["kind"]=="stop" and key in memory
            and not isinstance(memory[key],bool) and memory[key]==expected and confirmed)


def audit(root):
    study=json.loads((root/"study.json").read_text())
    counts=Counter();details=[]
    for model in study["models"]:
        output=root/f"seed-{model['seed']}-{model['arm']}"
        predictions=rows(output/"predictions.jsonl")
        episodes=rows(output/"episodes.jsonl")
        pairs=rows(output/"pairs.jsonl")
        assert len(predictions)==len(episodes)==128
        for row in predictions:
            check_prediction(row);counts["prefix_predictions"]+=1
        by_id={r["case_id"]:r for r in predictions}
        for pair in pairs:
            assert pair["both_correct"]==(by_id[pair["left"]]["correct"] and by_id[pair["right"]]["correct"])
            counts["pairs"]+=1
        for episode in episodes:
            trajectory=episode["trajectory"]
            for i,t in enumerate(trajectory):
                assert t["step"]==i and t["schema_version"]=="aster-transition-1"
                assert t["evaluation"]["goal_contract"]=="calculate-and-store-current-v1"
                if i:
                    assert trajectory[i-1]["state_after"]==t["state_before"]
                counts["transitions"]+=1
            for row,t in zip(episode["decisions"],trajectory[episode["prefix_length"]:],strict=True):
                check_prediction(row)
                assert row["selected"]==t["action"]
                counts["visited_predictions"]+=1
            first=next((r for r in episode["decisions"] if not r["correct"]),None)
            assert first==episode["first_error"]
            expected=independent_success(trajectory)
            assert expected==episode["summary"]["task_success"]
            assert episode["success"]==(None if episode["invalid"] else expected)
            counts["episodes"]+=1
        for group in ("normal","factorial","completion"):
            selected=[r for r in predictions if r["group"]==group]
            eps=[e for e in episodes if e["group"]==group]
            assert model[group]==dict(count=len(selected),correct=sum(r["correct"] for r in selected),
                                      invalid=sum(r["invalid"] is not None for r in selected),
                                      prefix_success=sum(e["success"] is True for e in eps),
                                      prefix_invalid=sum(e["invalid"] is not None for e in eps))
        details.append(dict(seed=model["seed"],arm=model["arm"],predictions=128,episodes=128))
    evidence={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='cached-audit.json'}
    result=dict(status="passed",counts=dict(counts),models=details,evidence_sha256=evidence)
    (root/"cached-audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return dict(counts)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    print(audit(parser.parse_args().run))

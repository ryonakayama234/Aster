"""Observable relation extraction; no teacher, evaluator, arithmetic or stage labels."""
import json

from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.runtime.state import RuntimeState

RELATION_SERIALIZER_ID = "aster-decision-relations-input-0"


def observable_relations(state: RuntimeState, history: Trajectory) -> dict:
    """Scan successful raw observations independently of the teacher's helpers.

    None means unknown/unobserved, not a negative observation. The extractor
    externalizes equality and temporal comparison; it does not learn them.
    """
    key = state.task.get("store_as")
    arguments = {k: state.task.get(k) for k in ("operation", "left", "right")}
    observed = False
    calculation = None
    last_put = -1
    last_get = -1
    get_output = None
    for index, transition in enumerate(history.transitions):
        action, obs = transition.action, transition.observation
        if not obs.ok:
            continue
        if action.name == "calculator" and action.arguments == arguments:
            observed, calculation = True, obs.output
        if action.arguments.get("key") != key:
            continue
        if action.name == "memory.put":
            last_put = index
        elif action.name == "memory.get":
            last_get, get_output = index, obs.output
    present = isinstance(key, str) and key in state.memory
    value = state.memory.get(key) if isinstance(key, str) else None
    fresh = last_get > last_put if last_get >= 0 else None
    get_matches = None
    if observed and last_get >= 0:
        get_matches = (isinstance(get_output, dict) and get_output.get("key") == key
                       and not isinstance(get_output.get("value"), bool)
                       and get_output.get("value") == calculation)
    return {
        "calculation_observed": observed,
        "calculation_value": calculation,
        "memory_present": present,
        "memory_value": value,
        "memory_matches_calculation":
            (present and not isinstance(value, bool) and value == calculation) if observed else None,
        "get_observed": last_get >= 0,
        "get_after_last_target_put": fresh,
        "get_matches_calculation": get_matches,
        "last_tool_failed": bool(history.last and not history.last.observation.ok),
    }


def relation_input(state: RuntimeState, history: Trajectory, candidate: Action) -> str:
    """Task/candidate retain real values; raw event list and event counts are removed."""
    return json.dumps({"schema": RELATION_SERIALIZER_ID, "task": state.task,
                       "relations": observable_relations(state, history),
                       "candidate": candidate.to_dict()}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))

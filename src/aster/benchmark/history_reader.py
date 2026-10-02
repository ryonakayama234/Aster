"""Executed history-to-CMF registration, with no torch or learned model dependency."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from aster.agent.policy import RuleBasedPolicy
from aster.benchmark.state_representation import canonical, digest, executor
from aster.evaluator.episode import evaluate_episode
from aster.evaluator.goal_contract import CURRENT_GOAL_V1
from aster.evaluator.verifier import TaskEvaluator
from aster.records.decision_relations import observable_relations
from aster.records.recorder import TrajectoryRecorder
from aster.records.runlog import RunLog
from aster.records.transition import Action, Observation, Transition
from aster.runtime.context import RuntimeContext

SCHEMA = 'aster-history-reader-preflight-0'
INPUT_ID = 'aster-history-reader-input-0'
CONTEXT = 2048
FAMILIES = ('normal0', 'normal1', 'normal2', 'normal3', 'wrong_overwrite',
            'same_write', 'unrelated_write', 'repair_unconfirmed',
            'other_calculation', 'wrong_before_calculation')
TASKS = {
    'train': (('add', 2, 3, 'total'), ('add', 4, 7, 'sum_slot'),
              ('add', -2, 8, 'sum_neg'), ('add', 0, 6, 'sum_zero'),
              ('subtract', 9, 4, 'result'), ('subtract', 12, 3, 'difference_slot'),
              ('subtract', -3, 4, 'diff_neg'), ('subtract', 6, 0, 'diff_zero')),
    'dev': (('add', 31, -8, 'reader_dev_a'), ('add', -11, 0, 'reader_dev_b'),
            ('subtract', 43, 17, 'reader_dev_c'), ('subtract', -13, -5, 'reader_dev_d')),
}


class ExecutedHistory:
    def __init__(self, task):
        self.context = RuntimeContext(task)
        self.recorder = TrajectoryRecorder()
        self.tools = executor()
        self.available = self.tools.registry.names() + ('stop',)
        self.teacher = RuleBasedPolicy(goal_contract=CURRENT_GOAL_V1)
        self.evaluator = TaskEvaluator(goal_contract=CURRENT_GOAL_V1)

    def execute(self, action):
        history = self.recorder.trajectory()
        before = self.context.snapshot(len(history))
        obs = (Observation(True, True, {'stopped': True, 'reason': action.arguments.get('reason')})
               if action.kind == 'stop' else self.tools.execute(action, self.context))
        if not obs.ok:
            raise ValueError('failed_tool_outside_reader_scope')
        after = self.context.snapshot(len(history) + 1)
        transition = Transition(len(history), before.to_dict(), self.available, action, obs,
                               after.to_dict(), self.evaluator.evaluate(self.context.task, after, action, obs, history))
        self.recorder.record(transition)
        return transition

    def next_action(self):
        history = self.recorder.trajectory()
        return self.teacher.decide(self.context.snapshot(len(history)), history, self.available)

    def input_text(self):
        history = self.recorder.trajectory()
        return canonical({'schema': INPUT_ID, 'task': self.context.task,
                          'memory': self.context.memory,
                          'events': [{'action': t.action.to_dict(),
                                      'observation': {'ok': t.observation.ok, 'output': t.observation.output}}
                                     for t in history.transitions]})


def read_facts(text):
    """Nonlearned JSON oracle; never presented as reader-model accuracy."""
    data = json.loads(text)
    if set(data) != {'schema', 'task', 'memory', 'events'} or data['schema'] != INPUT_ID:
        raise ValueError('Reader input schema mismatch')
    task, events = data['task'], data['events']
    observed, value, last_put, last_get, retrieved = False, None, -1, -1, None
    for i, event in enumerate(events):
        action, obs = event['action'], event['observation']
        if not obs['ok']:
            raise ValueError('failed_tool_outside_reader_scope')
        if action['name'] == 'calculator' and action['arguments'] == {
                k: task[k] for k in ('operation', 'left', 'right')}:
            observed, value = True, obs['output']
        if action['arguments'].get('key') == task['store_as']:
            if action['name'] == 'memory.put':
                last_put = i
            elif action['name'] == 'memory.get':
                last_get, retrieved = i, obs['output']
    memory = data['memory']
    m = (task['store_as'] in memory and not isinstance(memory[task['store_as']], bool)
         and memory[task['store_as']] == value) if observed else None
    f = (last_get > last_put and isinstance(retrieved, dict)
         and retrieved.get('key') == task['store_as']
         and not isinstance(retrieved.get('value'), bool)
         and retrieved.get('value') == value) if observed and last_get >= 0 else None
    return {'C': observed, 'M': m, 'F': f}


def feature_vector(facts):
    return [int(facts[k] is True) for k in ('C', 'M', 'F')]


def expected_type(facts):
    c, m, f = feature_vector(facts)
    return 'calculator' if not c else 'memory.put' if not m else 'memory.get' if not f else 'stop'


def make_prefix(task, family):
    session = ExecutedHistory(task)
    if family.startswith('normal'):
        for _ in range(int(family[-1])):
            session.execute(session.next_action())
        return session
    key = task['store_as']
    if family == 'wrong_before_calculation':
        session.execute(Action.tool('memory.put', key=key, value=-999))
        return session
    if family == 'other_calculation':
        other = 'subtract' if task['operation'] == 'add' else 'add'
        session.execute(Action.tool('calculator', operation=other, left=task['left'], right=task['right']))
        return session
    for _ in range(3):
        session.execute(session.next_action())
    value = session.context.memory[key]
    if family == 'unrelated_write':
        session.execute(Action.tool('memory.put', key='reader_noise', value=101))
    elif family == 'same_write':
        session.execute(Action.tool('memory.put', key=key, value=value))
    elif family in ('wrong_overwrite', 'repair_unconfirmed'):
        session.execute(Action.tool('memory.put', key=key, value=-999))
        if family == 'repair_unconfirmed':
            session.execute(Action.tool('memory.put', key=key, value=value))
    else:
        raise ValueError('Unknown prefix family')
    return session


def replay(row):
    session = ExecutedHistory(row['task'])
    for saved in row['trajectory']:
        if session.execute(Action.from_dict(saved['action'])).to_dict() != saved:
            raise ValueError('Reader prefix replay mismatch')
    if session.input_text() != row['input']:
        raise ValueError('Reader input persistence mismatch')
    return session


def independent_completion(session):
    for _ in range(8):
        if session.execute(session.next_action()).action.kind == 'stop':
            break
    task = session.context.task
    wanted = task['left'] + task['right'] if task['operation'] == 'add' else task['left'] - task['right']
    key = task['store_as']
    put, get = -1, -1
    for i, t in enumerate(session.recorder.trajectory().transitions):
        if t.action.arguments.get('key') != key or not t.observation.ok:
            continue
        if t.action.name == 'memory.put':
            put = i
        elif t.action.name == 'memory.get' and t.observation.output == {'key': key, 'value': wanted}:
            get = i
    if (session.context.memory.get(key) != wanted or not 0 <= put < get
            or not session.recorder.trajectory().stopped
            or not evaluate_episode(session.recorder.trajectory()).task_success):
        raise ValueError('Independent completion check failed')


def build_cases():
    rows = []
    for split, specs in TASKS.items():
        for i, (operation, left, right, key) in enumerate(specs):
            task = dict(kind='calculate_and_store', operation=operation, left=left, right=right, store_as=key)
            for family in FAMILIES:
                s = make_prefix(task, family)
                rows.append(dict(case_id=f'{split}/task{i}/{family}', split=split,
                                 leakage_group=f'{split}/task{i}', family=family, task=task,
                                 input=s.input_text(), facts=read_facts(s.input_text()),
                                 trajectory=[t.to_dict() for t in s.recorder.trajectory().transitions]))
    return rows


def audit(rows):
    if Counter(r['split'] for r in rows) != Counter(train=80, dev=40):
        raise ValueError('Preregistered split counts mismatch')
    groups, labels, signatures, manifest, lengths = defaultdict(set), defaultdict(set), defaultdict(set), [], []
    task_splits, group_tasks, group_families = defaultdict(set), defaultdict(set), defaultdict(list)
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate reader case ID')
    for row in rows:
        s = replay(row)
        text, facts = row['input'], read_facts(row['input'])
        if facts != row['facts']:
            raise ValueError('Reader label mismatch')
        history = s.recorder.trajectory()
        existing = observable_relations(s.context.snapshot(len(history)), history)
        ref = {'C': existing['calculation_observed'], 'M': existing['memory_matches_calculation'],
               'F': (existing['get_after_last_target_put'] is True and existing['get_matches_calculation'] is True)
               if existing['calculation_observed'] and existing['get_observed'] else None}
        if facts != ref:
            raise ValueError('Independent oracle/extractor mismatch')
        target = s.next_action()
        if expected_type(facts) != (target.name if target.kind == 'tool' else 'stop'):
            raise ValueError('Fact/teacher type mismatch')
        independent_completion(s)
        raw = text.encode('utf-8')
        if bytes(list(raw)).decode('utf-8') != text or len(raw) + 2 > CONTEXT:
            raise ValueError('Reader byte context/roundtrip failure')
        sig = digest(text)
        lengths.append(len(raw) + 2)
        groups[row['leakage_group']].add(row['split'])
        task_splits[canonical(row['task'])].add(row['split'])
        group_tasks[row['leakage_group']].add(canonical(row['task']))
        group_families[row['leakage_group']].append(row['family'])
        labels[sig].add(canonical(facts))
        signatures[row['split']].add(sig)
        manifest.append(dict(case_id=row['case_id'], split=row['split'], group=row['leakage_group'],
                             family=row['family'], input_sha256=sig, facts=facts, byte_tokens=len(raw)+2))
    if any(len(v) > 1 for v in groups.values()) or any(len(v) > 1 for v in labels.values()):
        raise ValueError('Reader group leakage/input-label collision')
    if (len(groups) != 12 or any(len(v) != 1 for v in task_splits.values())
            or any(len(v) != 1 for v in group_tasks.values())
            or any(Counter(v) != Counter(FAMILIES) for v in group_families.values())):
        raise ValueError('Reader whole-task split/family coverage mismatch')
    pairs = []
    for group in sorted(groups):
        family = {r['family']: r for r in rows if r['leakage_group'] == group}
        for left, right, expected_same in (('same_write', 'unrelated_write', False),
                                           ('normal3', 'unrelated_write', True),
                                           ('same_write', 'repair_unconfirmed', True),
                                           ('wrong_overwrite', 'repair_unconfirmed', False)):
            a, b = family[left], family[right]
            if (a['facts'] == b['facts']) != expected_same:
                raise ValueError('Reader state contrast mismatch')
            pairs.append(dict(left=a['case_id'], right=b['case_id'], same_facts=expected_same,
                              same_event_count=len(a['trajectory']) == len(b['trajectory'])))
    overlap = signatures['train'] & signatures['dev']
    if overlap:
        raise ValueError('Reader cross-split input overlap')
    return dict(schema_version=SCHEMA, input_id=INPUT_ID, goal_contract=CURRENT_GOAL_V1,
                counts=dict(Counter(r['split'] for r in rows)), task_groups=len(groups),
                families=len(FAMILIES), max_byte_tokens=max(lengths), context_length=CONTEXT,
                collisions=0, cross_split_overlap=0, independent_checks=len(rows),
                split_kind='task-identity parameter transfer; shared procedural templates',
                structural_holdout=False, model_updates=0, model_scored=0,
                test_constructed=0, test_scored=0, suite_sha256=digest(rows),
                manifest_sha256=digest(manifest), manifest=manifest, pairs=pairs,
                state_contrasts=len(pairs),
                fact_counts={split: dict(Counter(canonical(r['facts']) for r in rows if r['split']==split))
                             for split in TASKS})


def run_preflight(root: Path, source):
    run = RunLog(root, 'history_reader_preflight', {'source': source}, producer='evaluator')
    try:
        rows = build_cases()
        report = audit(rows)
        with (run.path / 'cases.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
        saved = [json.loads(line) for line in (run.path / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
        if audit(saved) != report:
            raise ValueError('Saved reader audit mismatch')
        report.update(run_id=run.id, source=source, saved_audit_match=True)
        (run.path / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        run.finish('completed', report='report.json', suite_sha256=report['suite_sha256'])
    except BaseException as error:
        run.finish('failed', error=str(error))
        raise
    return run.path

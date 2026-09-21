"""学習の誤接続・情報漏れ・保存の取り違えを確認する。"""
import json
from pathlib import Path
import subprocess
import sys
import pytest

torch = pytest.importorskip('torch')
from aster.corpus.pipeline import digest, json_bytes
from aster.model.tiny_lm import TinyLM, ModelConfig
from aster.model.checkpoint import load_checkpoint
from aster.training.dataset import Window, collate, prepare_windows, load_training_tokenizer, IGNORE_INDEX
from aster.training.pretrain import TrainConfig, train, evaluate, mean_loss
from aster.training.tokenizer_run import evaluate as evaluate_tokenizer
from aster.inference.generate import generate_ids, inspect_next, describe_ids


@pytest.fixture
def inputs(tmp_path):
    files, samples = {}, []
    # Artificial structured records, exclusively for wiring tests.
    for name, split, domain, text in [('a', 'train', 'tool_json', '{"result":42}\n'),
                                       ('b', 'train', 'code', 'x = 42\n'),
                                       ('c', 'dev', 'tool_json', '{"result":17}\n'),
                                       ('d', 'test', 'tool_json', '{"result":99}\n')]:
        body = text.encode()
        path = f'{split}/{name}.txt'
        files[path] = body
        samples.append(dict(record_id=name, split=split, domain=domain, text_path=path,
                            text_sha256=digest(body), leakage_group=name))
    files['samples.jsonl'] = ''.join(json.dumps(s) + '\n' for s in samples).encode()
    manifest = json_bytes({'schema_version':'aster-training-view-0',
                           'files':{k:digest(v) for k,v in files.items()}})
    view = tmp_path / 'data/training' / digest(manifest)
    view.mkdir(parents=True)
    for path, body in files.items():
        p = view / path
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(body)
    (view/'manifest.json').write_bytes(manifest)
    tokenizer, _ = evaluate_tokenizer(tmp_path, view, 256)
    return tmp_path, view, tokenizer


def test_causality_and_right_padding():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    model = TinyLM(ModelConfig(258, 16, 32, 4, 2)).eval()
    original = torch.tensor([[256, 65, 66, 67, 68]])
    modified = torch.tensor([[256, 65, 66, 90, 91]])
    with torch.no_grad():
        first, second = model(original), model(modified)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=1e-7)
        assert not torch.equal(first[:, 3:], second[:, 3:])
        x, y = collate([Window((256,65,257), 'short','a',0),
                        Window((256,66,67,68,257),'long','b',0)],257)
        padded = model(x)[0,:2]
        alone = model(torch.tensor([[256,65]]))[0]
        torch.testing.assert_close(padded, alone, rtol=1e-5, atol=1e-6)
        assert y[0].tolist()==[65,257,IGNORE_INDEX,IGNORE_INDEX]
        before = mean_loss(model(x), y)
        changed = model(x).clone()
        changed[0,2:]=1000*torch.randn_like(changed[0,2:])
        torch.testing.assert_close(before, mean_loss(changed,y))


def test_document_windows_and_test_exclusion(inputs):
    _, view, path = inputs
    tokenizer,_=load_training_tokenizer(path,view.name)
    split=prepare_windows(view,tokenizer,3)
    assert set(split)=={'train','dev'}
    assert all(w.record_id!='d' for values in split.values() for w in values)
    windows=[w for w in split['train'] if w.record_id=='a']
    ids=tokenizer.encode('{"result":42}\n',add_bos=True,add_eos=True)
    assert [pair for w in windows for pair in zip(w.ids,w.ids[1:])]==list(zip(ids,ids[1:]))
    assert len({w.record_id for w in split['train']})==2
    with pytest.raises(ValueError,match='do not match'):
        load_training_tokenizer(path,'wrong-view')


def test_weighted_evaluation(inputs):
    _,view,path=inputs
    tokenizer,_=load_training_tokenizer(path,view.name)
    windows=prepare_windows(view,tokenizer,32)['train']
    model=TinyLM(ModelConfig(tokenizer.vocab_size,32,16,2,1))
    measured=evaluate(model,windows,tokenizer.eos_id,2)
    x,y=collate(windows,tokenizer.eos_id)
    assert measured['loss']==pytest.approx(float(mean_loss(model(x),y).detach()),rel=1e-6)
    assert measured['target_tokens']==sum(len(w.ids)-1 for w in windows)
    assert set(measured['by_domain'])=={'code','tool_json'}


def test_overfit_reload_generation_and_identity(inputs):
    root,view,path=inputs
    run=train(root,view,path,TrainConfig(context_length=32,width=32,heads=4,layers=2,
                                       batch_size=2,steps=100,eval_every=50))
    bundle=json.loads((run/'training-bundle.json').read_text())
    obs=bundle['observations']
    assert bundle['status']=='completed' and bundle['reload_exact_match']
    assert obs[-1]['train']['loss']<0.1*obs[0]['train']['loss']
    assert all(o['dev'] is None for o in obs)
    model,tokenizer,payload=load_checkpoint(run/obs[-1]['checkpoint'])
    with pytest.raises(ValueError,match='identity mismatch'):
        load_checkpoint(run/obs[-1]['checkpoint'], 'different-tokenizer')
    ids=tokenizer.encode('{"result',add_bos=True)
    continuation=generate_ids(model,tokenizer,ids,32)
    assert tokenizer.decode(ids+continuation)=='{"result":42}\n'
    assert continuation[-1]==tokenizer.eos_id
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    inspect_next(model,tokenizer,ids)
    assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
    # Separate Python process: no accidental dependence on in-memory model state.
    result=subprocess.run([sys.executable,'-m','aster.inference.generate','--checkpoint',str(run/obs[-1]['checkpoint']),
                           '--prompt','{"result','--max-new-tokens','32'],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['continuation']['token_ids']==continuation
    data=describe_ids(tokenizer,[255])
    assert not data['utf8_valid'] and data['bytes_hex']=='ff'
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    assert all(e['producer']=='trainer' for e in events)
    assert [e['seq'] for e in events]==list(range(1,len(events)+1))


def test_pilot_dev_and_failed_run(inputs):
    root,view,path=inputs
    run=train(root,view,path,TrainConfig(mode='pilot',context_length=16,width=16,heads=2,layers=1,
                                       steps=2,eval_every=1))
    bundle=json.loads((run/'training-bundle.json').read_text())
    assert all(o['dev']['target_tokens']>0 for o in bundle['observations'])
    assert bundle['observations'][-1]['tokens_seen']>0
    (view/'train/a.txt').write_text('altered')
    with pytest.raises(ValueError,match='mismatch'):
        train(root,view,path,TrainConfig(steps=1))
    failed=[json.loads(p.read_text()) for p in (root/'runs').glob('*/run.json')]
    assert any(r['status']=='failed' and r['kind']=='pretrain' for r in failed)

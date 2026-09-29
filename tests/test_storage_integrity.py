from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import subprocess
import sys
from dataclasses import replace
import pytest
from uncertain import AlternativeStore, Sampler, UnitSpec, MeasureSpec
from uncertain.store import StoreError
from uncertain.utils import canonical_json


def test_concurrent_atomic_publish(tmp_path, backend, sampling):
    a = Sampler(backend).sample((), 3, replace(sampling, n=10))
    def write(_):
        store = AlternativeStore(tmp_path)
        key = store.save(a)
        return store.load(key).content_hash
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(write, range(12)))
    assert keys == [a.content_hash]*12
    assert len(list((tmp_path/"alternatives").glob("*.json"))) == 1
    assert not list(tmp_path.rglob(".pending-*"))


def test_wrong_request_pointer_fails_even_if_content_valid(tmp_path, backend, sampling):
    store = AlternativeStore(tmp_path)
    sampler = Sampler(backend)
    first = store.get_or_sample(sampler, (), 1, sampling, UnitSpec())
    second = store.get_or_sample(sampler, (1,), 1, sampling, UnitSpec())
    stream = sampler.stream_key((), sampling, UnitSpec())
    path = tmp_path/"requests"/stream/f"{sampling.n}-1.json"
    row = json.loads(path.read_text())
    row["artifact"] = second.content_hash
    path.write_text(json.dumps(row))
    assert first.content_hash != second.content_hash
    with pytest.raises(StoreError, match="incompatible"):
        AlternativeStore(tmp_path).get_or_sample(sampler, (), 1, sampling, UnitSpec())


def test_fresh_process_replay_without_generation(tmp_path, backend, sampling):
    from uncertain import Measure, Stimulus
    spec = MeasureSpec(replace(sampling, n=20), 2, "compare_ragged")
    measure = Measure(spec, sampler=Sampler(backend), store=AlternativeStore(tmp_path))
    result = measure.compute(Stimulus((), (1,)))
    script = '''
import json,sys
from uncertain import *
backend=ToyBackend()
sampler=Sampler(backend)
def forbidden(*a,**kw): raise AssertionError("regenerated")
sampler.sample=forbidden
m=Measure(MeasureSpec.from_dict(json.loads(sys.argv[1])),sampler=sampler,store=AlternativeStore(sys.argv[2]))
r=m.compute(Stimulus((),(1,)))
print(json.dumps([r.value,r.se,r.spec_hash,list(r.artifact_ids)]))
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1]/"src")}
    actual = json.loads(subprocess.check_output([sys.executable, "-c", script,
                         canonical_json(spec), str(tmp_path)], env=env, text=True))
    assert actual == [result.value, result.se, result.spec_hash, list(result.artifact_ids)]


def test_import_does_not_load_model_libraries():
    script = "import uncertain,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1]/"src")}
    subprocess.check_call([sys.executable, "-c", script], env=env)

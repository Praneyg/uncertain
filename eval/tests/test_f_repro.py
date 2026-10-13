"""Category F: reproducibility, hashing, batching, caching, sweep output, manifest, golden."""
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest

from uncertain import (AlternativeStore, EncoderSpec, EstimatorSpec, Measure, MeasureSpec, ModelSpec,
                       Sampler, SamplingSpec, StatisticSpec, Stimulus, StimulusTransform, Sweep, UnitSpec,
                       positions)
from uncertain.distances import Discrete
from uncertain.utils import canonical

from conftest import make_measure
from toy_lm import ToyLM

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"


# --------------------------------------------------------------------------- criterion 22
@pytest.mark.req("22")
@pytest.mark.parametrize("statistic", ["mean", "energy"])
@pytest.mark.parametrize("H", [1, 3])
def test_same_seed_same_result(statistic, H):
    def run():
        lm = ToyLM([.4, .3, .2, .1])
        return make_measure(lm, n=64, seed=5, horizon=H, representation="tokens", distance="discrete",
                            statistic=statistic).compute(Stimulus((0,), (1,)))
    assert run() == run()


def _spec_variants():
    s = SamplingSpec(ModelSpec("m", "rev1"), n=8, seed=0)
    base = MeasureSpec(s, 2, "compare_ragged", distance="discrete", statistic="mean")
    sv = lambda **k: base.replace(sampling=replace(s, **k))  # noqa: E731
    return base, {
        "sampling.model.model_id": base.replace(sampling=replace(s, model=ModelSpec("m2", "rev1"))),
        "sampling.model.revision": base.replace(sampling=replace(s, model=ModelSpec("m", "rev2"))),
        "sampling.n": sv(n=9), "sampling.seed": sv(seed=1), "sampling.token_budget": sv(token_budget=64),
        "sampling.generation": sv(generation={"temperature": 0.7}),
        "horizon": base.replace(horizon=3), "short_samples": base.replace(short_samples="pad"),
        "unit": base.replace(unit="word"), "representation": base.replace(representation="strings"),
        "representation.config": base.replace(representation=EncoderSpec("tokens", {"x": 1})),
        "distance": base.replace(distance="ngram_distance"),
        "distance_config": base.replace(distance_config={"ngram": 2}),
        "statistic": base.replace(statistic="energy"),
        "statistic.config": base.replace(statistic=StatisticSpec("mean", {"block_size": 8})),
        "warping": base.replace(warping="neglog"),
        "estimator": base.replace(estimator=EstimatorSpec(method="mc")),
        "mode": base.replace(mode="entropy"), "probability": base.replace(probability="sampling"),
    }


@pytest.mark.req("22", "13")
def test_every_spec_field_changes_hash(measure):
    base, variants = _spec_variants()
    hashes = {k: v.spec_hash for k, v in variants.items()}
    measure("fields_checked", len(hashes))
    assert all(h != base.spec_hash for h in hashes.values())
    assert len(set(hashes.values())) == len(hashes)
    assert canonical(base)["unit"]["boundary"] == "token:v1"           # boundary detector is hashed
    assert MeasureSpec.from_dict(base.to_dict()) == base               # criterion 34


_PROC = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from toy_lm import ToyLM
from uncertain import (Measure, MeasureSpec, SamplingSpec, Sampler, AlternativeStore, EstimatorSpec,
                       Sweep, positions)
lm = ToyLM([.4, .3, .2, .1], pieces=("a", "b", " ", "."), eos_token_id=3)
spec = MeasureSpec(SamplingSpec(lm.model_spec, n=24, seed=11), 2, "compare_ragged",
                   distance="discrete", statistic="mean", estimator=EstimatorSpec(method="mc", bootstrap=29))
store = AlternativeStore(None if sys.argv[3] == "-" else sys.argv[3])
m = Measure(spec, sampler=Sampler(lm), store=store)
res = Sweep(m, horizon=[1, 2], statistic=["mean", "energy"]).run(positions((0, 1, 2), context=(1,)))
print(json.dumps([[r["spec_hash"], r["value"], r["se"]] for r in res.rows]))
"""


@pytest.mark.req("22", "19")
def test_cross_process_byte_identical_and_persistence_independent(scratch, measure):
    outs, stores = [], []
    for tag in ("p1", "p2", "-"):
        store = "-" if tag == "-" else str(scratch / tag)
        env = {**os.environ, "PYTHONHASHSEED": str(len(tag) * 17)}   # different hash seeds
        src = os.environ.get("EVAL_SRC", str(ROOT / "src"))
        done = subprocess.run([sys.executable, "-c", _PROC, src, str(EVAL), store],
                              capture_output=True, text=True, env=env, timeout=300)
        assert done.returncode == 0, done.stderr
        outs.append(json.loads(done.stdout))
        if store != "-":
            stores.append({p.relative_to(store).as_posix(): p.read_bytes()
                           for p in Path(store).rglob("*.json")})
    measure("artifact_files", len(stores[0]))
    assert stores[0] == stores[1]                                       # byte-identical artifacts
    assert outs[0] == outs[1] == outs[2]                                # incl. persistence off


# --------------------------------------------------------------------------- criterion 23
@pytest.mark.req("23")
@pytest.mark.parametrize("statistic", ["mean", "energy"])
def test_pairwise_block_size_does_not_change_values(statistic, measure):
    lm = ToyLM([.4, .3, .2, .1])
    vals = []
    for bs in (1, 7, 64):
        m = make_measure(lm, n=40, seed=2, horizon=2, representation="tokens", distance="discrete",
                         statistic=StatisticSpec(statistic, {"block_size": bs}))
        vals.append(m.compute(Stimulus((0,), (1,))))
    measure("values", [r.value for r in vals]); measure("ses", [r.se for r in vals])
    assert max(abs(r.value - vals[0].value) for r in vals) <= 1e-12
    assert max(abs(r.se - vals[0].se) for r in vals) <= 1e-12


@pytest.mark.req("23")
def test_study_sampler_batch_size_and_sample_identity(tiny_gpt2, measure):
    from uncertain.emnlp2024 import StudySampler
    spec = SamplingSpec(tiny_gpt2.model_spec, n=4, seed=3, token_budget=8)
    s1 = StudySampler(tiny_gpt2, batch_size=1).sample((1, 2), 4, spec)
    s1b = StudySampler(tiny_gpt2, batch_size=1).sample((1, 2), 4, spec)
    s4 = StudySampler(tiny_gpt2, batch_size=4).sample((1, 2), 4, spec)
    same = [a.tokens == b.tokens for a, b in zip(s1.sequences, s4.sequences)]
    measure("tokens_equal_across_batch_sizes", same)
    assert s1.sequences == s1b.sequences


# --------------------------------------------------------------------------- criterion 24
SENT = (0, 1, 2, 1)


def _sweep_counts(lm, H=2):
    base = make_measure(lm, n=32, horizon=H, representation="tokens", distance="discrete", statistic="mean")
    Sweep(base, horizon=[H]).run(positions(SENT, context=(3,)))
    return base.sampler.calls


@pytest.mark.req("24")
def test_sweep_samples_each_context_once(measure):
    lm = ToyLM([.4, .3, .2, .1])
    passes = _sweep_counts(lm)
    contexts = len(SENT) + 1
    measure("sampling_passes", passes); measure("distinct_contexts", contexts); measure("backend_calls", lm.calls)
    assert passes == contexts


@pytest.mark.req("24")
def test_sequential_compute_over_positions_reuses_samples(measure):
    lm_sweep = ToyLM([.4, .3, .2, .1])
    _sweep_counts(lm_sweep)
    lm = ToyLM([.4, .3, .2, .1])
    m = make_measure(lm, n=32, horizon=2, store=AlternativeStore(), representation="tokens",
                     distance="discrete", statistic="mean")
    for stim in positions(SENT, context=(3,)):
        m.compute(stim)
    measure("sequential_sampling_passes", m.sampler.calls); measure("sequential_backend_calls", lm.calls)
    measure("sweep_backend_calls", lm_sweep.calls)
    assert lm.calls <= lm_sweep.calls, (lm.calls, lm_sweep.calls)


class CountingDiscrete(Discrete):
    def __init__(self):
        self.calls = 0
        self.descriptor = {"distance": "eval-counting-discrete", "version": "1", "id": uuid4().hex}

    def __call__(self, a, b):
        self.calls += 1
        return super().__call__(a, b)


@pytest.mark.req("24")
def test_within_set_block_reused_across_comparisons(measure):
    lm = ToyLM([.4, .3, .2, .1])
    d = CountingDiscrete()
    base = make_measure(lm, n=16, horizon=2, representation="tokens", distance="discrete", statistic="energy")
    m = Measure(base.spec, sampler=base.sampler, store=base.store, distance=d)
    m.compute(Stimulus((0,), (1,)))
    first = d.calls
    m.compute(Stimulus((0,), (2,)))
    second = d.calls - first
    measure("first_calls", first); measure("second_calls", second)
    assert first == 16 * 16 * 3
    assert second == 16 * 16 * 2              # d(A,A) not recomputed


@pytest.mark.req("24")
def test_grid_sweep_costs_one_sampling_pass(measure):
    counts = {}
    for label, axes in (("one_cell", dict(horizon=[2])),
                        ("grid", dict(cells=[{"horizon": h, "statistic": st, "distance": d}
                                             for h in (1, 2) for st, d in
                                             (("mean", "discrete"), ("energy", "discrete"),
                                              ("mean", "ngram_distance"), ("min", "ngram_distance"))]))):
        lm = ToyLM([.4, .3, .2, .1])
        base = make_measure(lm, n=24, horizon=2, representation="tokens", distance="discrete", statistic="mean")
        Sweep(base, **axes).run(positions(SENT, context=(3,)))
        counts[label] = lm.calls
    measure("backend_calls", counts)
    assert counts["grid"] == counts["one_cell"]


# --------------------------------------------------------------------------- criterion 32 / 33 / 34
@pytest.mark.req("32", "33")
@pytest.mark.parametrize("with_data", [False, True])
def test_sweep_is_tidy_and_paired(with_data, measure):
    lm = ToyLM([.4, .3, .2, .1])
    base = make_measure(lm, n=16, horizon=2, representation="tokens", distance="discrete", statistic="mean")
    stims = positions(SENT, context=(3,), data=[1., 2., 3., 4.] if with_data else None)
    res = Sweep(base, horizon=[1, 2], statistic=["mean", "energy"]).run(stims)
    df = res.to_dataframe()
    measure("columns", list(df.columns))
    assert len(df) == len(stims) * 4
    for col in ("stimulus_id", "spec_hash", "value", "se", "n_a_used", "horizon", "statistic", "distance",
                "representation", "unit", "datum", "ci_low", "ci_high", "artifact_ids"):
        assert col in df.columns
    assert str(df["datum"].dtype) == "Float64"
    assert df["datum"].isna().all() != with_data
    for sid, group in df.groupby("stimulus_id"):
        for _, cells in group.groupby("horizon"):
            assert len({tuple(a) for a in cells["artifact_ids"]}) == 1   # paired: same samples


@pytest.mark.req("32")
@pytest.mark.parametrize("transform", [StimulusTransform("drop"), StimulusTransform("truncate", 1),
                                       StimulusTransform("replace", replacement=(2, 2)),
                                       StimulusTransform("shuffle", seed=1)])
def test_transforms_are_flagged(transform):
    lm = ToyLM([.4, .3, .2, .1])
    base = make_measure(lm, n=8, horizon=1, representation="tokens", distance="discrete", statistic="mean")
    stim = Stimulus((0, 1, 2), (1,), stimulus_id="s1")
    rows = Sweep(base, horizon=[1]).run([stim, transform(stim)]).rows
    assert rows[0]["transform_id"] == "identity" and rows[1]["transform_id"] != "identity"
    assert rows[1]["original_id"] == "s1"


@pytest.mark.req("34")
@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".json"])
def test_write_with_manifest(suffix, scratch, measure):
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    lm = ToyLM([.4, .3, .2, .1])
    base = make_measure(lm, n=8, horizon=2, representation="tokens", distance="discrete", statistic="mean")
    path = Sweep(base, horizon=[1, 2]).run(positions(SENT)).write(scratch / f"out{suffix}")
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    measure("manifest_keys", sorted(manifest))
    for key in ("specifications", "engine", "git_sha", "seeds", "environment"):
        assert key in manifest
    spec = next(iter(manifest["specifications"].values()))["spec"]
    assert spec["sampling"]["model"]["revision"] == lm.model_spec.revision
    assert path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- criterion 39
@pytest.mark.req("39")
def test_golden_values_pinned(measure):
    golden = ROOT / "tests" / "golden"
    gpt2 = json.loads((golden / "gpt2-values.json").read_text(encoding="utf-8"))
    measure("gpt2_golden_source", gpt2["source"])
    assert gpt2["model_id"] and gpt2["revision"] and "seed" in gpt2 and gpt2["items"]
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           str(ROOT / "tests" / "test_golden.py")], capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=600)
    measure("golden_test_tail", done.stdout.strip().splitlines()[-1:])
    assert done.returncode == 0


@pytest.mark.req("39")
def test_golden_values_are_corrected_rerun_numbers():
    gpt2 = json.loads((ROOT / "tests" / "golden" / "gpt2-values.json").read_text(encoding="utf-8"))
    # §9: "pinned to the corrected re-run numbers" -- the file says these are NOT re-run numbers.
    assert "NOT recovered study rerun numbers" not in gpt2["source"], gpt2["source"]

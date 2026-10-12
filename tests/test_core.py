from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from uncertain import (ModelSpec, UnitSpec, MeasureSpec, EncoderSpec, EstimatorSpec,
                       ConfigError, Sequence, AlternativeSet, Sampler, Measure, Stimulus,
                       Sweep, ToyBackend)
from uncertain.utils import canonical_json


def test_spec_roundtrip_and_immutable_nested(sampling):
    s = MeasureSpec(sampling, 2, "drop", distance_config={"x": [1, 2]})
    assert MeasureSpec.from_dict(s.to_dict()) == s
    assert MeasureSpec.from_dict(s.to_dict()).spec_hash == s.spec_hash
    assert hash(s) == hash(MeasureSpec.from_dict(s.to_dict()))
    with pytest.raises(TypeError):
        s.sampling.generation["top_k"] = 7  # type: ignore[index]  # the point of the test
    with pytest.raises(TypeError):
        s.distance_config["x"][0] = 9  # type: ignore[index]
    assert s.replace(horizon=3).spec_hash != s.spec_hash


def test_hash_cross_process(sampling):
    s = MeasureSpec(sampling, 1, "pad")
    script = "import json,sys; from uncertain import MeasureSpec; print(MeasureSpec.from_dict(json.loads(sys.argv[1])).spec_hash)"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1]/"src"), "PYTHONHASHSEED": "87"}
    out = subprocess.check_output([sys.executable, "-c", script, canonical_json(s)], env=env, text=True)
    assert out.strip() == s.spec_hash


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": True}, {"seed": .5}, {"token_budget": 0},
                                  {"generation": {"top_k": -1}}, {"generation": {"top_p": 0}},
                                  {"generation": {"temperature": float("nan")}},
                                  {"generation": {"num_beams": 2}},
                                  {"generation": {"repetition_penalty": 1.1}}])
def test_sampling_rejects_invalid(sampling, kwargs):
    with pytest.raises(ConfigError):
        replace(sampling, **kwargs)


def test_revision_and_unit_contract():
    with pytest.raises(ConfigError):
        ModelSpec("x", "main")
    with pytest.raises(ConfigError):
        UnitSpec("word")
    assert UnitSpec("word", "whitespace:v1").name == "word"


@pytest.mark.parametrize("args", [((0, 1), (0.,), (0.,), ("a",), ((0, 1),)),
                                 ((0,), (0.,), (0.,), ("a",), ((0, 2),)),
                                 ((0,), (.1,), (0.,), ("a",), ((0, 1),))])
def test_alignment_validation(args):
    with pytest.raises(ValueError):
        Sequence(*args)


def test_arrays_protect_alignment_and_ownership(backend, sampling):
    a = Sampler(backend).sample((), 3, replace(sampling, n=7))
    arrays = a.arrays()
    assert arrays["tokens"].shape == (7, 3)
    for i, seq in enumerate(a.sequences):
        np.testing.assert_array_equal(arrays["tokens"][i], seq.tokens)
        np.testing.assert_allclose(arrays["model_logprobs"][i], seq.model_logprobs)
    with pytest.raises(ValueError):
        arrays["tokens"].flags.writeable = True
    with pytest.raises(ValueError):
        arrays["tokens"][0, 0] = 9


def test_anchor_empty_and_pure_truncation(backend, sampling):
    from uncertain import Stimulus
    a = Sampler(backend).sample((), 3, sampling)
    original = a.content_hash
    observed = Stimulus((), (1,)).observed(backend, UnitSpec())
    b = AlternativeSet.anchor(observed, AlternativeSet.empty((1,), UnitSpec()), ())
    assert len(b) == 1 and b.deterministic and b.horizon == 1
    assert b.sequences[0].sampling_logprobs == (None,)
    assert b.sequences[0].observed == (True,)
    assert a.truncate(1).horizon == 1
    assert a.content_hash == original
    with pytest.raises(ValueError):
        a.truncate(4)
    with pytest.raises(ValueError):
        AlternativeSet.anchor(observed, AlternativeSet.empty((0,), UnitSpec()), ())


def test_nonprefix_refuses_truncation(backend, sampling):
    a = Sampler(backend).sample((), 3, sampling)
    a = replace(a, provenance={**a.provenance, "prefix_consistent": False})
    with pytest.raises(ValueError):
        a.truncate(2)


def test_invalid_pairing_before_sampling(make_measure):
    with pytest.raises(ConfigError, match="incompatible"):
        make_measure(distance="cosine")
    with pytest.raises(ConfigError, match="negative-type"):
        make_measure(statistic="energy")
    with pytest.raises(ConfigError):
        make_measure(representation=EncoderSpec("layer"), short_samples="pad")


def test_bad_estimator():
    with pytest.raises(ConfigError):
        EstimatorSpec(bootstrap=2)


def test_truncation_bound_comes_from_unterminated_mass():
    # p(EOS)=.5 per step, so the draws still open at H (no EOS yet) have mass .5**H.
    # Only they can change past H: the bound is that mass times D, and shrinks with H.
    # The old formula counted the EOS-finished draws instead: 0 at H=1, rising with H.
    backend = ToyBackend(("a", "<eos>"), {(): (.5, .5)}, eos_token_id=1)
    bounds = []
    for h in (1, 2, 4):
        m = Measure.from_config(backend, horizon=h, n=400, statistic="mean",
                                distance="discrete", estimator="mc",
                                short_samples="compare_ragged")
        d = m.compute(Stimulus(target=(1,))).diagnostics
        assert d["truncation_status"] == "bounded_by_unterminated_mass"
        assert d["truncation_bound"] == pytest.approx(1 - d["eos_fraction"])
        assert d["truncation_bound"] == pytest.approx(.5**h, abs=.08)
        bounds.append(d["truncation_bound"])
    assert bounds[0] > bounds[1] > bounds[2]
    # Anchored targets carry no unterminated count; the honest label survives.
    assert d["target"]["truncation_bound"] is None
    assert d["target"]["truncation_status"] == "no_general_bound"


def test_truncation_bound_follows_store_truncation():
    # A set sampled at H=4 and truncated to H=1 must report H=1's mass, not H=4's.
    backend = ToyBackend(("a", "<eos>"), {(): (.5, .5)}, eos_token_id=1)
    m = Measure.from_config(backend, horizon=4, n=400, statistic="mean", distance="discrete",
                            estimator="mc", short_samples="compare_ragged")
    short = m.replace(horizon=1)
    m.compute(Stimulus(target=(1,)))  # stores H=4 in the shared in-process store
    d = short.compute(Stimulus(target=(1,))).diagnostics
    assert d["truncation_bound"] == pytest.approx(.5, abs=.08)


def test_neglog_interval_survives_a_negative_lower_bound():
    from uncertain.core.results import StatisticEstimate
    from uncertain.warpings import NegLog
    value, se, low, high, _ = NegLog().apply(StatisticEstimate(.25, .16, -.07, .57))
    assert value == pytest.approx(-np.log(.25)) and high == np.inf
    assert low == pytest.approx(-np.log(.57))


def test_from_config_requires_n_and_short_samples():
    with pytest.raises(TypeError, match="n"):
        Measure.from_config(ToyBackend(), short_samples="compare_ragged")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="short_samples"):
        Measure.from_config(ToyBackend(), n=8)  # type: ignore[call-arg]


def test_manifest_carries_git_sha_and_seeds(backend, sampling):
    m = Measure(MeasureSpec(replace(sampling, n=4), 1, "compare_ragged"), sampler=Sampler(backend))
    manifest = Sweep(m).plan([Stimulus(target=(1,))]).run().manifest
    assert manifest["seeds"] == [sampling.seed]
    sha = manifest["git_sha"]
    assert sha is None or len(sha) == 40

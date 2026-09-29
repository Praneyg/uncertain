from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st
from uncertain import (ToyBackend, Sampler, SamplingSpec, MeasureSpec, Measure, Stimulus,
                       EstimatorSpec)
from uncertain.distances import (PrefixIndicator, NgramOverlap, NgramDistance, Discrete,
                                 Cosine, Euclidean)
from uncertain.encoders import TokensEncoder
from uncertain.statistics import Energy, Mean


@settings(max_examples=30, deadline=None, database=None)
@given(n=st.integers(1, 12), horizon=st.integers(1, 8), seed=st.integers(0, 2**32-1))
def test_sample_prefix_and_extension_properties(n, horizon, seed):
    backend = ToyBackend()
    sampler = Sampler(backend)
    spec = SamplingSpec(backend.model_spec, n, seed)
    a = sampler.sample((), horizon, spec)
    short_h = max(1, horizon//2)
    assert a.truncate(short_h).sequences == sampler.sample((), short_h, spec).sequences
    extended = sampler.extend(a, 2)
    assert extended.content_hash == sampler.sample((), horizon, replace(spec, n=n+2)).content_hash
    assert a.arrays()["tokens"].shape == (n, horizon)


@settings(max_examples=30, deadline=None, database=None)
@given(n=st.integers(1,100), horizon=st.integers(1,10), seed=st.integers(0,10**9),
       policy=st.sampled_from(["pad", "drop", "compare_ragged"]))
def test_config_roundtrip_identity(n, horizon, seed, policy):
    backend = ToyBackend()
    spec = MeasureSpec(SamplingSpec(backend.model_spec,n,seed), horizon, policy)
    assert spec.spec_hash == MeasureSpec.from_dict(spec.to_dict()).spec_hash
    assert hash(spec.sampling) == hash(MeasureSpec.from_dict(spec.to_dict()).sampling)


def test_short_sample_policies_move_the_score_in_the_documented_direction():
    """§4.2: the policy changes distances, so it cannot be a default.

    `pad` marks each missing unit with a symbol the target never contains, so a short draw
    scores further from a full-length target; `compare_ragged` simply ignores the missing
    units; `drop` removes those draws from the denominator. Hence drop < ragged < pad, and
    only drop changes n.
    """
    backend = ToyBackend(("a", "<eos>"), {(): (.6, .4)}, eos_token_id=1)
    sampling = SamplingSpec(backend.model_spec, 256, 3)
    scored = {}
    for policy in ("drop", "compare_ragged", "pad"):
        m = Measure(MeasureSpec(sampling, 3, policy, distance="ngram_distance", statistic="mean"),
                    sampler=Sampler(backend))
        r = m.compute(Stimulus((), (0,0,0)))
        scored[policy] = r
    assert scored["drop"].value < scored["compare_ragged"].value < scored["pad"].value
    assert scored["drop"].n_a_used < 256 == scored["compare_ragged"].n_a_used == scored["pad"].n_a_used
    assert scored["compare_ragged"].diagnostics.get("drop_fraction_a") is None
    assert 0 < scored["drop"].diagnostics["drop_fraction_a"] < 1


CORE_DISTANCES = (PrefixIndicator, NgramOverlap, NgramDistance, Discrete, Cosine, Euclidean)
UNIT_ROWS = (("a",), ("a","b"), ("b","a"), ("a","b","c"), ("c",))
VECTORS = (np.array([1.,0.]), np.array([0.,1.]), np.array([1.,1.]), np.array([-1.,2.]))


@pytest.mark.parametrize("factory", CORE_DISTANCES, ids=lambda f: f.__name__)
def test_core_distance_axioms(factory):
    """Only the axioms each distance actually declares.

    `kind` fixes what d(x,x) must be, `symmetric` whether d(x,y)==d(y,x), and `range` the
    interval `pairwise_blocks` enforces. No core distance declares metricity -- negative
    type does not imply it, and 1-cos violates the triangle inequality -- so no triangle
    assertion is made here. The one distance that does claim it is the TreeEdit plugin.
    """
    d = factory()
    points = VECTORS if d.input_type == "vector" else UNIT_ROWS
    identity = d.range[0] if d.kind == "dissimilarity" else d.range[1]
    for x in points:
        assert d(x, x) == pytest.approx(identity)
    for x in points:
        for y in points:
            value = d(x, y)
            assert d.range[0] <= value <= d.range[1]
            if d.symmetric:
                assert value == pytest.approx(d(y, x))


def test_effective_horizon_under_eos():
    backend = ToyBackend(("a", "<eos>"), {(): (.5, .5)}, eos_token_id=1)
    sampling = SamplingSpec(backend.model_spec, 128, 5)
    a = Sampler(backend).sample((), 4, sampling)
    stopped = [s for s in a.sequences if s.stop_reason == "eos"]
    assert stopped and all(s.tokens[-1] == 1 for s in stopped)
    assert all(s.effective_horizon == len(s.units) <= 4 for s in a.sequences)
    assert any(s.effective_horizon < 4 for s in stopped)      # EOS, not the token budget
    assert not any(s.stop_reason == "budget" for s in a.sequences)
    assert a.provenance["unterminated"] == sum(s.effective_horizon < 4 for s in a.sequences)
    diagnostics = Measure(MeasureSpec(sampling, 4, "compare_ragged"),
                          sampler=Sampler(backend)).compute(Stimulus((), (0,))).diagnostics
    assert list(diagnostics["effective_horizons"]) == [s.effective_horizon for s in a.sequences]
    assert diagnostics["eos_fraction"] > 0 and diagnostics["budget_fraction"] == 0
    assert diagnostics["truncation_bound"] == pytest.approx(a.provenance["unterminated"]/128)


def test_energy_vanishes_between_independent_draws_but_distance_does_not():
    """The contrast energy is the default statistic for.

    Two independent draws from the same predictive distribution are two different sets, so
    the mean pairwise distance stays at the population value 2p(1-p); energy sees through
    that to zero. `estimate(a, a)` is zero by construction and proves neither half.
    """
    backend = ToyBackend()
    sampler, encoder, d = Sampler(backend), TokensEncoder(), Discrete()
    estimator = EstimatorSpec(method="mc", bootstrap=199)

    def draw(seed):
        source = sampler.sample((), 1, SamplingSpec(backend.model_spec, 600, seed))
        return encoder.encode(source, "compare_ragged")

    a, b = draw(11), draw(12)
    assert a.source.content_hash != b.source.content_hash
    energy = Energy().estimate(a, b, d, estimator, (1, 2))
    mean = Mean().estimate(a, b, d, estimator, (1, 2))
    # Negative type is what energy needs and all it claims; the V-statistic's positive bias
    # is at most D/n + D/m with D = 1.
    assert energy.value >= 0
    assert energy.value < 4*energy.se + 2/600
    assert mean.value == pytest.approx(2*.75*.25, abs=.05)
    assert mean.value > 50*energy.value


def test_no_reshape_outside_the_buffer_roundtrip():
    """§10: axis reordering is named rearrangement only; reshape is banned in lint.

    Ruff's banned-api resolves qualified names, so it covers `np.reshape` but not a bare
    `x.reshape(...)` on an array. This closes that half over library code. The exemption is
    the `frombuffer` round-trip, which restores a stored shape rather than reordering axes.
    """
    root = Path(__file__).parents[1]
    offenders = [f"{path.relative_to(root)}:{i}: {line.strip()}"
                 for folder in ("src", "examples")
                 for path in (root/folder).rglob("*.py")
                 for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                 if ".reshape(" in line and "frombuffer(" not in line]
    assert offenders == []

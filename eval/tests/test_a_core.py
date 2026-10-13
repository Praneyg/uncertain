"""Category A: core measure correctness against the analytic toy-LM oracle."""
import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from uncertain import (AlternativeSet, ConfigError, EstimatorSpec, Measure, RepresentationSet,
                       Sampler, SamplingSpec, Sequence, Stimulus, UnitSpec, EncoderSpec,
                       contextual_probability, entropy, information_value, surprisal)
from uncertain.distances import Discrete, Cosine
from uncertain.encoders import TokensEncoder, LayerEncoder
from uncertain.statistics import Energy, Mean, pairwise

from conftest import make_measure
from toy_lm import ToyLM, discrete, exact_anchor, exact_mean, jaccard_distance, zscore

EST = EstimatorSpec(method="mc", bootstrap=99)
SLOW = settings(max_examples=60, deadline=None, suppress_health_check=list(HealthCheck))


def probs(seed, V, floor=0.03):
    """Random categorical with every entry >= floor (keeps P(no hit) negligible at n=512)."""
    raw = np.random.default_rng(seed).dirichlet(np.ones(V))
    return floor + (1 - V * floor) * raw


def tokens_set(lm, context, H, n, seed):
    spec = SamplingSpec(lm.model_spec, n=n, seed=seed, token_budget=16)
    return Sampler(lm).sample(context, H, spec, UnitSpec())


class Spy(Mean):
    """Mean statistic that records the operands Measure builds (for structural checks)."""
    seen = []

    def estimate(self, a, b, distance, estimator, seeds):
        Spy.seen.append((a, b))
        return super().estimate(a, b, distance, estimator, seeds)


# --------------------------------------------------------------------------- criterion 1
@pytest.mark.req("1")
def test_representation_set_retains_source():
    lm = ToyLM([.5, .3, .2], pieces=("a", "b", "c"))
    a = tokens_set(lm, (0,), 2, 8, 1)
    for enc in (TokensEncoder(), LayerEncoder(layer=0, backend=lm)):
        r = enc.encode(a, "compare_ragged")
        assert r.source is a
        assert len(r.reps) == len(a)
        row = r.source.sequences[0]
        assert row.tokens and row.model_logprobs and row.units and row.effective_horizon == 2


@pytest.mark.req("1", "9")
def test_non_factoring_statistic_reads_strings_through_measure():
    from plugin_components import SOURCE_STAT_NAME, source_text_statistic_value
    lm = ToyLM([.5, .3, .2], pieces=("a", "b", "c"))
    m = make_measure(lm, n=16, horizon=2, representation="tokens", distance="discrete",
                     statistic=SOURCE_STAT_NAME)
    r = m.compute(Stimulus((0,), (1,)))
    a = m.store.get_or_sample(m.sampler, (0,), 2, m.spec.sampling, m.spec.unit)
    assert r.raw_value == source_text_statistic_value(a)


# --------------------------------------------------------------------------- criterion 2
@pytest.mark.req("2")
@SLOW
@given(seed=st.integers(0, 10**6), V=st.integers(2, 8), data=st.data(),
       pair=st.sampled_from([("tokens", "discrete"), ("layer:0", "cosine")]))
def test_h1_information_value_equals_one_minus_p(seed, V, data, pair):
    p = probs(seed, V)
    o = data.draw(st.integers(0, V - 1))
    lm = ToyLM(p)
    m = make_measure(lm, n=512, seed=seed, representation=pair[0], distance=pair[1], statistic="mean")
    r = m.compute(Stimulus((0,), (o,)))
    assert r.n == 512 and r.se is not None
    assert zscore(r.value, r.se, 1 - p[o]) <= 4, (r.value, r.se, 1 - p[o])


@pytest.mark.req("2", "36")
def test_anchor_at_h1_is_singleton():
    lm = ToyLM([.5, .5])
    obs = Stimulus((0,), (1,)).observed(lm, UnitSpec())
    b = AlternativeSet.anchor(obs, AlternativeSet.empty((0, 1), UnitSpec()), (0,))
    assert len(b) == 1 and b.deterministic and b.sequences[0].tokens == (1,)


@pytest.mark.req("2")
def test_edge_set_of_size_one(measure):
    # Edge case: a set of size 1 -- SE is undefined; library must not crash or invent one.
    lm = ToyLM([.5, .3, .2])
    r = make_measure(lm, n=1, representation="tokens", distance="discrete", statistic="mean").compute(
        Stimulus((0,), (1,)))
    measure("status", r.status)
    assert r.value in (0., 1.) and r.n == 1
    assert r.se is None and "insufficient" in r.status


@pytest.mark.req("2", "4")
@pytest.mark.parametrize("statistic", ["mean", "energy"])
def test_edge_identical_alternatives(statistic):
    # Edge case: identical alternatives -- p is a point mass on the observed token.
    lm = ToyLM([0., 1., 0.])
    r = make_measure(lm, n=64, representation="tokens", distance="discrete", statistic=statistic
                     ).compute(Stimulus((0,), (1,)))
    assert r.value == 0.


# --------------------------------------------------------------------------- criterion 3 / 7
CTX_LM = dict(p=[.5, .3, .2], rows={0: [.6, .3, .1], 1: [.2, .2, .6], 2: [.1, .8, .1]})


@pytest.mark.req("3")
@pytest.mark.parametrize("H", [1, 2, 3])
@pytest.mark.parametrize("o", [0, 1, 2])
def test_operands_have_spec_structure(H, o):
    lm = ToyLM(**CTX_LM)
    Spy.seen.clear()
    spy = Spy()
    spy.descriptor = {"statistic": "eval-spy", "version": "1"}
    m = make_measure(lm, n=32, horizon=H, seed=3, representation="tokens", distance="discrete")
    m = Measure(m.spec, sampler=m.sampler, statistic=spy)
    m.compute(Stimulus((0,), (o,)))
    ra, rb = Spy.seen[-1]
    a, b = ra.source, rb.source
    direct = Sampler(lm).sample((0,), H, m.spec.sampling, m.spec.unit)
    assert a.sequences == direct.sequences                    # A = trunc(S_t, H)
    assert b.horizon == H and b.context == (0,)
    assert all(row.tokens[0] == o and row.observed[0] for row in b.sequences)
    if H == 1:
        assert len(b) == 1 and b.deterministic
    else:                                                     # continuation ~ p(.|c, o)
        cont = Sampler(lm).sample((0, o), H - 1, m.spec.sampling, m.spec.unit)
        assert [row.tokens[1:] for row in b.sequences] == [row.tokens for row in cont.sequences]


@pytest.mark.req("3")
@pytest.mark.parametrize("H", [1, 2, 3])
@pytest.mark.parametrize("o", [0, 1, 2])
@pytest.mark.parametrize("dist", ["discrete", "ngram_distance"])
def test_horizon_mean_matches_exact_enumeration(H, o, dist, measure):
    lm = ToyLM(**CTX_LM)
    m = make_measure(lm, n=384, horizon=H, seed=11 + o, representation="tokens", distance=dist,
                     statistic="mean")
    r = m.compute(Stimulus((0,), (o,)))
    d = discrete if dist == "discrete" else jaccard_distance(1)
    truth = exact_mean(lm.sequences((0,), H), exact_anchor(lm, (0,), (o,), H), d)
    z = zscore(r.value, r.se, truth)
    measure("abs_err", abs(r.value - truth)); measure("z", z)
    assert z <= 4, (r.value, r.se, truth)


@pytest.mark.req("7")
@pytest.mark.parametrize("target,H", [((1, 2), 2), ((1, 2), 3), ((1, 2, 1), 3)])
def test_multi_unit_span_matches_exact(target, H, measure):
    lm = ToyLM(**CTX_LM)
    m = make_measure(lm, n=384, horizon=H, seed=5, representation="tokens", distance="discrete",
                     statistic="mean")
    r = m.compute(Stimulus((0,), target))
    truth = exact_mean(lm.sequences((0,), H), exact_anchor(lm, (0,), target, H), discrete)
    measure("z", zscore(r.value, r.se, truth))
    assert zscore(r.value, r.se, truth) <= 4


@pytest.mark.req("7")
def test_span_longer_than_horizon_rejected():
    lm = ToyLM(**CTX_LM)
    m = make_measure(lm, n=8, horizon=1, representation="tokens", distance="discrete", statistic="mean")
    calls = lm.calls
    with pytest.raises(ConfigError):
        m.compute(Stimulus((0,), (1, 2)))
    assert lm.calls == calls


# --------------------------------------------------------------------------- criterion 4 / 5
def manual_set(rows, salt=0):
    seqs = tuple(Sequence(tuple(r), (-.1,) * len(r), (-.1,) * len(r), tuple(str(t) for t in r),
                          tuple((i, i + 1) for i in range(len(r))), sample_id=i)
                 for i, r in enumerate(rows))
    return AlternativeSet(seqs, (salt,), UnitSpec(), max(len(r) for r in rows))


row_lists = st.lists(st.lists(st.integers(0, 3), min_size=1, max_size=3), min_size=2, max_size=12)


@pytest.mark.req("4")
@settings(max_examples=250, deadline=None)
@given(rows_a=row_lists, rows_b=row_lists)
def test_energy_identity_nonneg_symmetry_formula(rows_a, rows_b):
    a, b = manual_set(rows_a, 1), manual_set(rows_b, 2)
    ra, rb = TokensEncoder().encode(a, "compare_ragged"), TokensEncoder().encode(b, "compare_ragged")
    d, e = Discrete(), Energy()
    ab = e.estimate(ra, rb, d, EST, (1, 2)).value
    ba = e.estimate(rb, ra, d, EST, (2, 1)).value
    aa = e.estimate(ra, ra, d, EST, (1, 1)).value
    formula = 2 * pairwise(ra, rb, d).mean() - pairwise(ra, ra, d).mean() - pairwise(rb, rb, d).mean()
    assert abs(aa) <= 1e-12
    assert ab >= -1e-12
    assert abs(ab - ba) <= 1e-12
    assert abs(ab - formula) <= 1e-12


@pytest.mark.req("4")
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(2, 10), m=st.integers(2, 10), dim=st.integers(2, 6))
def test_energy_cosine_vectors_nonneg_symmetric(seed, n, m, dim):
    rng = np.random.default_rng(seed)
    a, b = manual_set([[0]] * n, 1), manual_set([[0]] * m, 2)
    ra = RepresentationSet(tuple(rng.normal(size=dim) for _ in range(n)), a, {"e": "v", "k": (seed, n, m, dim, "a")}, "vector")
    rb = RepresentationSet(tuple(rng.normal(size=dim) for _ in range(m)), b, {"e": "v", "k": (seed, n, m, dim, "b")}, "vector")
    e, d = Energy(), Cosine()
    ab, ba = e.estimate(ra, rb, d, EST, (1, 2)).value, e.estimate(rb, ra, d, EST, (2, 1)).value
    assert ab >= -1e-12 and abs(ab - ba) <= 1e-12


@pytest.mark.req("4", "5")
@pytest.mark.parametrize("n", [128, 512, 1024])
def test_energy_vanishes_and_mean_does_not_when_p_equals_q(n, measure):
    p = np.array([.35, .25, .2, .12, .08])
    lm = ToyLM(p)
    a, b = tokens_set(lm, (0,), 1, n, 1), tokens_set(lm, (0,), 1, n, 2)
    ra, rb = TokensEncoder().encode(a, "compare_ragged"), TokensEncoder().encode(b, "compare_ragged")
    energy = Energy().estimate(ra, rb, Discrete(), EST, (1, 2))
    mean = Mean().estimate(ra, rb, Discrete(), EST, (1, 2))
    self_distance = 1 - (p ** 2).sum()
    for k, v in (("energy", energy.value), ("energy_se", energy.se), ("energy_ci_low", energy.ci_low),
                 ("energy_over_se", energy.value / energy.se), ("mean", mean.value),
                 ("mean_truth", self_distance), ("mean_ci_low", mean.ci_low)):
        measure(f"n{n}:{k}", v)
    # energy: indistinguishable from zero
    assert energy.ci_low <= 0 <= energy.ci_high
    assert abs(energy.value) <= 4 * energy.se, (energy.value, energy.se)
    # mean: bounded away from zero, at 1 - sum p^2
    assert zscore(mean.value, mean.se, self_distance) <= 4
    assert mean.ci_low > 0


@pytest.mark.req("4")
@pytest.mark.parametrize("o", [0, 2, 4])
def test_energy_against_point_mass_matches_closed_form(o, measure):
    # energy(P, delta_o) = 1 - 2p[o] + sum p^2 (H = 1, right operand is a singleton)
    p = np.array([.35, .25, .2, .12, .08])
    lm = ToyLM(p)
    r = make_measure(lm, n=1024, seed=o, representation="tokens", distance="discrete",
                     statistic="energy").compute(Stimulus((0,), (o,)))
    truth = 1 - 2 * p[o] + (p ** 2).sum()
    measure("z", zscore(r.value, r.se, truth))
    assert zscore(r.value, r.se, truth) <= 4


# --------------------------------------------------------------------------- criterion 6 / 38
@pytest.mark.req("6")
def test_layer_selection_changes_vectors_and_hash(tiny_gpt2, measure):
    from uncertain import Measure as M
    spec = SamplingSpec(tiny_gpt2.model_spec, n=4, seed=0, token_budget=8)
    a = Sampler(tiny_gpt2).sample((1, 2, 3), 2, spec, UnitSpec())
    v0 = np.array(LayerEncoder(layer=0, backend=tiny_gpt2).encode(a, "compare_ragged").reps)
    v2 = np.array(LayerEncoder(layer=2, backend=tiny_gpt2).encode(a, "compare_ragged").reps)
    measure("max_abs_diff", float(np.abs(v0 - v2).max()))
    assert np.abs(v0 - v2).max() > 1e-3
    kw = dict(n=4, short_samples="compare_ragged", distance="cosine", statistic="mean")
    m0 = M.from_config(tiny_gpt2, representation="layer:0", **kw)
    m2 = M.from_config(tiny_gpt2, level="layer:2", **kw)
    assert m0.spec.representation == EncoderSpec("layer", {"layer": 0})
    assert m2.spec.representation == EncoderSpec("layer", {"layer": 2})
    assert m0.spec_hash != m2.spec_hash
    assert m0.replace(level="layer:2").spec_hash == m2.spec_hash


@pytest.mark.req("6")
def test_toy_layer_is_one_hot():
    lm = ToyLM([.25] * 4)
    a = tokens_set(lm, (0,), 1, 16, 0)
    reps = LayerEncoder(layer=0, backend=lm).encode(a, "compare_ragged").reps
    for row, v in zip(a.sequences, reps):
        assert np.array_equal(v, np.eye(4)[row.tokens[0]])


@pytest.mark.req("38")
def test_pooling_policy_changes_vectors_and_hash():
    lm = ToyLM([.25] * 4)
    a = tokens_set(lm, (0,), 3, 16, 0)
    seen, hashes = {}, set()
    for pooling in ("mean", "first", "last", "sum"):
        seen[pooling] = np.array(LayerEncoder(layer=0, pooling=pooling, backend=lm).encode(a, "compare_ragged").reps)
        m = make_measure(lm, n=4, horizon=3, distance="cosine", statistic="mean",
                         representation=EncoderSpec("layer", {"layer": 0, "pooling": pooling}))
        hashes.add(m.spec_hash)
    assert len(hashes) == 4
    assert not np.array_equal(seen["first"], seen["last"])
    assert not np.array_equal(seen["mean"], seen["sum"])


# --------------------------------------------------------------------------- criterion 26
P26 = np.array([.4, .3, .2, .1])


def preset_measure(lm, spec_fn, **kw):
    sampling = SamplingSpec(lm.model_spec, n=kw.pop("n", 2048), seed=kw.pop("seed", 0), token_budget=16)
    return Measure(spec_fn(sampling, short_samples="compare_ragged",
                           estimator=EstimatorSpec(method="mc", bootstrap=99), **kw), sampler=Sampler(lm))


@pytest.mark.req("26")
@pytest.mark.parametrize("name,spec_fn,kw,truth", [
    ("surprisal", surprisal, {}, -math.log(P26[2])),
    ("contextual_probability", contextual_probability, {}, P26[2]),
    ("next_unit_entropy", entropy, {}, float(-(P26 * np.log(P26)).sum())),
    ("sequence_entropy_H2", entropy, {"horizon": 2}, float(-2 * (P26 * np.log(P26)).sum())),
    ("information_value", information_value, {}, 1 - P26[2]),
    ("anticipatory_iv", information_value, {"mode": "anticipatory", "n": 512}, 1 - (P26 ** 2).sum()),
])
def test_published_presets_match_analytic(name, spec_fn, kw, truth, measure):
    lm = ToyLM(P26)
    r = preset_measure(lm, spec_fn, **kw).compute(Stimulus((0,), (2,)))
    z = zscore(r.value, r.se, truth)
    measure("value", r.value); measure("truth", truth); measure("se", r.se); measure("z", z)
    assert z <= 4, (r.value, r.se, truth)


# --------------------------------------------------------------------------- criterion 30
@pytest.mark.req("30")
def test_alternative_set_fields_and_provenance():
    lm = ToyLM([.4, .3, .2, .1], pieces=("a", "b", " ", "."), eos_token_id=3)
    spec = SamplingSpec(lm.model_spec, n=12, seed=9, token_budget=10, generation={"temperature": 1.0})
    a = Sampler(lm).sample((0,), 2, spec, UnitSpec("word", "whitespace:v1"))
    arr = a.arrays()
    width = max(len(s.tokens) for s in a.sequences)
    for key in ("tokens", "mask", "model_logprobs", "sampling_logprobs"):
        assert arr[key].shape == (12, width)
    for i, s in enumerate(a.sequences):
        assert arr["mask"][i].sum() == len(s.tokens)
        assert s.effective_horizon == len(s.units) == len(s.unit_spans)
    prov = a.provenance
    assert prov["sampling"]["model"] == {"model_id": "eval-toy", "revision": lm.model_spec.revision}
    assert prov["sampling"]["seed"] == 9 and prov["sampling"]["n"] == 12
    assert prov["sampling"]["generation"]["top_k"] == 0
    assert a.context == (0,) and a.horizon == 2                       # context + H_max
    assert a.unit.name == "word" and a.unit.boundary == "whitespace:v1"  # unit + boundary detector

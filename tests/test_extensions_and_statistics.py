from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from uncertain import (Measure, Stimulus, EncoderSpec, StatisticSpec, ConfigError, contrast,
                       RepresentationSet, register_distance)
from uncertain.distances import Discrete, Cosine, Euclidean
from uncertain.statistics import (pairwise, Energy, Mean, DirectedMin, lifted, _WITHIN_BLOCKS)


def load_example(name):
    path = Path(__file__).parents[1]/"examples"/name/"plugin.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pos_extension_requires_no_core_changes(make_measure, backend):
    plugin = load_example("pos")
    encoder = plugin.POSEncoder(lambda text: ["A" if c == "a" else "B" for c in text],
                               tagger_id="fixture:v1", backend=backend)
    base = make_measure(distance="ngram_distance")
    m = Measure(base.spec, sampler=base.sampler, encoder=encoder)
    assert 0 <= m.compute(Stimulus((), (1,))).value <= 1
    assert m.spec_hash != base.spec_hash


def test_nonfactoring_entailment_extension(make_measure):
    module = load_example("entailment")
    base = make_measure()
    spec = base.spec.replace(representation=EncoderSpec("strings"))
    stat = module.EntailmentEntropy(lambda a,b: a == b, model_id="exact-equality:v1")
    m = Measure(spec, sampler=base.sampler, distance=module.StringMismatch(), statistic=stat)
    r = m.compute(Stimulus((), (1,)))
    assert 0 < r.value < 1 and r.status == "cluster_uncertainty_unavailable"


def test_custom_registered_distance(make_measure):
    @register_distance("test_custom")
    class Custom(Discrete):
        descriptor = {"distance": "test_custom", "version": "1"}
    r = make_measure(distance="test_custom").compute(Stimulus((), (1,)))
    assert r.value > .5
    with pytest.raises(ConfigError, match="already registered"):
        register_distance("test_custom")(Custom)


def test_blocked_energy_matches_dense_and_permutations(make_measure):
    m = make_measure(horizon=2, distance="discrete", statistic=StatisticSpec("energy", {"block_size": 7}))
    a = m._encode(m._get((), 2).select(range(13)))
    b = m._encode(m._get((1,), 2).select(range(9)))
    d = Discrete()
    expected = 2*pairwise(a,b,d).mean()-pairwise(a,a,d).mean()-pairwise(b,b,d).mean()
    r = Energy(block_size=3).estimate(a,b,d,m.spec.estimator,(1,2))
    assert r.value == pytest.approx(expected)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        ia, ib = rng.permutation(len(a.source)), rng.permutation(len(b.source))
        pa = RepresentationSet(tuple(a.reps[i] for i in ia), a.source.select(ia), a.descriptor, a.input_type)
        pb = RepresentationSet(tuple(b.reps[i] for i in ib), b.source.select(ib), b.descriptor, b.input_type)
        actual = Energy(block_size=4).estimate(pa,pb,d,m.spec.estimator,(1,2))
        assert actual.value == pytest.approx(expected)
    same = Energy().estimate(a,a,d,m.spec.estimator,(1,1))
    assert same.value == pytest.approx(0, abs=1e-15)


def test_cosine_mean_only_limitation():
    d = Cosine()
    a = [np.array([1.,0.]), np.array([-1.,0.])]
    b = [np.array([0.,1.]), np.array([0.,-1.])]
    def avg(x,y):
        return np.mean([d(u,v) for u in x for v in y])
    assert 2*avg(a,b)-avg(a,a)-avg(b,b) == pytest.approx(0)
    with pytest.raises(ValueError, match="zero vector"):
        d(np.zeros(2), a[0])


def test_contrasts_preserve_shared_samples(make_measure):
    m = make_measure()
    left = m.compute(Stimulus((), (1,)))
    same = m.compute(Stimulus((), (1,)))
    r = contrast(left, same)
    assert r.value == 0 and r.se == pytest.approx(0)
    other = m.compute(Stimulus((), (0,)))
    c = contrast(left, other)
    # Complementary indicator probabilities have negative covariance.
    assert c.se > np.sqrt(left.se**2+other.se**2)*1.1


def test_contrast_rejects_different_shared_subsets(make_measure):
    r = make_measure().compute(Stimulus((), (1,)))
    changed = replace(r, diagnostics={**r.diagnostics, "sample_ids_a": [999]})
    with pytest.raises(ConfigError, match="retained samples"):
        contrast(r, changed)


def test_external_tree_edit_dependency():
    pytest.importorskip("zss")
    module = load_example("tree_edit")
    distance = module.TreeEdit()
    a = ("root", (("a", ()), ("b", ())))
    b = ("root", (("a", ()), ("c", ())))
    assert distance(a,a) == 0
    assert distance(a,b) == 1
    assert distance(a,b) == distance(b,a)


class CountingDiscrete(Discrete):
    """A distinct descriptor keeps these counts out of any other test's cache entries."""
    descriptor = {"distance": "counting_discrete", "version": "1"}

    def __init__(self):
        self.calls = 0

    def __call__(self, a, b):
        self.calls += 1
        return super().__call__(a, b)


@pytest.fixture
def energy_operands(make_measure):
    m = make_measure(horizon=2, distance="discrete", statistic=StatisticSpec("energy"))
    sets = [m._encode(m._get(ctx, 2).select(range(20))) for ctx in ((), (1,), (0,))]
    return m.spec.estimator, sets


def test_within_set_blocks_are_reused_across_calls_and_cells(energy_operands):
    estimator, (a, b, c) = energy_operands
    stat, d = Energy(), CountingDiscrete()
    n = len(a.reps)*len(b.reps)
    _WITHIN_BLOCKS.clear()
    first = stat.estimate(a, b, d, estimator, (1, 2))
    assert d.calls == 3*n  # cross(a,b) + within(a,a) + within(b,b)
    repeat = stat.estimate(a, b, d, estimator, (1, 2))
    assert d.calls == 4*n  # only the cross block; both within blocks were reused
    assert repeat.value == pytest.approx(first.value)
    assert repeat.bootstrap_values == first.bootstrap_values
    stat.estimate(a, c, d, estimator, (1, 3))
    assert d.calls == 6*n  # cross(a,c) + within(c,c); d(A,A) not re-evaluated


def test_within_set_cache_key_separates_block_size_and_seed(energy_operands):
    estimator, (a, b, _) = energy_operands
    d = CountingDiscrete()
    n = len(a.reps)*len(b.reps)
    _WITHIN_BLOCKS.clear()
    Energy(block_size=7).estimate(a, b, d, estimator, (1, 2))
    Energy(block_size=5).estimate(a, b, d, estimator, (1, 2))
    assert d.calls == 6*n  # block_size is in the key: no stale reuse
    Energy(block_size=7).estimate(a, b, d, estimator, (4, 2))
    assert d.calls == 8*n  # a's seed changed, so only within(b,b) is reused


def test_injected_parameters_enter_the_spec_hash(make_measure):
    class Scaled:
        input_type, range = "units", (0., 1.)
        negative_type, kind = True, "dissimilarity"

        def __init__(self, w):
            self.w = w

        def __call__(self, a, b):
            return self.w*float(a != b)
    m = make_measure(horizon=2, statistic="mean")
    full, quarter = m.replace(distance=Scaled(1.)), m.replace(distance=Scaled(.25))
    assert full.spec_hash != quarter.spec_hash
    assert full.spec_hash == m.replace(distance=Scaled(1.)).spec_hash
    unhashable = Scaled(1.)
    unhashable.w = object()
    with pytest.raises(ConfigError, match="descriptor"):
        m.replace(distance=unhashable)


def test_u_energy_declares_its_negative_range(make_measure):
    assert Energy(variant="u").range_for(Discrete()) == (-2., 2.)
    with pytest.raises(ConfigError, match="domain"):
        make_measure(horizon=2, distance="discrete", warping="neglog",
                     statistic=StatisticSpec("energy", {"variant": "u"}))


def test_neglog_reaches_zero_only_for_probabilities(make_measure):
    # §4.5: -log is undefined at 0. A probability (mean of a [0,1] similarity) may
    # still hit 0, the infinite surprisal of an unseen event; nothing else may.
    for distance in ("prefix_indicator", "ngram_overlap"):
        make_measure(horizon=2, distance=distance, statistic="mean", warping="neglog")
    for distance, statistic in (("discrete", "mean"), ("discrete", "energy"),
                                ("discrete", "min"), ("prefix_indicator", "min")):
        with pytest.raises(ConfigError, match="only a probability"):
            make_measure(horizon=2, distance=distance, statistic=statistic, warping="neglog")


def test_statistic_range_for_tracks_the_distance():
    assert Mean().range_for(Discrete()) == (0., 1.)
    assert DirectedMin().range_for(Cosine()) == (0., 2.)
    assert Energy().range_for(Cosine()) == (0., 4.)
    assert Energy().range_for(Euclidean()) == (0., float("inf"))
    assert Mean().range_for(Euclidean()) == (0., float("inf"))


def test_lifted_builds_a_new_factorable_statistic(energy_operands):
    estimator, (a, b, _) = energy_operands
    d = Discrete()
    stat = lifted(d, np.median)
    hand_rolled = np.median([[d(u, v) for v in b.reps] for u in a.reps])
    assert stat.estimate(a, b, d, estimator, (1, 2)).value == pytest.approx(hand_rolled)
    assert stat.range_for(d) == (0., 1.)
    assert stat.descriptor["statistic"] == "lifted_median"
    with pytest.raises(ConfigError, match="named reduction"):
        lifted(d, lambda m: m.max())
    with pytest.raises(ConfigError, match="bound to a different distance"):
        stat.estimate(a, b, Cosine(), estimator, (1, 2))

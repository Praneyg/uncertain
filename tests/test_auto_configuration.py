import math
import numpy as np
import pytest
from uncertain import (Measure, MeasureSpec, SamplingSpec, Sampler, ToyBackend, Stimulus,
                       UnitSpec, EncoderSpec, EstimatorSpec, RepresentationSet, Sweep, ConfigError,
                       register_encoder, register_distance, register_warping, register_unit,
                       register_sampler)
from uncertain.registry import resolve


@register_encoder("test_auto_vectors")
class TestVectors:
    __test__ = False
    output_type = "vector"
    supported_units = {"token"}
    descriptor = {"encoder": "test_auto_vectors", "version": "1"}

    def __init__(self, backend=None):
        pass

    def encode(self, source, short_samples):
        vectors = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        return RepresentationSet(tuple(vectors[r.tokens[0]] for r in source.sequences),
                                 source, self.descriptor, "vector")


@register_distance("test_auto_similarity")
class Similarity:
    input_type = "vector"
    kind = "similarity"
    range = (0., 1.)
    negative_type = False
    descriptor = {"distance": "test_auto_similarity", "version": "1"}

    def __call__(self, a, b):
        return float(math.exp(-np.linalg.norm(a-b)))


@register_warping("test_auto_square")
class Square:
    descriptor = {"warping": "test_auto_square", "version": "1"}
    domain = (-math.inf, math.inf)

    def apply(self, estimate):
        return (estimate.value**2, 2*estimate.value*estimate.se,
                estimate.ci_low**2, estimate.ci_high**2, 0.)


def test_defaults_are_raw_sampling_and_exact_token_energy():
    b = ToyBackend()
    m = Measure.from_config(b)
    assert m.spec.sampling.generation == {
        'temperature': 1., 'top_k': 0, 'top_p': 1., 'do_sample': True, 'num_beams': 1}
    r = m.compute(Stimulus(target=(1,)))
    assert r.value == pytest.approx(1.125)
    assert r.method == 'exact' and r.se == 0 and r.n_a_used == r.n_b_used == 0
    assert m.sampler.calls == 0
    assert r.diagnostics['support_size'] == 2


def test_auto_routes_longer_horizon_to_mc_but_exact_override_still_works():
    m = Measure.from_config(ToyBackend(), distance='prefix_indicator', statistic='mean',
                            horizon=2, n=40)
    r = m.compute(Stimulus(target=(1,)))
    assert r.method == 'mc' and m.sampler.calls > 0
    explicit = m.replace(estimator=EstimatorSpec(method='exact'))
    before = m.sampler.calls
    assert explicit.compute(Stimulus(target=(1,))).value == .25
    assert m.sampler.calls == before


@pytest.mark.parametrize('unit,vocab', [('word', ('a ', 'b ')), ('sentence', ('A. ', 'B. '))])
def test_auto_routes_text_units_to_mc(unit, vocab):
    m = Measure.from_config(ToyBackend(vocab), unit=unit, n=32)
    r = m.compute(Stimulus(target=(1,)))
    assert r.method == 'mc' and r.n_a_used == 32


def test_explicit_mc_at_horizon_one_preserves_sampling():
    m = Measure.from_config(ToyBackend(), estimator='mc', n=32)
    r = m.compute(Stimulus(target=(1,)))
    assert r.method == 'mc' and m.sampler.calls == 1


def test_weighted_mean_and_meister_style_similarity_warp():
    b = ToyBackend()
    m = Measure.from_config(b, representation='test_auto_vectors',
                            distance='test_auto_similarity', statistic='mean', warping='neglog')
    r = m.compute(Stimulus(target=(1,)))
    expected_similarity = .75 * math.exp(-math.sqrt(2)) + .25
    assert r.raw_value == pytest.approx(expected_similarity)
    assert r.value == pytest.approx(-math.log(expected_similarity))
    assert r.se == 0 and m.sampler.calls == 0


@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
def test_exact_energy_against_independent_dense_matrix_including_eos(distance):
    from uncertain.distances import Cosine, Euclidean
    p = np.array([.6, .3, .1])
    b = ToyBackend(('a', 'b', '<eos>'), {(): tuple(p)}, eos_token_id=2)
    m = Measure.from_config(b, representation=EncoderSpec('test_auto_vectors'), distance=distance)
    r = m.compute(Stimulus(target=(1,)))
    v = np.array([[1.,0.], [0.,1.], [-1.,0.]])
    d = Cosine() if distance == 'cosine' else Euclidean()
    matrix = np.array([[d(x,y) for y in v] for x in v])
    assert r.value == pytest.approx(2*p@matrix[:,1] - p@matrix@p)
    assert r.diagnostics['support_size'] == 3
    assert r.diagnostics['enumerated_probability_mass'] == pytest.approx(1)
    assert m.sampler.calls == 0


@register_unit("test_auto_clause")
class ClauseUnit:
    boundary = "clause:v1"
    descriptor = {"unit": "test_auto_clause", "boundary": "clause:v1", "version": "1"}


@register_encoder("test_auto_clause_tokens")
class ClauseTokens:
    __test__ = False
    output_type = "units"
    supported_units = {"test_auto_clause"}
    descriptor = {"encoder": "test_auto_clause_tokens", "version": "1"}

    def __init__(self, backend=None):
        pass

    def encode(self, source, short_samples):  # pragma: no cover - not reached without a detector
        raise AssertionError


@register_sampler("test_auto_named")
class NamedSampler(Sampler):
    pass


def test_fourth_unit_needs_no_core_edit():
    """§7's own test: this unit and its encoder are declared entirely in this file."""
    assert UnitSpec("test_auto_clause", "clause:v1").name == "test_auto_clause"
    m = Measure.from_config(ToyBackend(), unit="test_auto_clause",
                            representation="test_auto_clause_tokens", statistic="mean",
                            distance="prefix_indicator")
    assert m.spec.unit == UnitSpec("test_auto_clause", "clause:v1")
    with pytest.raises(ConfigError, match="supported boundary detector"):
        UnitSpec("test_auto_clause", "whitespace:v1")
    with pytest.raises(ConfigError, match="Unknown unit"):
        Measure.from_config(ToyBackend(), unit="test_auto_absent")


def test_sampler_resolves_by_registered_name():
    b = ToyBackend()
    assert isinstance(resolve("sampler", "test_auto_named", backend=b), NamedSampler)
    assert isinstance(Measure.from_config(b, sampler="test_auto_named").sampler, NamedSampler)
    with pytest.raises(ConfigError, match="requires a backend"):
        Measure.from_config(sampler="test_auto_named")


def test_printed_examples_from_section_five_run():
    m = Measure.from_config(ToyBackend(), statistic="mean", distance="prefix_indicator")
    assert m.replace(unit="token").spec.unit == UnitSpec()
    assert m.replace(representation="tokens").spec.representation == EncoderSpec("tokens")
    # §5's Sweep(level=["layer:0", "layer:8"]) grid, parsed without a hidden-state backend.
    assert [m.spec.replace(level=x).representation for x in ("layer:0", "layer:8")] == [
        EncoderSpec("layer", {"layer": 0}), EncoderSpec("layer", {"layer": 8})]
    assert Measure.from_config(ToyBackend(), level="test_auto_vectors",
                               distance="test_auto_similarity",
                               statistic="mean").spec.representation.name == "test_auto_vectors"
    assert Measure.from_config(ToyBackend(), mode="entropy", statistic="mean",
                               distance="prefix_indicator").spec.mode == "entropy"
    # §5 spells the encoder axis "level"; the grid values normalize like any other.
    assert [x.spec.representation.name for x in Sweep(m, level=["tokens"]).measures] == ["tokens"]


def test_warping_domain_must_contain_statistic_range():
    @register_warping("test_auto_positive_only")
    class PositiveOnly:
        descriptor = {"warping": "test_auto_positive_only", "version": "1"}
        domain = (1., 2.)

        def apply(self, estimate):  # pragma: no cover - rejected before use
            raise AssertionError
    with pytest.raises(ConfigError, match="does not contain statistic"):
        Measure.from_config(ToyBackend(), warping="test_auto_positive_only")


def test_custom_warp_can_follow_energy():
    m = Measure.from_config(ToyBackend(), warping='test_auto_square')
    r = m.compute(Stimulus(target=(1,)))
    assert r.raw_value == pytest.approx(1.125)
    assert r.value == pytest.approx(1.125**2)


def test_filtered_q_is_explicit_and_exact():
    with pytest.raises(ConfigError, match='samples follow q'):
        Measure.from_config(ToyBackend(), generation={'temperature': .5})
    m = Measure.from_config(ToyBackend(), distance='prefix_indicator', statistic='mean',
                            generation={'temperature': .5}, probability='sampling')
    assert m.compute(Stimulus(target=(1,))).value == pytest.approx(.1)


def test_custom_sampler_selection_and_no_work_in_exact_sweep():
    class SelectedSampler(Sampler):
        def sample(self, *args, **kwargs):
            raise AssertionError('Exact configuration must not call the selected sampler')
    sampler = SelectedSampler(ToyBackend())
    m = Measure.from_config(sampler=sampler)
    assert m.sampler is sampler
    plan = Sweep(m, statistic=['mean', 'energy']).plan([Stimulus(target=(1,))])
    assert plan.describe()['sample_requests'] == 0
    assert all(r['method'] == 'exact' for r in plan.run().rows)


def test_unsupported_custom_statistic_is_not_falsely_labeled_exact():
    m = Measure.from_config(ToyBackend(), statistic='min', n=16)
    r = m.compute(Stimulus(target=(1,)))
    assert r.method == 'mc'
    assert r.diagnostics['estimator_selection'] == 'mc_exact_configuration_unsupported'
    with pytest.raises(ConfigError, match='no exact path'):
        m.replace(estimator=EstimatorSpec(method='exact'))


def test_low_level_presets_also_default_to_auto():
    b = ToyBackend()
    m = Measure(MeasureSpec(SamplingSpec(b.model_spec, 2, 0), 1, 'compare_ragged'),
                sampler=Sampler(b))
    assert m.compute(Stimulus(target=(1,))).value == .25
    assert m.sampler.calls == 0

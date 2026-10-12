from dataclasses import replace
import math
import numpy as np
import pytest
from uncertain import (Stimulus, EstimatorSpec, SamplingSpec, ToyBackend, Sampler, Measure,
                       MeasureSpec, StatisticSpec, ConfigError, UnitSpec)


def test_exact_probability_surprisal_and_span(make_measure):
    exact = EstimatorSpec(method="exact")
    item = Stimulus((), (1,))
    p = make_measure(estimator=exact).compute(item)
    s = make_measure(estimator=exact, warping="neglog").compute(item)
    assert p.value == .25
    assert s.value == pytest.approx(math.log(4))
    assert s.se == 0 and s.n_a_used == 0 and s.method == "exact"
    joint = make_measure(horizon=3, estimator=exact).compute(Stimulus((), (0,1)))
    assert joint.value == .75*.25


def test_exact_entropy_and_mc_sequence_entropy(make_measure):
    expected = -.75*math.log(.75)-.25*math.log(.25)
    r = make_measure(mode="entropy", estimator=EstimatorSpec(method="exact")).compute(Stimulus())
    assert r.value == pytest.approx(expected)
    r = make_measure(mode="entropy", horizon=3).compute(Stimulus())
    assert abs(r.value-3*expected) < 4*r.se


def test_probability_mc_convergence_and_interval(make_measure, sampling):
    errors = []
    for n in (64, 2048):
        per_seed = []
        covered = 0
        for seed in range(12):
            m = make_measure(sampling=replace(sampling, n=n, seed=seed))
            r = m.compute(Stimulus((), (1,)))
            per_seed.append(abs(r.value-.25))
            covered += r.ci_low <= .25 <= r.ci_high
        errors.append(np.mean(per_seed))
        assert covered >= 9
    assert errors[1] < errors[0]/2


def test_zero_hit_surprisal_has_infinite_upper_ci():
    backend = ToyBackend(transitions={(): (1., 0.)})
    s = SamplingSpec(backend.model_spec, 10, 0)
    m = Measure(MeasureSpec(s, 2, "compare_ragged", warping="neglog"), sampler=Sampler(backend))
    r = m.compute(Stimulus((), (1,)))
    assert math.isinf(r.value) and math.isinf(r.ci_high)
    assert math.isfinite(r.ci_low) and r.se is None
    assert r.status == "zero_hits" and r.warping_bias is None


def test_exact_neglog_avoids_probability_underflow():
    backend = ToyBackend(transitions={(): (.99, .01)})
    spec = MeasureSpec(SamplingSpec(backend.model_spec, 1, 0, token_budget=200), 200,
                       "compare_ragged", warping="neglog", estimator=EstimatorSpec(method="exact"))
    r = Measure(spec, sampler=Sampler(backend)).compute(Stimulus((), (1,)*200))
    assert r.raw_value == 0.
    assert r.value == pytest.approx(-200*math.log(.01))


def test_warp_bias_and_correction(make_measure):
    m = make_measure(warping="neglog", estimator=EstimatorSpec(method="mc", correct_bias=True))
    r = m.compute(Stimulus((), (1,)))
    assert r.warping_bias > 0
    assert r.corrected_value == r.value-r.warping_bias
    assert r.ci_low <= r.value <= r.ci_high


def test_energy_against_closed_form_deterministic_target(make_measure):
    # d(x,y)=1[x!=y], P(a)=.75, target=b => population energy=2*(.75)^2.
    m = make_measure(distance="discrete", statistic="energy")
    r = m.compute(Stimulus((), (1,)))
    samples = m._get((), 1)
    a_fraction = np.mean([row.tokens[0] == 0 for row in samples.sequences])
    assert r.value == pytest.approx(2*a_fraction*a_fraction)
    assert r.se >= 0 and r.status == "bounded_energy_ci_bootstrap_se"
    assert abs(r.value-1.125) < 4*r.se


def test_exact_energy_matches_monte_carlo_at_large_n(make_measure, sampling):
    """R4: the closed-form path is validated against the Monte Carlo path.

    Both were previously pinned to separately maintained analytic constants, which is the
    divergence R4 exists to prevent. The tolerance is the MC path's own reported error;
    n is capped at 2000 because energy's within-set blocks are quadratic in n.
    """
    item = Stimulus((), (1,))
    kwargs = dict(distance="discrete", statistic="energy")
    exact = make_measure(estimator=EstimatorSpec(method="exact"), **kwargs).compute(item)
    mc = make_measure(sampling=replace(sampling, n=2000, seed=7),
                      estimator=EstimatorSpec(method="mc", bootstrap=199), **kwargs).compute(item)
    assert exact.method == "exact" and mc.method == "mc" and mc.se > 0
    assert abs(exact.value-mc.value) < 4*mc.se


def test_u_energy_small_random_operand_rejected(make_measure, sampling):
    m = make_measure(sampling=replace(sampling, n=1), distance="discrete",
                     statistic=StatisticSpec("energy", {"variant": "u"}))
    with pytest.raises(ConfigError, match=">=2"):
        m.compute(Stimulus((), (1,)))


def test_min_reports_bootstrap_uncertainty(make_measure):
    r = make_measure(distance="discrete", statistic="min").compute(Stimulus((), (1,)))
    assert r.value == 0 and r.se == 0 and r.status == "bootstrap_approximate"


def test_drop_policy_recomputed_for_shorter_horizon():
    backend = ToyBackend(("a", "<eos>"), {(): (0.,1.)}, eos_token_id=1)
    sampling = SamplingSpec(backend.model_spec, 20, 0)
    m = Measure(MeasureSpec(sampling, 3, "drop", estimator=EstimatorSpec(method="mc")),
                sampler=Sampler(backend))
    r = m.compute(Stimulus((), (0,)))
    assert r.status == "all_samples_dropped" and math.isnan(r.value)
    r = m.replace(horizon=1).compute(Stimulus((), (1,)))
    assert r.value == 1 and r.n_a_used == 20


def test_word_distance_and_required_policy():
    backend = ToyBackend(("a ", "b "), {(): (.75,.25)})
    spec = MeasureSpec(SamplingSpec(backend.model_spec, 32, 0), 1, "compare_ragged",
                       unit=UnitSpec("word", "whitespace:v1"), distance="ngram_distance")
    r = Measure(spec, sampler=Sampler(backend)).compute(Stimulus((), (1,)))
    assert 0 <= r.value <= 1
    probability = Measure(spec.replace(distance="prefix_indicator"), sampler=Sampler(backend)).compute(Stimulus((), (1,)))
    assert 0 <= probability.value <= 1 and probability.se is not None
    with pytest.raises(ConfigError, match="exact path"):
        Measure(spec.replace(distance="prefix_indicator", estimator=EstimatorSpec(method="exact")),
                sampler=Sampler(backend))


def test_anticipatory_surprisal_equals_entropy(make_measure):
    m = make_measure(mode="anticipatory", warping="neglog", estimator=EstimatorSpec(method="auto"))
    r = m.compute(Stimulus())
    expected = -.75*math.log(.75)-.25*math.log(.25)
    assert r.value == pytest.approx(expected)
    assert r.se == 0


def test_unsupported_exact_and_entropy_rejected(make_measure):
    with pytest.raises(ConfigError, match="exact path"):
        make_measure(distance="discrete", statistic="min", estimator=EstimatorSpec(method="exact"))
    with pytest.raises(ConfigError, match="no dropping"):
        make_measure(mode="entropy", short_samples="drop")

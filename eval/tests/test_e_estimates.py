"""Category E: SE and n on every score, closed form vs MC, SE scaling, R6 error accounting."""
import math
import numpy as np
import pytest

from uncertain import Stimulus

from conftest import make_measure
from toy_lm import ToyLM, spearman, zscore

P12 = np.linspace(2, 1, 12) ** 3
P12 = P12 / P12.sum()            # 12 distinct probabilities, min ~ 0.024


# --------------------------------------------------------------------------- criterion 20
GRID = [(s, w, H) for s in ("mean", "energy", "min") for w in ("identity", "neglog") for H in (1, 2)]


def _grid_result(statistic, warping, H):
    lm = ToyLM([.4, .3, .2, .1])
    return make_measure(lm, n=128, horizon=H, representation="tokens", distance="discrete",
                        statistic=statistic, warping=warping).compute(Stimulus((0,), (1,)))


@pytest.mark.req("31")
@pytest.mark.parametrize("statistic,warping,H", GRID)
def test_result_carries_required_fields(statistic, warping, H):
    r = _grid_result(statistic, warping, H)
    for attr in ("value", "se", "ci_low", "ci_high", "n", "spec_hash", "diagnostics", "warping_bias"):
        assert hasattr(r, attr)
    assert "truncation_bound" in r.diagnostics and len(r.spec_hash) == 64


@pytest.mark.req("20")
@pytest.mark.parametrize("statistic,warping,H", GRID)
def test_every_score_has_se_and_n(statistic, warping, H, measure):
    r = _grid_result(statistic, warping, H)
    measure("se", r.se); measure("status", r.status)
    assert r.n > 0
    # -log 0 = inf: an infinite score can only carry an infinite (or zero-spread) SE.
    assert isinstance(r.se, float) and (math.isfinite(r.se) or math.isinf(r.value)), \
        f"{statistic}/{warping}/H={H}: se={r.se} ({r.status})"


@pytest.mark.req("20")
def test_zero_hit_surprisal_reports_status_not_se(measure):
    # Observation: with zero hits the prefix-indicator estimate has no SE; record the behaviour.
    lm = ToyLM([.999, .001])
    r = make_measure(lm, n=16, representation="tokens", distance="prefix_indicator", statistic="mean",
                     warping="neglog").compute(Stimulus((0,), (1,)))
    measure("value", r.value); measure("se", r.se); measure("status", r.status)
    assert r.value == math.inf and r.status == "zero_hits" and r.ci_low is not None


# --------------------------------------------------------------------------- criterion 21
@pytest.mark.req("21")
def test_closed_form_vs_mc_surprisal_rank_correlation(measure):
    lm = ToyLM(P12)
    exact, mc = [], []
    for o in range(12):
        stim = Stimulus((0,), (o,))
        kw = dict(representation="tokens", distance="prefix_indicator", statistic="mean", warping="neglog")
        exact.append(make_measure(lm, method="exact", n=8, **kw).compute(stim).value)
        mc.append(make_measure(lm, method="mc", n=1024, seed=o, **kw).compute(stim).value)
    rho = spearman(exact, mc)
    measure("spearman_rho", rho)
    assert np.allclose(exact, -np.log(P12), atol=1e-12)
    assert rho >= 0.95


@pytest.mark.req("21", "21b")
def test_closed_form_vs_mc_converges_and_se_scales(measure):
    lm = ToyLM(P12)
    ns = [32, 128, 512, 2048]
    mae, ses = [], []
    for n in ns:
        errs, se = [], []
        for o in range(12):
            stim = Stimulus((0,), (o,))
            kw = dict(representation="tokens", distance="discrete", statistic="mean")
            closed = make_measure(lm, method="exact", n=8, **kw).compute(stim).value
            r = make_measure(lm, method="mc", n=n, seed=1000 * n + o, **kw).compute(stim)
            errs.append(abs(r.value - closed))
            se.append(r.se)
        mae.append(float(np.mean(errs)))
        ses.append(float(np.mean(se)))
    slope = float(np.polyfit(np.log(ns), np.log(ses), 1)[0])
    measure("mean_abs_err_by_n", dict(zip(ns, mae))); measure("mean_se_by_n", dict(zip(ns, ses)))
    measure("se_loglog_slope", slope)
    assert all(b < a for a, b in zip(mae, mae[1:]))
    assert mae[-1] <= mae[0] / 3
    assert -0.65 <= slope <= -0.35


@pytest.mark.req("21")
@pytest.mark.parametrize("statistic", ["mean", "energy"])
@pytest.mark.parametrize("pair", [("tokens", "discrete"), ("layer:0", "cosine")])
@pytest.mark.parametrize("o", [0, 5, 11])
def test_exact_fast_path_agrees_with_mc(statistic, pair, o, measure):
    lm = ToyLM(P12)
    kw = dict(representation=pair[0], distance=pair[1], statistic=statistic)
    stim = Stimulus((0,), (o,))
    closed = make_measure(lm, method="exact", n=8, **kw).compute(stim)
    n = 512 if pair[1] == "cosine" and statistic == "energy" else 2048   # O(n^2) python cosine calls
    mc = make_measure(lm, method="mc", n=n, seed=o, **kw).compute(stim)
    truth = 1 - P12[o] if statistic == "mean" else 1 - 2 * P12[o] + (P12 ** 2).sum()
    measure("closed", closed.value); measure("mc", mc.value); measure("z", zscore(mc.value, mc.se, closed.value))
    assert closed.method == "exact" and abs(closed.value - truth) <= 1e-12
    assert zscore(mc.value, mc.se, closed.value) <= 4


# --------------------------------------------------------------------------- criterion 27
@pytest.mark.req("27")
def test_truncation_bound_separate_and_shrinks_with_horizon(measure):
    lm = ToyLM([.35, .35, .3], eos_token_id=2)
    bounds = []
    for H in (1, 2, 4, 8):
        r = make_measure(lm, n=256, seed=0, horizon=H, representation="tokens", distance="ngram_distance",
                         statistic="mean").compute(Stimulus((0,), (1,)))
        bounds.append(r.diagnostics["truncation_bound"])
        assert r.se is not None and r.diagnostics["truncation_bound"] is not None
    measure("truncation_bound_by_H", bounds)
    assert all(b <= a for a, b in zip(bounds, bounds[1:])) and bounds[-1] < bounds[0]


@pytest.mark.req("27")
@pytest.mark.parametrize("o", [0, 1, 2])
@pytest.mark.parametrize("distance", ["prefix_indicator", "discrete"])
def test_warping_gap_equals_cv2_over_2n(o, distance, measure):
    lm = ToyLM([.5, .3, .2])
    n = 300
    m = make_measure(lm, n=n, seed=o, representation="tokens", distance=distance, statistic="mean",
                     warping="neglog")
    r = m.compute(Stimulus((0,), (o,)))
    a = m.store.get_or_sample(m.sampler, (0,), 1, m.spec.sampling, m.spec.unit)
    draws = np.array([float(s.tokens[0] == o) if distance == "prefix_indicator" else float(s.tokens[0] != o)
                      for s in a.sequences])
    cv2 = draws.var(ddof=1) / draws.mean() ** 2
    measure("warping_bias", r.warping_bias); measure("cv2_over_2n", cv2 / (2 * n))
    assert abs(r.warping_bias - cv2 / (2 * n)) <= 1e-9
    corrected = make_measure(lm, n=n, seed=o, representation="tokens", distance=distance, statistic="mean",
                             warping="neglog", method="mc")
    from uncertain import EstimatorSpec, Measure
    c = Measure(corrected.spec.replace(estimator=EstimatorSpec(method="mc", bootstrap=99, correct_bias=True)),
                sampler=corrected.sampler).compute(Stimulus((0,), (o,)))
    assert abs(c.corrected_value - (c.value - c.warping_bias)) <= 1e-12

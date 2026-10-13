"""Integration on pretrained GPT-2 (local cache, pinned revision, CPU)."""
import math
from functools import lru_cache
import numpy as np
import pytest

from uncertain import EstimatorSpec, Measure, MeasureSpec, Sampler, SamplingSpec, Stimulus, UnitSpec

from toy_lm import spearman

pytestmark = pytest.mark.gpt2

SENTENCES = [
    "The cat sat on the mat.", "She opened the door slowly.", "We walked to the old bridge.",
    "He forgot his keys again.", "The train arrived very late.", "They planted trees in spring.",
    "My brother likes cold coffee.", "The river flows to the sea.", "I read a book last night.",
    "The children laughed at the clown.", "Rain fell on the quiet town.", "She wrote a long letter.",
    "The dog barked at the mailman.", "We ate soup for dinner.", "The sun rose over the hills.",
    "He painted the fence blue.", "The baby slept all afternoon.", "Birds sang in the garden.",
    "Edge case: non-ASCII -- the café in Zürich was naïve.", "Edge case: non-ASCII -- 東京 is a big city.",
]
FOLLOW = " It was a quiet day."


def last_word_split(sentence):
    body = sentence.rstrip(".")
    head, _, last = body.rpartition(" ")
    return head, " " + last


def spec(backend, unit, H, n, budget, **kw):
    sampling = SamplingSpec(backend.model_spec, n=n, seed=0, token_budget=budget)
    return MeasureSpec(sampling, H, "compare_ragged", unit=unit,
                       estimator=EstimatorSpec(method="mc", bootstrap=29), **kw)


CONFIGS = {
    "token": dict(H=2, n=8, budget=8, representation="layer:6", distance="cosine", statistic="energy"),
    "word": dict(H=1, n=8, budget=8, representation="tokens", distance="discrete", statistic="mean"),
    "sentence": dict(H=1, n=4, budget=20, representation="tokens", distance="discrete", statistic="mean"),
}


@pytest.mark.req("11")
@pytest.mark.parametrize("unit", ["token", "word", "sentence"])
@pytest.mark.parametrize("i", range(len(SENTENCES)))
def test_pipeline_runs_on_gpt2(gpt2, unit, i, measure):
    cfg = dict(CONFIGS[unit])
    H, n, budget = cfg.pop("H"), cfg.pop("n"), cfg.pop("budget")
    if unit == "sentence":
        context, target = SENTENCES[i], FOLLOW
    else:
        context, target = last_word_split(SENTENCES[i])
    stim = Stimulus.from_text(context, target, gpt2)
    if unit == "token":
        H = max(H, len(stim.target))
    m = Measure(spec(gpt2, unit, H, n, budget, **cfg), sampler=Sampler(gpt2))
    r = m.compute(stim)
    measure("value", r.value); measure("se", r.se); measure("status", r.status)
    measure("effective_horizons", list(r.diagnostics["effective_horizons"]))
    assert math.isfinite(r.value)
    assert r.n == n and r.se is not None and math.isfinite(r.se)
    assert len(r.diagnostics["effective_horizons"]) == n


@pytest.mark.req("21")
def test_gpt2_closed_form_vs_mc_surprisal(gpt2, monkeypatch, measure):
    cached = lru_cache(maxsize=None)(gpt2.probabilities)
    monkeypatch.setattr(gpt2, "probabilities", cached)   # H=1 MC asks for the same p n times
    items = []
    for s in SENTENCES[:18]:                                     # every word position, first token
        words = s.rstrip(".").split(" ")
        for k in range(2, len(words)):
            stim = Stimulus.from_text(" ".join(words[:k]), " " + words[k], gpt2)
            items.append(Stimulus(stim.context, stim.target[:1]))
    kw = dict(representation="tokens", distance="prefix_indicator", statistic="mean", warping="neglog")
    probe = [math.exp(-Measure(spec(gpt2, "token", 1, 8, 1, **kw).replace(
        estimator=EstimatorSpec(method="exact")), sampler=Sampler(gpt2)).compute(it).value) for it in items]
    keep = [it for it, p in zip(items, probe) if p > 0.02][:20]  # selection uses the closed form only
    exact, mc = [], []
    for it in keep:
        exact.append(Measure(spec(gpt2, "token", 1, 8, 1, **kw).replace(
            estimator=EstimatorSpec(method="exact")), sampler=Sampler(gpt2)).compute(it).value)
        mc.append(Measure(spec(gpt2, "token", 1, 1024, 1, **kw), sampler=Sampler(gpt2)).compute(it).value)
    rho = spearman(exact, mc)
    measure("items", len(keep)); measure("spearman_rho", rho)
    measure("mean_abs_diff_nats", float(np.mean(np.abs(np.array(exact) - np.array(mc)))))
    assert len(keep) >= 15 and rho >= 0.95


@pytest.mark.req("18")
def test_gpt2_logprobs_match_teacher_forced(gpt2, measure):
    worst = 0.
    for s in SENTENCES[:3]:
        context = gpt2.encode(s)
        a = Sampler(gpt2).sample(context, 4, SamplingSpec(gpt2.model_spec, n=2, seed=0, token_budget=4),
                                 UnitSpec())
        for row in a.sequences:
            worst = max(worst, float(np.abs(np.array(row.model_logprobs) - gpt2.score(context, row.tokens)).max()))
    measure("max_abs_err", worst)
    assert worst <= 1e-4

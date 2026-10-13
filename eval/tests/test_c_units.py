"""Category C: units, horizons, short samples, truncation, prefix consistency."""
import math
import pytest
from hypothesis import given, settings, strategies as st

from uncertain import (AlternativeStore, ConfigError, EstimatorSpec, Measure, MeasureSpec, Sampler,
                       SamplingSpec, Stimulus, Sweep, UnitSpec, positions)
from uncertain.alignment import AlignmentError

from conftest import make_measure
from toy_lm import ToyLM

UNITS = {"token": UnitSpec(), "word": UnitSpec("word", "whitespace:v1"),
         "sentence": UnitSpec("sentence", "punctuation:v1")}
# "d e" is one token that spans a word boundary; "é"/"日本" exercise non-ASCII text.
PIECES = ("ab", "c", " ", "d e", ".", "é", "日本", "<eos>")
P = [.2, .15, .25, .08, .12, .07, .05, .08]
EOS = 7


def word_lm(p=P, eos=EOS):
    return ToyLM(p, pieces=PIECES, eos_token_id=eos)


def sample(lm, unit, H, n, seed, budget=24):
    spec = SamplingSpec(lm.model_spec, n=n, seed=seed, token_budget=budget)
    return Sampler(lm).sample((0,), H, spec, UNITS[unit])


# --------------------------------------------------------------------------- criterion 10
@pytest.mark.req("10")
def test_one_unit_per_run(measure):
    lm = word_lm()
    m = make_measure(lm, n=4, representation="tokens", distance="discrete", statistic="mean")
    with pytest.raises(ConfigError) as e1:
        Sweep(m, unit=["token", "word"])
    with pytest.raises(ConfigError) as e2:
        Sweep(m, cells=[{"horizon": 1}, {"unit": "word"}])
    with pytest.raises(ConfigError) as e3:
        make_measure(lm, n=4, unit="morpheme")
    for i, e in enumerate((e1, e2, e3)):
        measure(f"msg{i}", str(e.value))
        assert "unit" in str(e.value).lower()


# --------------------------------------------------------------------------- criterion 11 / 12
@pytest.mark.req("11", "12")
@pytest.mark.parametrize("unit,H", [("word", 1), ("word", 3), ("sentence", 1), ("sentence", 2)])
def test_unit_boundary_stopping(unit, H, measure):
    lm = word_lm()
    budget = 24
    a = sample(lm, unit, H, 200, seed=H, budget=budget)
    reasons = {}
    for s in a.sequences:
        reasons[s.stop_reason] = reasons.get(s.stop_reason, 0) + 1
        assert s.stop_reason in {"horizon", "eos", "budget"}
        if s.stop_reason == "horizon":
            assert s.effective_horizon == H
        else:
            assert s.effective_horizon < H                              # criterion 12
        if s.stop_reason == "eos":
            assert s.tokens[-1] == EOS
        if s.stop_reason == "budget":
            assert len(s.tokens) == budget
        assert s.effective_horizon == len(s.units) == len(s.unit_spans)
        for (ca, cb), u in zip(s.char_spans, s.units):
            assert s.text[ca:cb] == u
        for (ta, tb), u in zip(s.unit_spans, s.units):
            assert u.strip() and u.strip()[0] in lm.decode(s.tokens[ta:tb])
    by_units = {}
    for s in a.sequences:
        by_units.setdefault(len(s.units), set()).add(len(s.tokens))
    measure("stop_reasons", reasons)
    assert any(len(v) > 1 for v in by_units.values())                  # token-ragged, unit-aligned


@pytest.mark.req("12")
def test_effective_horizon_reported_in_result():
    lm = word_lm()
    m = make_measure(lm, n=64, horizon=3, unit="word", representation="tokens", distance="discrete",
                     statistic="mean")
    r = m.compute(Stimulus((0,), (1, 2)))
    a = m.store.get_or_sample(m.sampler, (0,), 3, m.spec.sampling, m.spec.unit)
    assert list(r.diagnostics["effective_horizons"]) == [s.effective_horizon for s in a.sequences]
    assert any(h < 3 for h in r.diagnostics["effective_horizons"])


@pytest.mark.req("11", "12")
@pytest.mark.parametrize("policy", ["pad", "drop", "compare_ragged"])
@pytest.mark.parametrize("unit", ["token", "word"])
def test_edge_every_alternative_hits_eos_immediately(policy, unit, measure):
    # Edge case: a context where every alternative hits EOS immediately.
    lm = ToyLM([0, 0, 0, 0, 0, 0, 0, 1.], pieces=PIECES, eos_token_id=EOS)
    m = make_measure(lm, n=16, horizon=2, unit=unit, short_samples=policy,
                     representation="tokens", distance="discrete", statistic="mean")
    r = m.compute(Stimulus((0,), (0, 2) if unit == "word" else (0,)))
    measure("value", r.value); measure("status", r.status)
    assert r.diagnostics["eos_fraction"] == 1.0
    if policy == "drop":
        assert math.isnan(r.value) and r.status == "all_samples_dropped" and r.n_a_used == 0
    else:
        assert math.isfinite(r.value)


@pytest.mark.req("3", "12")
@pytest.mark.parametrize("H", [2, 4, 6])
def test_edge_horizon_longer_than_remaining_text(H):
    # Edge case: H larger than the remaining text -- last position of a 3-token document,
    # and a target that is itself EOS (nothing can follow).
    lm = word_lm()
    stims = positions((0, 1, EOS), context=(2,))
    m = make_measure(lm, n=32, horizon=H, representation="tokens", distance="discrete", statistic="mean")
    last = m.compute(stims[-1])
    assert math.isfinite(last.value) and last.n_b_used == 1
    assert math.isfinite(m.compute(stims[1]).value)


@pytest.mark.req("10", "11")
@pytest.mark.parametrize("target", [(), (2,), (2, 2)])
def test_edge_empty_or_whitespace_target(target, measure):
    # Edge case: empty or whitespace-only input -- must be a ConfigError before sampling.
    lm = word_lm()
    m = make_measure(lm, n=8, unit="word", representation="tokens", distance="discrete", statistic="mean")
    with pytest.raises(ConfigError) as info:
        m.compute(Stimulus((0,), target))
    measure("message", str(info.value))
    assert lm.calls == 0


@pytest.mark.req("11")
@pytest.mark.parametrize("unit", ["word", "sentence"])
def test_edge_non_ascii_units(unit):
    # Edge case: non-ASCII text in word/sentence units.
    lm = ToyLM([.3, .1, .2, .1, .2, .1], pieces=("é", "日本", " ", "ü", ".", "<eos>"), eos_token_id=5)
    a = sample(lm, unit, 2, 64, 0)
    assert any(any(ord(ch) > 127 for ch in u) for s in a.sequences for u in s.units)
    m = make_measure(lm, n=16, horizon=2, unit=unit, representation="tokens", distance="discrete",
                     statistic="mean")
    assert math.isfinite(m.compute(Stimulus((2,), (0, 4, 2) if unit == "sentence" else (1, 2))).value)


class ByteToyLM(ToyLM):
    """Byte-level pieces like GPT-2: "東" is split over two tokens (e6 9d | b1)."""
    BYTES = ("東".encode()[:2], "東".encode()[2:], b" ", b"a", b".")

    def __init__(self):
        super().__init__([.2] * 5, pieces=("<e69d>", "<b1>", " ", "a", "."), name="eval-bytes")

    def decode(self, tokens):
        return b"".join(self.BYTES[t] for t in tokens).decode("utf-8", errors="replace")


@pytest.mark.req("11")
@pytest.mark.parametrize("unit", ["word", "sentence"])
def test_edge_multibyte_char_split_across_tokens(unit, measure):
    # Edge case: non-ASCII character split across byte-level tokens (as GPT-2 does for CJK/emoji).
    from uncertain.alignment import align
    lm = ByteToyLM()
    assert lm.decode((0, 1)) == "東"
    try:
        units = align((3, 2, 0, 1, 4, 2), lm, UNITS[unit], final=True)[0]
    except AlignmentError as exc:
        measure("error", str(exc))
        raise
    assert "a" in units[0]


@pytest.mark.req("11", "14")
def test_edge_token_spanning_word_boundary():
    # Edge case: one token ("d e") spans a word boundary -> both words are owned by that token.
    lm = word_lm()
    a = sample(lm, "word", 3, 300, 4)
    spanning = [s for s in a.sequences if 3 in s.tokens and len(s.units) >= 2]
    assert spanning
    for s in spanning:
        for i, ((ta, tb), (ca, cb)) in enumerate(zip(s.unit_spans, s.char_spans)):
            assert s.text[ca:cb] == s.units[i]


# --------------------------------------------------------------------------- criterion 13
@pytest.mark.req("13")
def test_short_sample_policy_required_and_hashed():
    lm = word_lm()
    sampling = SamplingSpec(lm.model_spec, n=4, seed=0)
    with pytest.raises(TypeError):
        MeasureSpec(sampling, 1)  # noqa
    with pytest.raises(TypeError):
        Measure.from_config(lm, n=4, sampler=Sampler(lm))
    with pytest.raises(ConfigError):
        MeasureSpec(sampling, 1, "truncate")
    specs = [MeasureSpec(sampling, 2, p, distance="discrete") for p in ("pad", "drop", "compare_ragged")]
    assert len({s.spec_hash for s in specs}) == 3
    ms = [Measure(s, sampler=Sampler(lm)) for s in specs]
    assert len({m.spec_hash for m in ms}) == 3


@pytest.mark.req("13")
def test_short_sample_policies_differ_when_samples_are_short(measure):
    # §9: "each short-sample policy moves the result in the documented direction" -- no
    # direction is documented in the PDF or the repo docs; record values (AMBIGUOUS).
    lm = ToyLM([.3, .2, .1, .4], eos_token_id=3)
    values = {}
    for policy in ("pad", "drop", "compare_ragged"):
        m = make_measure(lm, n=512, seed=1, horizon=3, short_samples=policy, representation="tokens",
                         distance="ngram_distance", statistic="mean")
        r = m.compute(Stimulus((0,), (1,)))
        values[policy] = (r.value, r.se, r.n_a_used)
        measure(policy, values[policy])
    assert len({v[0] for v in values.values()}) == 3
    assert values["drop"][2] < 512


# --------------------------------------------------------------------------- criterion 14
@pytest.mark.req("14")
@settings(max_examples=60, deadline=None)
@given(unit=st.sampled_from(["token", "word", "sentence"]), hmax=st.integers(2, 4),
       data=st.data(), seed=st.integers(0, 10**6))
def test_truncate_equals_direct_sample(unit, hmax, data, seed):
    H = data.draw(st.integers(1, hmax - 1))
    lm = word_lm()
    long = sample(lm, unit, hmax, 24, seed, budget=40)
    short = sample(lm, unit, H, 24, seed, budget=40)
    cut = long.truncate(H)
    assert cut.horizon == H
    for x, y in zip(cut.sequences, short.sequences):
        assert x.tokens == y.tokens and x.units == y.units and x.unit_spans == y.unit_spans
        assert x.model_logprobs == y.model_logprobs and x.stop_reason == y.stop_reason


@pytest.mark.req("14")
@settings(max_examples=40, deadline=None)
@given(unit=st.sampled_from(["token", "word", "sentence"]), hmax=st.integers(2, 4),
       data=st.data(), seed=st.integers(0, 10**6))
def test_truncate_equals_direct_sample_without_spanning_tokens(unit, hmax, data, seed):
    # Same property on a vocabulary with no boundary-spanning token, so a failure here is not
    # masked by the known spanning-token defect (keeps the truncation mutation detectable).
    H = data.draw(st.integers(1, hmax - 1))
    lm = ToyLM([.3, .2, .25, .15, .1], pieces=("ab", "c", " ", ".", "<eos>"), eos_token_id=4)
    long, short = sample(lm, unit, hmax, 24, seed, budget=40), sample(lm, unit, H, 24, seed, budget=40)
    for x, y in zip(long.truncate(H).sequences, short.sequences):
        assert x.tokens == y.tokens and x.units == y.units and x.unit_spans == y.unit_spans


@pytest.mark.req("14")
def test_truncate_when_last_token_spans_units_minimal():
    # Edge case (minimal repro): tokens ("ab", "d e") decode to words "abd", "e"; the last token
    # owns both units. truncate(1) must keep one unit, as sample(H=1) does.
    from uncertain import Sequence, AlternativeSet
    seq = Sequence((0, 3), (-.1, -.1), (-.1, -.1), ("abd", "e"), ((0, 2), (1, 2)),
                   ((0, 3), (4, 5)), text="abd e", stop_reason="horizon")
    assert seq.truncate(1).units == ("abd",)
    AlternativeSet((seq,), (), UNITS["word"], 2).truncate(1)


@pytest.mark.req("14", "24")
def test_horizon_sweep_reuses_one_sampling_pass(measure):
    calls = {}
    for label, horizons in (("H4_only", [4]), ("sweep_1_2_4", [1, 2, 4])):
        lm = word_lm()
        base = make_measure(lm, n=64, horizon=4, representation="tokens", distance="discrete",
                            statistic="mean")
        Sweep(base, horizon=horizons, statistic=["mean", "energy"]).run(positions((0, 1, 4), context=(2,)))
        calls[label] = lm.calls
        measure(f"backend_calls_{label}", lm.calls)
    assert calls["sweep_1_2_4"] == calls["H4_only"]
    lm = word_lm()
    store = AlternativeStore()
    m4 = make_measure(lm, n=64, horizon=4, store=store, representation="tokens", distance="discrete",
                      statistic="mean")
    m4.compute(Stimulus((2,), (0,)))
    before = lm.calls
    for H in (1, 2, 3):
        m4.replace(horizon=H).compute(Stimulus((2,), (0,)))
    measure("extra_calls_after_H4", lm.calls - before)
    assert lm.calls == before


# --------------------------------------------------------------------------- criterion 28
class NonPrefixSampler(Sampler):
    prefix_consistent = False


@pytest.mark.req("28")
def test_prefix_inconsistent_sampler_resamples(measure):
    lm = word_lm()
    sampler = NonPrefixSampler(lm)
    a = sampler.sample((0,), 3, SamplingSpec(lm.model_spec, n=8, seed=0), UnitSpec())
    with pytest.raises(ValueError):
        a.truncate(2)
    base = make_measure(lm, n=16, horizon=2, sampler=sampler, representation="tokens",
                        distance="discrete", statistic="mean")
    before = sampler.calls
    rows = Sweep(base, horizon=[1, 2]).run([Stimulus((0,), (1,))]).rows
    measure("sampling_passes", sampler.calls - before)
    assert sampler.calls - before == 3     # contexts: (0,)@H1, (0,)@H2, (0,1)@H1 -- no truncation shortcut
    assert all(math.isfinite(r["value"]) for r in rows)

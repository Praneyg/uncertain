from dataclasses import replace
import math
import numpy as np
import pytest
from uncertain import ToyBackend, Sampler, SamplingSpec, UnitSpec, ConfigError, Stimulus
from uncertain.samplers import sampling_probabilities
from uncertain.alignment import boundaries, AlignmentError


def test_indexed_extension_matches_whole(backend, sampling):
    sampler = Sampler(backend)
    first = sampler.sample((), 4, replace(sampling, n=11))
    extended = sampler.extend(first, 8)
    whole = sampler.sample((), 4, replace(sampling, n=19))
    assert extended.content_hash == whole.content_hash
    assert extended.sequences[:11] == first.sequences


def test_prefix_pathwise_identity(backend, sampling):
    sampler = Sampler(backend)
    long = sampler.sample((0,), 5, sampling)
    short = sampler.sample((0,), 2, sampling)
    assert long.truncate(2).sequences == short.sequences


def test_deterministic_sampling_preserves_multiplicity():
    backend = ToyBackend(transitions={(): (1., 0.)})
    a = Sampler(backend).sample((), 2, SamplingSpec(backend.model_spec, 8, 0))
    assert len(a) == 8
    assert len({s.tokens for s in a.sequences}) == 1
    assert len({s.sample_id for s in a.sequences}) == 8


def test_recorded_q_and_p_are_distinct():
    backend = ToyBackend(transitions={(): (.8, .2)})
    spec = SamplingSpec(backend.model_spec, 12, 0, generation={"top_k": 1})
    a = Sampler(backend).sample((), 3, spec)
    for row in a.sequences:
        assert row.tokens == (0, 0, 0)
        assert row.sampling_logprobs == (0., 0., 0.)
        np.testing.assert_allclose(row.model_logprobs, [math.log(.8)]*3)


def test_temperature_and_nucleus():
    spec = SamplingSpec(ToyBackend().model_spec, 1, 0, generation={"temperature": 2., "top_p": .75})
    p = np.array([.64, .25, .11])
    q = sampling_probabilities(p, spec.generation)
    expected = np.sqrt(p)
    expected[2] = 0
    expected /= expected.sum()
    np.testing.assert_allclose(q, expected)


def test_q_mismatch_fails_before_sampling(make_measure, sampling):
    with pytest.raises(ConfigError, match="follow q"):
        make_measure(sampling=replace(sampling, generation={"top_k": 1}))


def test_eos_and_token_budget():
    backend = ToyBackend(("a", "<eos>"), {(): (0., 1.)}, eos_token_id=1)
    a = Sampler(backend).sample((), 4, SamplingSpec(backend.model_spec, 3, 0))
    assert all(s.stop_reason == "eos" and len(s.units) == 1 for s in a.sequences)
    backend = ToyBackend()
    a = Sampler(backend).sample((), 4, SamplingSpec(backend.model_spec, 3, 0, token_budget=2))
    assert all(s.stop_reason == "budget" and len(s.units) == 2 for s in a.sequences)


def test_word_boundaries_and_lookahead():
    backend = ToyBackend(("hello", " ", "world"),
                         {(): (1.,0.,0.), (0,): (0.,1.,0.), (1,): (0.,0.,1.), (2,): (0.,1.,0.)})
    a = Sampler(backend).sample((), 2, SamplingSpec(backend.model_spec, 2, 0),
                               UnitSpec("word", "whitespace:v1"))
    assert a.sequences[0].units == ("hello", "world")
    assert a.sequences[0].unit_spans == ((0,1), (2,3))
    assert a.sequences[0].char_spans == ((0,5), (6,11))
    # The 4th token was generated to confirm "world" but is lookahead past H=2, so the
    # stored sequence stops at the span end, as truncate() would.
    assert len(a.sequences[0].tokens) == 3
    assert a.truncate(1).sequences[0].units == ("hello",)


def test_partial_word_budget_is_not_completed():
    backend = ToyBackend(("word",), {(): (1.,)})
    a = Sampler(backend).sample((), 1, SamplingSpec(backend.model_spec, 2, 0, token_budget=1),
                               UnitSpec("word", "whitespace:v1"))
    assert a.sequences[0].effective_horizon == 0
    assert a.sequences[0].stop_reason == "budget"


def test_sentence_and_unicode_detector():
    unit = UnitSpec("sentence", "punctuation:v1")
    text = "  Hi! नमस्ते? Final"
    assert tuple(text[a:b] for a,b in boundaries(text, unit)) == ("Hi!", "नमस्ते?")
    assert tuple(text[a:b] for a,b in boundaries(text, unit, final=True))[-1] == "Final"


def test_retokenization_is_rejected():
    backend = ToyBackend(("a", "b", "ab"), {(): (.3,.3,.4)})
    with pytest.raises(AlignmentError, match="retokenizes"):
        Stimulus.from_text("a", "b", backend)


def test_prefix_pathwise_identity_on_text_units():
    # A word is recognised only once the next word's first token arrives, so sample()
    # must drop that lookahead or truncate() cannot reproduce what sample() returns.
    # Sentence pieces carry their own trailing space: `boundaries` closes a sentence on a
    # terminator at end of text, so sentence units need no lookahead, but the trailing
    # space still puts the char span short of the token span.
    for vocabulary, unit in ((("x", " y", " z"), UnitSpec("word", "whitespace:v1")),
                             (("x. ", "y. ", "z. "), UnitSpec("sentence", "punctuation:v1")),
                             (("x.", " y.", " z."), UnitSpec("sentence", "punctuation:v1"))):
        backend = ToyBackend(vocabulary, {(): (.5, .3, .2)})
        sampler = Sampler(backend)
        spec = SamplingSpec(backend.model_spec, 16, 3)
        long = sampler.sample((0,), 4, spec, unit)
        short = sampler.sample((0,), 2, spec, unit)
        assert long.truncate(2).sequences == short.sequences
        assert all(len(s.tokens) == s.unit_spans[-1][1] for s in short.sequences)


def test_eos_completing_the_horizon_trims_to_the_horizon():
    # final=True can complete unit H and beyond on the EOS step; the stored sequence then
    # holds H units and no longer ends at EOS, so it reports "horizon", as truncate() does.
    backend = ToyBackend(("x", " y", "<eos>"),
                         {(): (1.,0.,0.), (0,): (0.,1.,0.), (1,): (0.,0.,1.)}, eos_token_id=2)
    unit = UnitSpec("word", "whitespace:v1")
    spec = SamplingSpec(backend.model_spec, 1, 0)
    sampler = Sampler(backend)
    short = sampler.sample((), 2, spec, unit)
    assert short.sequences[0].tokens == (0, 1)
    assert short.sequences[0].units == ("x", "y") and short.sequences[0].stop_reason == "horizon"
    long = sampler.sample((), 3, spec, unit)
    assert long.sequences[0].tokens == (0, 1, 2)
    assert long.sequences[0].stop_reason == "eos" and long.sequences[0].effective_horizon == 2
    assert long.truncate(2).sequences == short.sequences


def test_unterminated_count_is_recorded():
    backend = ToyBackend(("a", "<eos>"), {(): (.5, .5)}, eos_token_id=1)
    spec = SamplingSpec(backend.model_spec, 32, 5)
    sampler = Sampler(backend)
    a = sampler.sample((), 4, spec)
    assert a.provenance["unterminated"] == sum(s.effective_horizon < 4 for s in a.sequences) > 0
    assert (sampler.extend(a, 8).content_hash
            == sampler.sample((), 4, replace(spec, n=40)).content_hash)


def test_sentence_terminator_at_end_of_text_is_unconfirmed():
    # "z?" may still become "z?x." once the next token arrives, so a non-final
    # detector that emits it revises the unit itself as generation continues and
    # truncate(sample(H_max), H) can never reproduce sample(H). The word detector
    # already drops its unterminated tail; the sentence detector now matches it.
    unit = UnitSpec("sentence", "punctuation:v1")
    assert boundaries("a. b.", unit) == boundaries("a. b.x.", unit)
    assert tuple("a. b."[a:b] for a,b in boundaries("a. b.", unit)) == ("a.",)
    assert tuple("a. b."[a:b] for a,b in boundaries("a. b.", unit, final=True)) == ("a.", "b.")
    # A terminator confirmed by following whitespace is still closed without lookahead.
    assert tuple("a. b. "[a:b] for a,b in boundaries("a. b. ", unit)) == ("a.", "b.")


def test_registered_unit_produces_units_without_core_edits():
    # §7: a new unit is "implement boundaries(tokens), declare a detector id, register".
    # Configuring one is not enough - it must also drive alignment, so the detector is
    # resolved through the registry rather than an if-chain over known unit names.
    import re
    from uncertain.registry import register_unit
    from uncertain.alignment import align

    def comma_boundaries(text, *, final=False):
        return tuple((m.start(), m.end()) for m in re.finditer(r"[^,]+", text)
                     if final or m.end() < len(text))

    @register_unit("test_comma_clause")
    class CommaClause:
        boundary = "comma:v1"
        detect = staticmethod(comma_boundaries)
        descriptor = {"unit": "test_comma_clause", "boundary": "comma:v1", "version": "1"}

    unit = UnitSpec("test_comma_clause", "comma:v1")
    backend = ToyBackend(("a", ",b", ",c"), {(): (.4, .4, .2)})
    units, spans, chars, text = align(backend.encode("a,b,c"), backend, unit, final=True)
    assert units == ("a", "b", "c")

    sampler = Sampler(backend)
    drawn = sampler.sample((0,), 2, SamplingSpec(backend.model_spec, 8, 5), unit)
    assert all(len(s.units) <= 2 for s in drawn.sequences)
    assert any(len(s.units) == 2 for s in drawn.sequences)
    assert drawn.truncate(1).sequences == sampler.sample(
        (0,), 1, SamplingSpec(backend.model_spec, 8, 5), unit).sequences


def test_deepen_matches_fresh_sample_and_store_resumes():
    from uncertain.store import AlternativeStore
    cases = ((ToyBackend(), UnitSpec()),
             (ToyBackend(("x", " y", " z"), {(): (.5, .3, .2)}), UnitSpec("word", "whitespace:v1")),
             (ToyBackend(("x. ", "y. ", "z. "), {(): (.5, .3, .2)}), UnitSpec("sentence", "punctuation:v1")),
             (ToyBackend(("x", " y", "<eos>"), {(): (.6, .3, .1)}, eos_token_id=2),
              UnitSpec("word", "whitespace:v1")))
    for backend, unit in cases:
        sampler = Sampler(backend)
        spec = SamplingSpec(backend.model_spec, 32, 5)
        short = sampler.sample((0,), 2, spec, unit)
        assert sampler.deepen(short, 5).content_hash == sampler.sample((0,), 5, spec, unit).content_hash
        store = AlternativeStore()
        store.get_or_sample(sampler, (0,), 2, spec, unit)
        before = sampler.calls
        deep = store.get_or_sample(sampler, (0,), 5, spec, unit)
        assert sampler.calls == before+1 and deep.horizon == 5
        assert deep.truncate(2).sequences == short.sequences


def test_one_model_call_per_distinct_context(backend, sampling):
    seen, probabilities = [], backend.probabilities
    backend.probabilities = lambda context: (seen.append(tuple(context)), probabilities(context))[1]
    sampler = Sampler(backend)
    a = sampler.sample((0,), 3, replace(sampling, n=50))
    assert len(seen) == len(set(seen)) and seen[0] == (0,)
    assert a.content_hash == Sampler(ToyBackend()).sample((0,), 3, replace(sampling, n=50)).content_hash

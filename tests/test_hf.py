from dataclasses import replace
import numpy as np
import pytest
from uncertain import HFBackend, Sampler, SamplingSpec, MeasureSpec, Measure, Stimulus, EncoderSpec


@pytest.fixture(scope="module")
def hf():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1)
    torch.manual_seed(52)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=9, n_positions=32, n_embd=16,
                                     n_layer=2, n_head=2, bos_token_id=0, eos_token_id=8,
                                     attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.))
    return HFBackend(model=model, bos_token_id=0, eos_token_id=8)


@pytest.mark.hf
def test_teacher_forced_logprob_oracle_and_mutation(hf):
    spec = SamplingSpec(hf.model_spec, 7, 42)
    a = Sampler(hf).sample((0,1), 3, spec)

    def oracle(table):
        """The §9 assertion itself, applied to whichever table it is handed."""
        for row, scored in zip(a.sequences, table, strict=True):
            np.testing.assert_allclose(scored[:len(row.tokens)],
                                       hf.score((0,1), row.tokens), atol=2e-7)

    for row in a.sequences:
        np.testing.assert_allclose(row.model_logprobs, hf.score((0,1), row.tokens), atol=2e-7)
    table = a.arrays()["model_logprobs"]
    oracle(table)
    # Deliberate historical bug shape: reinterpreting step/sample order without transposing
    # back. The negative control is that the oracle *rejects* it, not merely that it differs.
    bad = table.T.flatten().reshape(table.shape)  # the bug itself; src/ may not do this
    assert not np.allclose(table, bad, equal_nan=True)
    with pytest.raises(AssertionError):
        oracle(bad)


@pytest.mark.hf
def test_processed_probability_oracle(hf):
    torch = pytest.importorskip("torch")
    spec = SamplingSpec(hf.model_spec, 4, 3, generation={"temperature": .7, "top_k": 3})
    a = Sampler(hf).sample((0,), 3, spec)
    for row in a.sequences:
        for i, token in enumerate(row.tokens):
            with torch.inference_mode():
                logits = hf._model(torch.tensor([(0,)+row.tokens[:i]])).logits[0,-1].double()/.7
                retained = torch.topk(logits, 3).indices
                filtered = torch.full_like(logits, -torch.inf)
                filtered[retained] = logits[retained]
                expected = torch.log_softmax(filtered, dim=-1)[token].item()
            assert row.sampling_logprobs[i] == pytest.approx(expected, abs=3e-7)


@pytest.mark.hf
def test_hf_repeatability_extension_and_layer_encoding(hf):
    sampler = Sampler(hf)
    sampling = SamplingSpec(hf.model_spec, 4, 8)
    a = sampler.sample((0,1), 2, sampling)
    assert a.content_hash == sampler.sample((0,1), 2, sampling).content_hash
    assert sampler.extend(a, 2).content_hash == sampler.sample((0,1), 2, replace(sampling, n=6)).content_hash
    spec = MeasureSpec(sampling, 1, "compare_ragged", representation=EncoderSpec("layer", {"layer": 1}),
                       distance="cosine", statistic="mean")
    r = Measure(spec, sampler=sampler).compute(Stimulus((0,), (1,)))
    assert 0 <= r.value <= 2 and r.se is not None


@pytest.mark.hf
def test_hf_exact_entropy_and_logprob(hf):
    from uncertain import EstimatorSpec
    s = SamplingSpec(hf.model_spec, 8, 0)
    spec = MeasureSpec(s, 1, "compare_ragged", estimator=EstimatorSpec(method="exact"), warping="neglog")
    r = Measure(spec, sampler=Sampler(hf)).compute(Stimulus((0,), (1,)))
    assert r.value == pytest.approx(-hf.score((0,), (1,))[0])


# §9 oracle 1, second half: LM surprisal against the Monte Carlo estimate across items.
# Targets were chosen once for a wide surprisal spread (0.1-3.8 nats) that n=512 can
# resolve; at p below ~20/n the estimator returns a zero-hit infinity and no correlation
# is definable at any feasible n.
GPT2_SURPRISAL_ITEMS = [
    ("The quick brown fox jumps over the lazy", " dog"),
    ("The capital of France is", " Paris"),
    ("The quick brown fox jumps", " over"),
    ("She opened the door and walked into the", " room"),
    ("The capital", " of"),
    ("In the beginning God", " created"),
    ("She opened the", " door"),
    ("The capital of France is Paris and the capital of Germany is", " Berlin"),
    ("The capital of France is Paris and the capital of Germany", " is"),
    ("In the beginning God created the heaven and", " the"),
]


@pytest.mark.gpt2
@pytest.mark.slow
def test_mc_surprisal_correlates_with_lm_surprisal(gpt2):
    from uncertain import EstimatorSpec
    spec = MeasureSpec(SamplingSpec(gpt2.model_spec, 512, 20260921), 1, "compare_ragged",
                       warping="neglog", estimator=EstimatorSpec(method="mc", bootstrap=99))
    m = Measure(spec, sampler=Sampler(gpt2))
    lm, mc = [], []
    for context, target in GPT2_SURPRISAL_ITEMS:
        item = Stimulus.from_text(context, target, gpt2)
        r = m.compute(item)
        teacher_forced = -float(gpt2.score(item.context, item.target).sum())
        assert r.se is not None and abs(r.value-teacher_forced) < 4*r.se
        lm.append(teacher_forced)
        mc.append(r.value)
    # Measured 0.9975 at this n and seed; pinned at 0.99 for float and platform drift.
    assert float(np.corrcoef(lm, mc)[0,1]) >= .99

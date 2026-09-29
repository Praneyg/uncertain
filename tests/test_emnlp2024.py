from dataclasses import replace
import math
import numpy as np
import pytest

from uncertain import HFBackend, SamplingSpec
from uncertain.emnlp2024 import (GPT2StudyEncoder, StudySampler, cosine_u,
                                 estimate_study, normalize, pairwise_estimate)


@pytest.fixture(scope="module")
def study_backend():
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1)
    torch.manual_seed(771)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=32, n_embd=16,
                                     n_layer=2, n_head=2, bos_token_id=0, eos_token_id=6,
                                     attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.))
    return HFBackend(model=model, bos_token_id=0, eos_token_id=6)


def test_off_diagonal_formula_and_duplicate_multiplicity():
    vectors = np.array([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]])
    z = normalize(vectors)
    direct = [1 - z[i] @ z[j] for i in range(4) for j in range(4) if i != j]
    assert cosine_u(vectors) == pytest.approx(np.mean(direct))
    assert cosine_u(vectors) != pytest.approx(cosine_u(np.unique(vectors, axis=0)))


def test_pairwise_bootstrap_resamples_both_axes_and_degeneracy():
    vectors = np.array([[1., 0.], [0., 1.], [-1., 0.], [1., 1.]])
    result = pairwise_estimate(vectors, bootstrap=30, seed=6)
    weights = np.random.default_rng(6).multinomial(4, np.full(4, .25), size=30)
    direct = [cosine_u(np.repeat(vectors, w, axis=0)) for w in weights]
    np.testing.assert_allclose(result.bootstrap_values, direct, atol=1e-14)
    equal = pairwise_estimate(np.ones((4, 2)), bootstrap=30)
    assert equal.value == pytest.approx(0, abs=1e-14)
    assert equal.ci_high > 0  # no false certainty at equal embeddings


def test_pairwise_unbiased_in_finite_enumeration():
    # All ordered samples of size 2 from a binary uniform population.
    pop = np.eye(2)
    expected = np.mean([cosine_u(pop[[i, j]]) for i in range(2) for j in range(2)])
    assert expected == pytest.approx(.5)


@pytest.mark.hf
def test_batched_sampler_teacher_forcing_eos_and_q(study_backend):
    b = study_backend
    a = StudySampler(b, batch_size=4).sample((0, 1), 12, SamplingSpec(b.model_spec, 20, 7))
    first_eos = b.probabilities((0, 1))[6]
    assert any(r.stop_reason == "eos" for r in a.sequences)
    for row in a.sequences:
        assert row.tokens[0] != 6
        assert 6 not in row.tokens[:-1]
        np.testing.assert_allclose(row.model_logprobs, b.score((0, 1), row.tokens), atol=4e-7)
        assert row.sampling_logprobs[0] == pytest.approx(
            row.model_logprobs[0] - math.log1p(-first_eos), abs=3e-7)
        np.testing.assert_allclose(row.sampling_logprobs[1:], row.model_logprobs[1:])


@pytest.mark.hf
def test_indexed_draws_extend_and_truncate(study_backend):
    s = StudySampler(study_backend, batch_size=4)
    spec = SamplingSpec(study_backend.model_spec, 5, 91)
    a = s.sample((0, 2), 4, spec)
    assert a.content_hash == s.sample((0, 2), 4, spec).content_hash
    assert s.extend(a, 4).content_hash == s.sample((0, 2), 4, replace(spec, n=9)).content_hash
    short = s.sample((0, 2), 2, spec)
    for x, y in zip(short.sequences, a.truncate(2).sequences, strict=True):
        assert x.tokens == y.tokens
        np.testing.assert_array_equal(x.model_logprobs, y.model_logprobs)


@pytest.mark.hf
def test_encoder_matches_original_hidden_state_zero(study_backend):
    import torch
    b = study_backend
    encoder = GPT2StudyEncoder(b)
    ids = torch.tensor([[1, 2, 3], [3, 4, 6]])
    with torch.inference_mode():
        hidden = b._model(ids, output_hidden_states=True).hidden_states[0].numpy()
    np.testing.assert_allclose(encoder.encode_tokens([1, 3]), hidden[:, 0], atol=1e-7)
    a = StudySampler(b, batch_size=4).sample((0, 1), 4, SamplingSpec(b.model_spec, 5, 9))
    for row, embedding in zip(a.sequences, encoder.encode_sequences(a), strict=True):
        with torch.inference_mode():
            h = b._model(torch.tensor([row.tokens]), output_hidden_states=True).hidden_states[0]
        valid = np.asarray(row.tokens) != b.eos_token_id
        np.testing.assert_allclose(embedding, h.numpy()[0, valid].mean(axis=0), atol=1e-7)


@pytest.mark.hf
def test_study_point_estimates_and_uncertainty(study_backend):
    b = study_backend
    a = StudySampler(b, batch_size=4).sample((0, 1), 3, SamplingSpec(b.model_spec, 8, 3))
    encoder = GPT2StudyEncoder(b)
    results = estimate_study(a, 2, encoder, bootstrap=30)
    assert results['when_first_token'].value == sum(r.tokens[0] == 2 for r in a.sequences)/8
    z = normalize(encoder.encode_sequences(a))
    matrix = 1 - z @ z.T
    assert results['expected_seq_decontextualised'].value == pytest.approx(
        matrix[~np.eye(8, dtype=bool)].mean(), abs=1e-14)
    cross = results['cross_entropy_model_bits'].value
    entropy = results['entropy_sampling_bits'].value
    assert cross - entropy == pytest.approx(-math.log2(1-b.probabilities((0, 1))[6]), abs=5e-7)
    assert all(r.ci_low <= r.value <= r.ci_high for r in results.values())


@pytest.mark.hf
def test_study_sampler_cannot_be_mislabeled_as_generic_raw_model(study_backend):
    from uncertain import ConfigError, Measure, surprisal
    spec = SamplingSpec(study_backend.model_spec, 4, 0)
    with pytest.raises(ConfigError, match="first-step EOS conditioning"):
        Measure(surprisal(spec, horizon=1, short_samples="compare_ragged"),
                sampler=StudySampler(study_backend))

"""Audited GPT-2 study estimators, separate from generic energy/anticipatory presets.

Sampling follows q: raw GPT-2 with EOS forbidden at the first step only.
Later EOS terminates the sequence, and no padding probabilities are scored.
The original study's point formulas are retained; pairwise uncertainty resamples
whole alternatives on both axes, rather than treating dependent row means as iid.
"""
from dataclasses import replace
import math
import numpy as np

from uncertain import AlternativeSet, ConfigError, Sequence, UnitSpec
from uncertain.samplers import Sampler
from uncertain.statistics import binomial, sample_mean
from uncertain.utils import canonical, seed_for


class StudySampler(Sampler):
    """Fixed-width GPT-2 KV-cache batches with independent indexed numpy RNGs.

    Fixed-width blocks (including unused rows) keep batching independent of n/start.
    Changing batch size changes the execution identity. Not a generic HF sampler.
    """
    study_specific_distribution = True

    def __init__(self, backend, batch_size=16):
        super().__init__(backend)
        if type(batch_size) is not int or batch_size < 1:
            raise ConfigError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.descriptor = {**self.descriptor, "sampler": "emnlp2024-gpt2-kv",
                           "version": "1", "batch_size": batch_size,
                           "first_eos": "suppressed", "later_eos": "stop"}

    def sample(self, context, horizon, spec, unit=None, *, start=0):
        unit = unit or UnitSpec()
        if unit.name != "token" or not spec.model_distribution:
            raise ConfigError("Study sampler requires token units and temperature=1, top_k=0")
        if horizon < 1 or horizon > spec.token_budget or start < 0:
            raise ConfigError("Invalid horizon, budget, or sample start")
        torch = self.backend._load()
        model = self.backend._model
        if model.config.model_type != "gpt2":
            raise ConfigError("StudySampler currently supports GPT-2 only")
        eos = self.backend.eos_token_id
        if eos is None:
            raise ConfigError("Study sampler requires EOS")
        context = tuple(context)
        prefix = self.backend._prefix(context)
        if len(prefix) + horizon > model.config.n_positions:
            raise ConfigError("Context plus horizon exceeds GPT-2 position capacity")
        stream = self.stream_key(context, spec, unit)
        self.calls += 1
        rows = []
        width = self.batch_size
        with torch.inference_mode():
            for block in range(start // width * width, start + spec.n, width):
                rngs = [np.random.default_rng(seed_for(stream, i))
                        for i in range(block, block + width)]
                tokens: list[list[int]] = [[] for _ in range(width)]
                lp: list[list[float]] = [[] for _ in range(width)]
                lq: list[list[float]] = [[] for _ in range(width)]
                done: np.ndarray = np.zeros(width, dtype=bool)
                ids = torch.tensor([prefix] * width, device=self.backend.device)
                past = None
                for step in range(horizon):
                    # Project only the final hidden state, avoiding B x T x vocab logits.
                    out = model.transformer(ids, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = model.lm_head(out.last_hidden_state[:, -1]).double()
                    logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                    proposal = logp.copy()
                    if step == 0:
                        proposal[:, eos] = -np.inf
                    q = np.exp(proposal - proposal.max(axis=1, keepdims=True))
                    q /= q.sum(axis=1, keepdims=True)
                    following = []
                    for j in range(width):
                        if done[j]:
                            following.append(eos)
                            continue
                        token = int(rngs[j].choice(q.shape[1], p=q[j]))
                        tokens[j].append(token)
                        lp[j].append(float(logp[j, token]))
                        lq[j].append(float(math.log(q[j, token])))
                        following.append(token)
                        done[j] = token == eos
                    if done.all():
                        break
                    ids = torch.tensor(following, device=self.backend.device).unsqueeze(1)
                for j in range(width):
                    sample_id = block + j
                    if start <= sample_id < start + spec.n:
                        ts = tuple(tokens[j])
                        rows.append(Sequence(ts, tuple(lp[j]), tuple(lq[j]),
                                             tuple(str(t) for t in ts),
                                             tuple((i, i+1) for i in range(len(ts))),
                                             text=self.backend.decode(ts),
                                             stop_reason="eos" if done[j] else "horizon",
                                             sample_id=sample_id))
        return AlternativeSet(tuple(rows), context, unit, horizon,
                              {"stream": stream, "prefix_consistent": True,
                               "sampling": canonical(spec),
                               "backend": canonical(self.backend.descriptor),
                               "sampler": self.descriptor,
                               "distribution": "q=p conditioned on first token != EOS"})


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norm == 0):
        raise ConfigError("Cosine distance needs finite nonzero embeddings")
    return vectors / norm


def cosine_u(vectors):
    """Off-diagonal mean cosine distance, including duplicate draw positions."""
    z = normalize(vectors)
    n = len(z)
    if n < 2:
        raise ConfigError("Pairwise expectation requires at least two draws")
    # Sum_i,j (1 - zi.zj), diagonal exactly zero. O(n*d), not O(n*n*d).
    return float(np.clip((n*n - np.square(z.sum(axis=0)).sum()) / (n*(n-1)), 0, 2))


def pairwise_estimate(vectors, *, bootstrap=199, seed=0, confidence=.95):
    from uncertain.core import StatisticEstimate
    z = normalize(vectors)
    n = len(z)
    value = cosine_u(z)
    if bootstrap < 20 or not 0 < confidence < 1:
        raise ConfigError("Invalid bootstrap count or confidence")
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=bootstrap)
    sums = weights @ z
    boots = np.clip((n*n - np.square(sums).sum(axis=1)) / (n*(n-1)), 0, 2)
    # Bounded differences for cosine in [0,2]: replacing one sample changes
    # the off-diagonal mean by at most 4/n. Valid even at degeneracy.
    radius = math.sqrt(8 * math.log(2/(1-confidence)) / n)
    return StatisticEstimate(value, float(boots.std(ddof=1)), max(0., value-radius),
                             min(2., value+radius), "bounded_pairwise_ci_bootstrap_se",
                             tuple(float(x) for x in boots))


class GPT2StudyEncoder:
    """Layer 0 at local positions: token embedding PLUS positional embedding.

Matches outputs.hidden_states[0] in the original GPT-2 scorer in eval mode.
The sequence pool excludes EOS and uses positions starting at zero, without BOS.
"""
    def __init__(self, backend):
        backend._load()
        if backend._model.config.model_type != "gpt2":
            raise ConfigError("Study encoder currently supports GPT-2 only")
        self.backend = backend
        self.token = backend._model.transformer.wte.weight.detach().cpu().numpy().copy()
        self.position = backend._model.transformer.wpe.weight.detach().cpu().numpy().copy()
        self.descriptor = {"encoder": "gpt2-study-layer0", "version": "1",
                           "model": canonical(backend.model_spec),
                           "pooling": "mean non-EOS", "position_origin": 0}

    def encode_tokens(self, ids):
        return (self.token[np.asarray(ids)] + self.position[0]).astype(float)

    def encode_sequences(self, alternatives):
        result = []
        for row in alternatives.sequences:
            ids = np.asarray(row.tokens, dtype=int)
            valid = ids != self.backend.eos_token_id
            if not valid.any():
                raise ConfigError("Empty non-EOS sequence has no study embedding")
            embeds = self.token[ids] + self.position[np.arange(len(ids))]
            result.append(embeds[valid].mean(axis=0).astype(float))
        return np.asarray(result)


def estimate_study(alternatives, target, encoder, *, bootstrap=199, seed=0):
    """Point estimates and uncertainty for one observed token and one horizon.

All expectations are under explicitly conditioned sampling law q. Original
sequence_entropy is exposed as cross_entropy_model_bits, since q != raw p.
"""
    if len(alternatives) < 2:
        raise ConfigError("Study estimation requires at least two draws")
    if any(not row.tokens for row in alternatives.sequences):
        raise ConfigError("Empty alternative")
    first = np.array([r.tokens[0] for r in alternatives.sequences])
    token_embeds = encoder.encode_tokens(first)
    target_embed = normalize(encoder.encode_tokens([target]))[0]
    iv = np.clip(1 - normalize(token_embeds) @ target_embed, 0, 2)
    prob = binomial(int(np.sum(first == target)), len(first), .95)
    lo = -math.log2(prob.ci_high)
    hi = -math.log2(prob.ci_low) if prob.ci_low else math.inf
    mc_surprisal = replace(prob, value=-math.log2(prob.value) if prob.value else math.inf,
                          se=(prob.se/(prob.value*math.log(2))
                              if prob.se is not None and prob.value else None),
                          ci_low=lo, ci_high=hi)
    result = {
        "when_first_token": prob,
        "mc_surprisal_bits": mc_surprisal,
        "decontextualised": sample_mean(iv, .95),
        "expected_decontextualised": pairwise_estimate(
            token_embeds, bootstrap=bootstrap, seed=seed),
        "expected_seq_decontextualised": pairwise_estimate(
            encoder.encode_sequences(alternatives), bootstrap=bootstrap, seed=seed),
    }
    for field, name in (("model_logprobs", "cross_entropy_model_bits"),
                        ("sampling_logprobs", "entropy_sampling_bits")):
        values = [-sum(getattr(r, field))/math.log(2) for r in alternatives.sequences]
        result[name] = sample_mean(values, .95)
    return result

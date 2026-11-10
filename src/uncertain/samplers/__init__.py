"""Reference ancestral sampler: one indexed RNG stream per draw, so any draw is reproducible
alone, `extend` adds draws without changing earlier ones, and a horizon-H set is the
exact prefix of a longer one (prefix consistency)."""
from dataclasses import replace
from functools import lru_cache
import math
import numpy as np
from uncertain.core import AlternativeSet, Sequence, SamplingSpec, UnitSpec, ConfigError
from uncertain.utils import seed_for, digest, canonical, immutable_array
from uncertain.alignment import align


def sampling_probabilities(p, generation):
    """Temperature, then stable top-k, then nucleus filtering; renormalize once."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or not np.all(np.isfinite(p)) or not np.isclose(p.sum(), 1):
        raise ValueError("backend returned an invalid probability distribution")
    logp = np.full_like(p, -np.inf)
    np.log(p, out=logp, where=p > 0)
    scaled = logp / generation["temperature"]
    q = np.exp(scaled - scaled.max())
    order = np.argsort(-q, kind="stable")
    k = generation["top_k"]
    if k:
        q[order[k:]] = 0
    q /= q.sum()
    cumulative: np.ndarray = np.cumsum(q[order])
    # Keep the token that crosses the threshold.
    remove = np.r_[False, cumulative[:-1] >= generation["top_p"]]
    q[order[remove]] = 0
    return q / q.sum()


def unterminated(sequences, horizon):
    """Sequences that stopped short of H units (EOS or budget).

    The name predates the §4.7 fix and is kept because provenance enters the content
    hash; the truncation bound counts non-EOS stops instead (Measure._open_fraction).
    """
    return sum(s.effective_horizon < horizon for s in sequences)


class Sampler:
    """Draws `spec.n` sequences of up to `horizon` units from any backend with
    `probabilities(context)`. `calls` counts sampling passes (cache misses)."""
    prefix_consistent = True

    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.descriptor = {"sampler": "ancestral-reference", "version": "1",
                           "prefix_consistent": self.prefix_consistent, "numpy": np.__version__}
        # Rows share prefixes (every row starts at p(.|context)), so one forward per
        # distinct context. ponytail: 256 full-vocab rows is ~100 MB for GPT-2; a
        # KV cache or batched rows in the backend is next if long horizons dominate.
        self._probabilities = lru_cache(maxsize=256)(
            lambda context: immutable_array(backend.probabilities(context), float))

    def stream_key(self, context, spec, unit):
        if spec.model != self.backend.model_spec:
            raise ConfigError("sampling ModelSpec does not match backend model identity")
        return digest({"backend": self.backend.descriptor, "sampler": self.descriptor, "context": tuple(context),
                       "generation": spec.generation, "seed": spec.seed,
                       "unit": unit, "token_budget": spec.token_budget})

    def request_key(self, context, horizon, spec, unit):
        return digest((self.stream_key(context, spec, unit), horizon, spec.n))

    def sample(self, context, horizon, spec: SamplingSpec, unit=None, *, start=0):
        """Draw sample IDs start..start+n-1 as an AlternativeSet of `horizon` units."""
        unit = unit if unit is not None else UnitSpec()
        if horizon < 1:
            raise ConfigError("sample horizon must be positive; use AlternativeSet.empty for H=0")
        if start < 0:
            raise ConfigError("sample start must be nonnegative")
        context = tuple(context)
        stream = self.stream_key(context, spec, unit)
        self.calls += 1
        rows = [self._row(context, horizon, spec, unit, stream, sample_id)
                for sample_id in range(start, start+spec.n)]
        return self._wrap(rows, context, unit, horizon, stream, spec)

    def _row(self, context, horizon, spec, unit, stream, sample_id, prefix=None):
        """Draw one row; `prefix` resumes a stored row of a shorter horizon on the same stream."""
        rng = np.random.default_rng(seed_for(stream, sample_id))
        tokens: list[int] = list(prefix.tokens) if prefix else []
        lp: list[float] = list(prefix.model_logprobs) if prefix else []
        lq: list[float] = list(prefix.sampling_logprobs) if prefix else []
        rng.random(len(tokens))  # choice(p=q) takes one uniform per token; skip the stored ones
        reason = "budget"
        for _ in range(spec.token_budget - len(tokens)):
            p = self._probabilities(context + tuple(tokens))
            q = sampling_probabilities(p, spec.generation)
            token = int(rng.choice(len(q), p=q))
            tokens.append(token)
            lp.append(math.log(p[token]))
            lq.append(math.log(q[token]))
            eos = token == self.backend.eos_token_id
            units, spans, chars, text = align(tuple(tokens), self.backend, unit, final=eos)
            if eos:
                reason = "eos"
                break
            if len(units) >= horizon:
                reason = "horizon"
                break
        units, spans, chars, text = align(tuple(tokens), self.backend, unit,
                                         final=reason == "eos")
        end, text_end = len(tokens), len(text)
        if len(units) >= horizon:
            # Word and sentence boundaries are only confirmed by the next unit's first
            # token, so drop that lookahead: the artifact must hold exactly H units, or
            # sample(H) and sample(H_max).truncate(H) disagree token for token.
            units, spans, chars = units[:horizon], spans[:horizon], chars[:horizon]
            end = spans[-1][1]
            text_end = chars[-1][1] if chars else 0
            if end < len(tokens):
                reason = "horizon"  # what survives the cut no longer ends at EOS
        return Sequence(tuple(tokens[:end]), tuple(lp[:end]), tuple(lq[:end]),
                        units, spans, chars, text=text[:text_end],
                        stop_reason=reason, sample_id=sample_id)

    def _wrap(self, rows, context, unit, horizon, stream, spec):
        return AlternativeSet(tuple(rows), context, unit, horizon,
                              {"stream": stream, "prefix_consistent": self.prefix_consistent,
                               "sampling": canonical(spec), "backend": canonical(self.backend.descriptor),
                               "context_text": self.backend.decode(context),
                               "unterminated": unterminated(rows, horizon)})

    def deepen(self, alternatives, horizon):
        """sample(horizon) rebuilt from a stored set of a shorter horizon on the same stream.

        Rows that stopped at the old horizon resume from their stored tokens; EOS and
        budget rows are already complete. Returns None for subclasses that draw differently.
        """
        if type(self).sample is not Sampler.sample:
            return None
        spec = self._spec(alternatives)
        stream = alternatives.provenance["stream"]
        self.calls += 1
        rows = [self._row(alternatives.context, horizon, spec, alternatives.unit, stream, s.sample_id,
                          prefix=s) if s.stop_reason == "horizon" else s
                for s in alternatives.sequences]
        return self._wrap(rows, alternatives.context, alternatives.unit, horizon, stream, spec)

    @staticmethod
    def _spec(alternatives):
        from uncertain.core import ModelSpec
        d = dict(alternatives.provenance["sampling"])
        d["model"] = ModelSpec(**d["model"])
        return SamplingSpec(**d)

    def extend(self, alternatives, k):
        """Append k new draws; existing draws are unchanged (same streams, new IDs)."""
        if not k > 0 or alternatives.deterministic:
            raise ValueError("extend requires k>0 and a sampled set")
        original = self._spec(alternatives)
        ids = tuple(s.sample_id for s in alternatives.sequences)
        if ids != tuple(range(len(ids))):
            raise ValueError("extend requires an unselected contiguous sample starting at zero")
        extra = self.sample(alternatives.context, alternatives.horizon, replace(original, n=k),
                            alternatives.unit, start=len(ids))
        sequences = alternatives.sequences + extra.sequences
        provenance = dict(alternatives.provenance)
        provenance["sampling"] = canonical(replace(original, n=len(ids)+k))
        if "unterminated" in provenance:  # subclasses need not record it
            provenance["unterminated"] = unterminated(sequences, alternatives.horizon)
        return replace(alternatives, sequences=sequences, provenance=provenance)


class HFSampler(Sampler):
    """§5 shape: ``HFSampler("gpt2", n=32, horizon=5, top_k=0, seed=0)``.

    Takes a model id (loaded with HFBackend at ``revision``) or a backend. n, horizon,
    seed and generation knobs (top_k, temperature, ...) become defaults for
    ``Measure(sampler=...)``, which may override them. The draws are the reference
    ancestral sampler's, so the descriptor -- and every spec_hash -- is Sampler's.
    """

    def __init__(self, backend, *, n=None, horizon=None, seed=None, revision=None, **generation):
        if isinstance(backend, str):
            from uncertain.backends.hf import HFBackend
            backend = HFBackend(backend, revision=revision)
        super().__init__(backend)
        self.defaults = {k: v for k, v in dict(n=n, horizon=horizon, seed=seed).items()
                         if v is not None}
        if generation:
            self.defaults["generation"] = generation


ToySampler = Sampler

__all__ = ["Sampler", "HFSampler", "ToySampler", "sampling_probabilities"]

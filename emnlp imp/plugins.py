"""Components for the EMNLP 2024 study, written against uncertain's extension API.

BatchedSampler: KV-cache, fixed-width batches of the reference ancestral sampler. Same
indexed per-draw RNG streams and the same inverse-CDF draw as Generator.choice(p=q), so
draws match uncertain.Sampler token for token (up to float ties); raw p, no EOS suppression.

layer0_local encoder: hidden_states[0] of the alternative alone, positions from 0, no BOS,
mean over non-EOS tokens -- the original study's "decontextualised" representation.
The built-in LayerEncoder(context_mode="isolated") prepends BOS, shifting positions to 1.

within_u statistic: off-diagonal mean cosine distance within A (the study's "expected"
measures); ignores B. Vectorized; the library's pairwise_blocks makes one Python call per pair.
"""
import math
import numpy as np

from uncertain import AlternativeSet, ConfigError, Sequence, UnitSpec
from uncertain.alignment import align
from uncertain.core import RepresentationSet
from uncertain.registry import register_distance, register_encoder, register_statistic
from uncertain.statistics import Mean, bootstrap_result, weights
from uncertain.samplers import Sampler
from uncertain.utils import seed_for


class BatchedSampler(Sampler):
    def __init__(self, backend, batch_size=512):
        super().__init__(backend)
        self.batch_size = batch_size
        self.descriptor = {**self.descriptor, "sampler": "batched-kv-ancestral", "version": "1",
                           "batch_size": batch_size}

    def sample(self, context, horizon, spec, unit=None, *, start=0):
        unit = unit or UnitSpec()
        g = spec.generation
        if unit.name != "token" or g["temperature"] != 1 or g["top_k"] or g["top_p"] != 1:
            raise ConfigError("BatchedSampler supports token units and unfiltered p only")
        torch = self.backend._load()
        model, eos, dev = self.backend._model, self.backend.eos_token_id, self.backend.device
        context = tuple(context)
        prefix = self.backend._prefix(context)
        stream = self.stream_key(context, spec, unit)
        self.calls += 1
        rows = []
        width = self.batch_size
        with torch.inference_mode():
            for block in range(start // width * width, start + spec.n, width):
                rngs = [np.random.default_rng(seed_for(stream, i)) for i in range(block, block + width)]
                tokens = [[] for _ in range(width)]
                lp = [[] for _ in range(width)]
                done = np.zeros(width, dtype=bool)
                ids, past = torch.tensor([prefix] * width, device=dev), None
                for _ in range(horizon):
                    out = model(ids, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    p = torch.softmax(out.logits[:, -1].double(), dim=-1)
                    # Generator.choice(p=q): cdf=cumsum(q)/cdf[-1]; searchsorted(cdf, random(), right)
                    cdf = p.cumsum(dim=-1)
                    cdf /= cdf[:, -1:].clone()
                    u = torch.tensor([r.random() if not d else 0. for r, d in zip(rngs, done)],
                                     device=dev, dtype=torch.float64)
                    picked = torch.searchsorted(cdf, u.unsqueeze(1), right=True).squeeze(1)
                    logp = torch.log(p.gather(1, picked.unsqueeze(1))).squeeze(1).cpu().numpy()
                    picked = picked.cpu().numpy()
                    following = []
                    for j in range(width):
                        if done[j]:
                            following.append(eos)
                            continue
                        t = int(picked[j])
                        tokens[j].append(t)
                        lp[j].append(float(logp[j]))
                        following.append(t)
                        done[j] = t == eos
                    if done.all():
                        break
                    ids = torch.tensor(following, device=dev).unsqueeze(1)
                for j in range(width):
                    sid = block + j
                    if start <= sid < start + spec.n:
                        ts = tuple(tokens[j])
                        units, spans, chars, text = align(ts, self.backend, unit, final=done[j])
                        rows.append(Sequence(ts, tuple(lp[j]), tuple(lp[j]), units, spans, chars,
                                             text=text, stop_reason="eos" if done[j] else "horizon",
                                             sample_id=sid))
        return self._wrap(rows, context, unit, horizon, stream, spec)


@register_encoder("layer0_local")
class Layer0Local:
    output_type = "vector"
    supported_units = {"token"}

    def __init__(self, backend=None):
        backend._load()
        self.backend = backend
        self.descriptor = {"encoder": "layer0_local", "version": "1", "pooling": "mean non-EOS",
                           "position_origin": 0, "backend": backend.descriptor}

    def encode(self, source, short_samples):
        torch = self.backend._load()
        eos = self.backend.eos_token_id
        rows = [r.tokens for r in source.sequences]
        if not all(rows):
            raise ConfigError("layer0_local cannot encode an empty sequence")
        width = max(map(len, rows))
        ids = torch.tensor([list(r) + [eos] * (width - len(r)) for r in rows], device=self.backend.device)
        # Layer 0 has no attention, so right padding cannot leak into real positions.
        with torch.inference_mode():
            h = self.backend._model(ids, output_hidden_states=True).hidden_states[0].double().cpu().numpy()
        reps = []
        for i, r in enumerate(rows):
            keep = np.array(r) != eos
            if not keep.any():
                keep[0] = True  # EOS-only alternative: its own embedding (the original suppressed it)
            reps.append(h[i, :len(r)][keep].mean(axis=0))
        return RepresentationSet(tuple(reps), source, self.descriptor, self.output_type)


@register_distance("cosine_fast")
class CosineFast:
    """Built-in Cosine's formula without per-call numpy overhead. The library's
    pairwise_blocks calls a distance once per pair (512^2 calls per set)."""
    input_type = "vector"
    range = (0., 2.)
    kind = "dissimilarity"
    negative_type = True
    symmetric = True
    descriptor = {"distance": "cosine_fast", "version": "1", "zero_vector": "error",
                  "same_as": "cosine:1"}

    def __call__(self, a, b):
        scale = math.sqrt(float(a @ a) * float(b @ b))
        if scale == 0:
            raise ValueError("cosine distance is undefined for a zero vector")
        return min(2., max(0., 1. - float(a @ b) / scale))


@register_statistic("within_u")
class WithinU(Mean):
    """mean_{i!=j} d(a_i, a_j) for cosine d, via sum_{i,j}(1 - z_i.z_j) = n^2 - |sum z|^2.
    Bootstrap replicates use the library's whole-alternative weights, exactly as the
    within-A term of energy(variant="u"): (n^2 - |w @ z|^2) / (n (n-1))."""

    def __init__(self, block_size=64):
        super().__init__(block_size)
        self.descriptor = {"statistic": "within_u", "version": "1", "ignores": "B"}

    def estimate(self, a, b, distance, estimator, seeds):
        if distance.descriptor.get("distance") not in {"cosine", "cosine_fast"}:
            raise ConfigError("within_u is implemented for cosine distance only")
        z = np.asarray(a.reps, dtype=float)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if len(z) < 2 or np.any(norms == 0):
            raise ConfigError("within_u needs >= 2 nonzero vectors")
        z /= norms
        n = len(z)
        u = lambda s: np.clip((n * n - np.square(s).sum(axis=-1)) / (n * (n - 1)), 0, 2)
        boots = u(weights(a.source, estimator.bootstrap, seeds[0]) @ z)
        return bootstrap_result(float(u(z.sum(axis=0))), boots, a, b, estimator)


LN2 = math.log(2)

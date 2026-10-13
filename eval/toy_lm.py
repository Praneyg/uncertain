"""Deterministic toy LM with closed-form answers: the analytic oracle for the eval suite.

Written independently of `uncertain.backends.ToyBackend`, so a bug there cannot hide here.

* Next-token distribution: a fixed categorical `p`, optionally overridden per last token
  (`rows`) to make the continuation context-dependent for H > 1 tests.
* Embeddings: one-hot at every layer, so cosine distance is 0 for equal tokens, 1 otherwise.
* `calls` counts forward passes (`probabilities` + `hidden`), for caching/spy tests.

Closed forms used by the tests (horizon 1, one-hot / discrete distance):
    E d(A, o)      = 1 - p[o]
    E d(A, A')     = 1 - sum p_i^2          (non-zero when P = Q: the mean-distance flaw)
    energy(P, d_o) = 1 - 2 p[o] + sum p_i^2
    energy(P, P)   = 0
"""
import itertools
import math
import numpy as np
from uncertain import ModelSpec
from uncertain.utils import digest


class ToyLM:
    def __init__(self, p, *, pieces=None, eos_token_id=None, rows=None, name="eval-toy"):
        self.p = np.asarray(p, dtype=float)
        assert self.p.ndim == 1 and np.all(self.p >= 0) and math.isclose(self.p.sum(), 1, abs_tol=1e-12)
        self.V = len(self.p)
        self.pieces = tuple(pieces) if pieces is not None else tuple(f"<{i}>" for i in range(self.V))
        assert len(self.pieces) == self.V
        self.rows = {int(k): np.asarray(v, dtype=float) for k, v in (rows or {}).items()}
        self.eos_token_id = eos_token_id
        self.bos_token_id = None
        self.calls = 0
        identity = {"p": self.p.tolist(), "pieces": self.pieces, "eos": eos_token_id,
                    "rows": sorted((k, v.tolist()) for k, v in self.rows.items())}
        self.model_spec = ModelSpec(name, digest(identity))
        self.descriptor = {"backend": "eval-toy:v1", "model": self.model_spec, **identity}

    # --- Backend protocol -------------------------------------------------------------
    def next_p(self, context):
        context = tuple(context)
        if context and context[-1] in self.rows:
            return self.rows[context[-1]]
        return self.p

    def probabilities(self, context):
        self.calls += 1
        return self.next_p(context).copy()

    def hidden(self, context, tokens, layer):
        self.calls += 1
        return np.eye(self.V)[list(tokens)] if tokens else np.empty((0, self.V))

    def decode(self, tokens):
        return "".join(self.pieces[t] for t in tokens if t != self.eos_token_id)

    def encode(self, text):
        out = []
        while text:
            best = max((len(s), i) for i, s in enumerate(self.pieces)
                       if s and text.startswith(s) and i != self.eos_token_id)
            out.append(best[1])
            text = text[best[0]:]
        return tuple(out)

    # --- Exact enumeration (token unit) ------------------------------------------------
    def sequences(self, context, H):
        """{token tuple: probability} for ancestral samples of H tokens (stopping at EOS)."""
        out = {}

        def walk(ctx, seq, prob):
            if len(seq) == H or (seq and seq[-1] == self.eos_token_id):
                out[seq] = out.get(seq, 0.) + prob
                return
            for t, q in enumerate(self.next_p(ctx)):
                if q > 0:
                    walk(ctx + (t,), seq + (t,), prob * q)
        walk(tuple(context), (), 1.)
        return out


def exact_mean(left, right, d):
    """E_{a~left, b~right} d(a, b) for two finite distributions {seq: prob}."""
    return sum(pa * pb * d(a, b) for a, pa in left.items() for b, pb in right.items())


def exact_anchor(lm, context, observed, H):
    """Distribution of the right operand anchor(observed, trunc(S_{t+1}, H-k))."""
    k = len(observed)
    if H == k or (lm.eos_token_id is not None and observed[-1] == lm.eos_token_id):
        return {tuple(observed): 1.}
    return {tuple(observed) + s: q for s, q in lm.sequences(tuple(context) + tuple(observed), H - k).items()}


def discrete(a, b):
    return float(a != b)


def jaccard_distance(n):
    def d(a, b):
        x = {a[i:i+n] for i in range(len(a)-n+1)}
        y = {b[i:i+n] for i in range(len(b)-n+1)}
        return 1. - (len(x & y) / len(x | y) if x or y else 1.)
    return d


def spearman(x, y):
    """Spearman rho with average ranks for ties (scipy is not installed)."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):
            idx = v == value
            r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def zscore(estimate, se, truth):
    return abs(estimate - truth) / se if se else (0. if math.isclose(estimate, truth, abs_tol=1e-12) else math.inf)


def all_tuples(alphabet, max_len):
    for k in range(max_len + 1):
        yield from itertools.product(alphabet, repeat=k)

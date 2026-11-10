"""Immutable sample containers: Sequence (one draw), AlternativeSet (n draws from one
context) and RepresentationSet (their encodings)."""
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
import numpy as np
from uncertain.core.specs import UnitSpec, freeze
from uncertain.utils import digest, immutable_array


@dataclass(frozen=True)
class Sequence:
    """One draw: tokens, per-step log p (model) and log q (sampling), and the completed
    units with their token/character spans. `observed` marks tokens given, not sampled."""
    tokens: tuple[int, ...]
    model_logprobs: tuple[float | None, ...]
    sampling_logprobs: tuple[float | None, ...]
    units: tuple[str, ...]
    unit_spans: tuple[tuple[int, int], ...]
    char_spans: tuple[tuple[int, int], ...] = ()
    observed: tuple[bool, ...] = ()
    text: str = ""
    stop_reason: str = "horizon"
    sample_id: int = 0

    def __post_init__(self):
        for key in ("tokens", "model_logprobs", "sampling_logprobs", "units", "observed"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for key in ("unit_spans", "char_spans"):
            object.__setattr__(self, key, tuple(tuple(x) for x in getattr(self, key)))
        if not self.observed:
            object.__setattr__(self, "observed", (False,) * len(self.tokens))
        if any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0 for t in self.tokens):
            raise ValueError("token IDs must be nonnegative integers")
        if not (len(self.tokens) == len(self.model_logprobs) == len(self.sampling_logprobs)
                == len(self.observed)):
            raise ValueError("tokens/logprobs/observed must align step by step")
        if len(self.units) != len(self.unit_spans):
            raise ValueError("one token span per completed unit is required")
        previous = (-1, -1)
        for a, b in self.unit_spans:
            if not 0 <= a < b <= len(self.tokens) or a < previous[0] or b < previous[1]:
                raise ValueError("unit spans must be ordered, nonempty and in bounds")
            previous = (a, b)
        if self.char_spans and len(self.char_spans) != len(self.units):
            raise ValueError("character spans must align to units")
        for a, b in self.char_spans:
            if not 0 <= a < b <= len(self.text):
                raise ValueError("character span out of bounds")
        for stream in (self.model_logprobs, self.sampling_logprobs):
            if any(x is not None and (not np.isfinite(x) or x > 1e-10) for x in stream):
                raise ValueError("stored sampled log probabilities must be finite and <=0")
        if self.stop_reason not in {"horizon", "eos", "budget", "observed"}:
            raise ValueError("invalid stop_reason")

    @property
    def effective_horizon(self):
        return len(self.units)

    def truncate(self, horizon):
        """The first `horizon` units of this draw."""
        if horizon < 0:
            raise ValueError("negative horizon")
        if horizon > len(self.units):
            return self
        end = self.unit_spans[horizon - 1][1] if horizon else 0
        if end == len(self.tokens) and horizon == len(self.units):
            return self  # nothing past unit H: an EOS stop at exactly H keeps its reason
        text_end = self.char_spans[horizon - 1][1] if horizon and self.char_spans else 0
        # end == len(tokens) with units left over: the last token spans into unit H+1.
        # Keep the tokens, drop the extra units, as the sampler does for sample(H).
        return replace(self, tokens=self.tokens[:end], model_logprobs=self.model_logprobs[:end],
                       sampling_logprobs=self.sampling_logprobs[:end],
                       observed=self.observed[:end], units=self.units[:horizon],
                       unit_spans=self.unit_spans[:horizon], char_spans=self.char_spans[:horizon],
                       text=self.text[:text_end],
                       stop_reason=self.stop_reason if end == len(self.tokens) else "horizon")


@dataclass(frozen=True)
class AlternativeSet:
    """Draws conditioned on one context. `deterministic` marks a single fixed operand
    (an observed target), which bootstrap resampling leaves unchanged."""
    sequences: tuple[Sequence, ...]
    context: tuple[int, ...]
    unit: UnitSpec
    horizon: int
    provenance: Mapping = field(default_factory=dict)
    deterministic: bool = False

    def __post_init__(self):
        object.__setattr__(self, "sequences", tuple(self.sequences))
        object.__setattr__(self, "context", tuple(self.context))
        if any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0 for t in self.context):
            raise ValueError("context token IDs must be nonnegative integers")
        object.__setattr__(self, "provenance", freeze(self.provenance))
        if self.horizon < 0 or any(s.effective_horizon > self.horizon for s in self.sequences):
            raise ValueError("effective horizon exceeds requested horizon")
        ids = [s.sample_id for s in self.sequences]
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be unique; duplicate draws retain distinct IDs")
        if self.deterministic and len(self.sequences) != 1:
            raise ValueError("a deterministic operand contains exactly one sequence")

    @property
    def content_hash(self):
        return digest(self)

    def __len__(self):
        return len(self.sequences)

    def truncate(self, horizon):
        """Every draw cut to `horizon` units; equals sampling at that horizon directly."""
        if not 0 <= horizon <= self.horizon:
            raise ValueError("truncation horizon must be in [0, current horizon]")
        if horizon != self.horizon and self.provenance.get("prefix_consistent") is False:
            raise ValueError("sampler does not support prefix-consistent truncation")
        return replace(self, sequences=tuple(s.truncate(horizon) for s in self.sequences),
                       horizon=horizon)

    def select(self, indices):
        return replace(self, sequences=tuple(self.sequences[i] for i in indices))

    @classmethod
    def empty(cls, context, unit):
        return cls((Sequence((), (), (), (), (), stop_reason="observed"),),
                   tuple(context), unit, 0, deterministic=True)

    @classmethod
    def anchor(cls, observed: Sequence, continuation: "AlternativeSet", context):
        """B: the observed target followed by each sampled continuation of it."""
        k, offset = len(observed.units), len(observed.tokens)
        if not k:
            raise ValueError("anchor requires at least one observed unit")
        if tuple(context) + observed.tokens != continuation.context:
            raise ValueError("anchor continuation must be conditioned on the observed token prefix")
        rows = []
        for s in continuation.sequences:
            chars = len(observed.text)
            rows.append(Sequence(
                observed.tokens + s.tokens, (None,) * offset + s.model_logprobs,
                (None,) * offset + s.sampling_logprobs, observed.units + s.units,
                observed.unit_spans + tuple((a+offset, b+offset) for a,b in s.unit_spans),
                observed.char_spans + tuple((a+chars, b+chars) for a,b in s.char_spans),
                (True,) * offset + s.observed, observed.text + s.text,
                "observed" if continuation.deterministic else s.stop_reason, s.sample_id))
        return cls(tuple(rows), tuple(context), continuation.unit, k + continuation.horizon,
                   {"anchor": digest(observed), "continuation": continuation.content_hash},
                   continuation.deterministic)

    def _grouped(self, top, decode=None):
        """Top distinct continuations as (text, count, share, mean -log p) rows.

        Token units carry no text; pass ``decode`` (e.g. ``backend.decode``) to label
        them, otherwise their unit ids are shown.
        """
        groups = {}
        for s in self.sequences:
            nll = None if None in s.model_logprobs else -sum(s.model_logprobs)
            label = s.text or (decode(s.tokens) if decode else " ".join(s.units))
            label += "<eos>" if s.stop_reason == "eos" else ""
            groups.setdefault(label, []).append(nll)
        rows = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top]
        return [(text, len(nlls), len(nlls)/len(self),
                 None if None in nlls else sum(nlls)/len(nlls)) for text, nlls in rows]

    def summary(self, top=20, decode=None):
        """Text table of the most frequent continuations."""
        context = self.provenance.get("context_text", "")
        lines = [f"{len(self)} alternatives, horizon {self.horizon}, context {context!r}",
                 f"{'count':>6}  {'share':>6} {'':20} {'-log p':>7}  text"]
        for text, count, share, nll in self._grouped(top, decode):
            nll_text = "-" if nll is None else f"{nll:.2f}"
            lines.append(f"{count:>6}  {share:>6.1%} {'#'*round(20*share):20} {nll_text:>7}  {text!r}")
        return "\n".join(lines)

    def plot(self, top=20, ax=None, decode=None):
        """Horizontal bar chart of continuation shares; needs the ``plot`` extra."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install uncertain-altsets[plot]") from None
        rows = self._grouped(top, decode)[::-1]
        ax = ax or plt.subplots(figsize=(7, .35*len(rows) + 1))[1]
        ax.barh([repr(text) for text, *_ in rows], [share for _, _, share, _ in rows])
        ax.set_xlabel("share of alternatives")
        ax.set_title(f"{len(self)} alternatives, horizon {self.horizon}")
        return ax

    def arrays(self):
        """Canonical (sample, step) views, with explicitly masked unavailable logprobs."""
        n, width = len(self), max((len(s.tokens) for s in self.sequences), default=0)
        tokens: np.ndarray = np.zeros((n, width), dtype=np.int64)
        mask: np.ndarray = np.zeros((n, width), dtype=bool)
        observed: np.ndarray = np.zeros((n, width), dtype=bool)
        model = np.full((n, width), np.nan)
        sampling = model.copy()
        for i, s in enumerate(self.sequences):
            k = len(s.tokens)
            tokens[i, :k], mask[i, :k], observed[i, :k] = s.tokens, True, s.observed
            model[i, :k] = [np.nan if x is None else x for x in s.model_logprobs]
            sampling[i, :k] = [np.nan if x is None else x for x in s.sampling_logprobs]
        return {key: immutable_array(value, value.dtype) for key, value in
                dict(tokens=tokens, mask=mask, observed=observed,
                     model_logprobs=model, sampling_logprobs=sampling).items()}


@dataclass(frozen=True)
class RepresentationSet:
    reps: tuple
    source: AlternativeSet
    descriptor: Mapping
    input_type: str

    def __post_init__(self):
        object.__setattr__(self, "descriptor", freeze(self.descriptor))
        object.__setattr__(self, "reps", tuple(
            immutable_array(x, x.dtype) if isinstance(x, np.ndarray) else x for x in self.reps))
        if len(self.reps) != len(self.source):
            raise ValueError("representations must preserve sample alignment")

    @property
    def content_hash(self):
        return digest((self.source.content_hash, self.descriptor))

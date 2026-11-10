"""Stimuli (context + observed target as token IDs), context transforms and loaders."""
from dataclasses import dataclass, replace
import json
import numpy as np
from uncertain.core import Sequence, ConfigError
from uncertain.alignment import align, AlignmentError
from uncertain.utils import digest, seed_for


@dataclass(frozen=True)
class Stimulus:
    """One item: `target` observed after `context`, with optional human `datum`."""
    context: tuple[int, ...] = ()
    target: tuple[int, ...] = ()
    stimulus_id: str = "item"
    position: int = 0
    datum: float | None = None
    document_id: str | None = None
    transform_id: str = "identity"
    original_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "context", tuple(self.context))
        object.__setattr__(self, "target", tuple(self.target))
        if any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0
               for t in self.context+self.target):
            raise ValueError("context/target token IDs must be nonnegative integers")
        if self.position < 0:
            raise ValueError("position must be nonnegative")
        if self.datum is not None and not np.isfinite(self.datum):
            raise ValueError("datum must be finite or None")

    @classmethod
    def from_text(cls, context, target, backend, **kwargs):
        prefix = backend.encode(context)
        whole = backend.encode(context+target)
        if whole[:len(prefix)] != prefix:
            raise AlignmentError("target retokenizes the context prefix; use explicit token boundaries")
        return cls(prefix, whole[len(prefix):], **kwargs)

    def observed(self, backend, unit):
        """The target as an observed Sequence of completed units."""
        if not self.target:
            raise ConfigError("observed measures require a nonempty target")
        units, spans, chars, text = align(self.target, backend, unit, final=True)
        return Sequence(self.target, (None,)*len(self.target), (None,)*len(self.target),
                        units, spans, chars, (True,)*len(self.target), text, "observed")


@dataclass(frozen=True)
class StimulusTransform:
    """Context manipulation: drop, truncate to the last `count`, replace, or shuffle."""
    kind: str
    count: int | None = None
    replacement: tuple[int, ...] = ()
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "replacement", tuple(self.replacement))
        if self.kind not in {"drop", "truncate", "replace", "shuffle"}:
            raise ConfigError("unknown context transform")
        if self.kind == "truncate" and (self.count is None or self.count < 0):
            raise ConfigError("truncate requires a nonnegative count")

    def __call__(self, stimulus):
        context = stimulus.context
        if self.kind == "drop":
            context = ()
        elif self.kind == "truncate":
            context = context[-self.count:] if self.count else ()
        elif self.kind == "replace":
            context = self.replacement
        else:
            rng = np.random.default_rng(seed_for(self.seed, stimulus.stimulus_id, context))
            context = tuple(int(x) for x in rng.permutation(context))
        return replace(stimulus, context=context, transform_id=digest((self, stimulus.transform_id)),
                       original_id=stimulus.original_id or stimulus.stimulus_id)


def positions(tokens, *, context=(), document_id="document", data=None):
    """Produce token-position stimuli; supplied datum values must align exactly."""
    tokens, context = tuple(tokens), tuple(context)
    if data is not None and len(data) != len(tokens):
        raise ValueError("one datum per token is required")
    return [Stimulus(context+tokens[:i], (t,), f"{document_id}:{i}", i,
                     None if data is None else data[i], document_id)
            for i,t in enumerate(tokens)]


def load_dataset(path):
    """Read a JSON array or JSONL stimulus corpus; never download named datasets."""
    from pathlib import Path
    text = Path(path).read_text(encoding="utf-8")
    records = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    return [Stimulus(**row) for row in records]


load_corpus = load_dataset

"""Backend protocol and finite exact reference model."""
from typing import Protocol
import numpy as np
from uncertain.core import ModelSpec
from uncertain.utils import digest
from .hf import HFBackend


class Backend(Protocol):
    model_spec: ModelSpec
    eos_token_id: int | None
    descriptor: dict

    def probabilities(self, context: tuple[int, ...]) -> np.ndarray: ...
    def decode(self, tokens) -> str: ...
    def encode(self, text: str) -> tuple[int, ...]: ...


class ToyBackend:
    """Finite suffix-conditioned model; rows are exact next-token probabilities.

    The longest matching suffix selects a row. The empty tuple supplies the
    fallback row. Vocabulary pieces concatenate exactly when decoded.
    """
    def __init__(self, vocabulary=("a", "b"), transitions=None, *, eos_token_id=None):
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary must contain unique pieces")
        if transitions is None:
            transitions = {(): (.75, .25)}
        self.transitions = {}
        for context, probs in transitions.items():
            p = np.asarray(probs, dtype=float)
            if p.shape != (len(self.vocabulary),) or not np.all(np.isfinite(p)) or np.any(p < 0):
                raise ValueError("invalid transition probabilities")
            if not np.isclose(p.sum(), 1, atol=1e-12):
                raise ValueError("transition probabilities must sum to one")
            self.transitions[tuple(context)] = tuple(float(x) for x in p / p.sum())
        if () not in self.transitions:
            raise ValueError("an empty-context fallback row is required")
        if eos_token_id is not None and not 0 <= eos_token_id < len(self.vocabulary):
            raise ValueError("EOS outside vocabulary")
        self.eos_token_id = eos_token_id
        identity = {"vocabulary": self.vocabulary, "rows": sorted(self.transitions.items()),
                    "eos": eos_token_id}
        self.model_spec = ModelSpec("toy", digest(identity))
        self.descriptor = {"backend": "toy:v1", **identity, "model": self.model_spec}

    def probabilities(self, context):
        context = tuple(context)
        matches = [key for key in self.transitions if not key or context[-len(key):] == key]
        return np.array(self.transitions[max(matches, key=len)], dtype=float)

    def decode(self, tokens):
        return "".join(self.vocabulary[t] for t in tokens if t != self.eos_token_id)

    def encode(self, text):
        result = []
        while text:
            candidates = [(len(s), i, s) for i,s in enumerate(self.vocabulary)
                          if s and text.startswith(s) and i != self.eos_token_id]
            if not candidates:
                raise ValueError(f"text cannot be encoded by toy vocabulary: {text!r}")
            _, i, piece = max(candidates)
            result.append(i)
            text = text[len(piece):]
        return tuple(result)


__all__ = ["Backend", "ToyBackend", "HFBackend"]

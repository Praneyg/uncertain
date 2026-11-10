"""Built-in encoders: an AlternativeSet -> one representation per draw.

`tokens` (unit tuples), `strings` (decoded text) and `layer` (pooled hidden states).
The short-sample policy decides whether short draws are padded with the unit's
terminal symbol (`pad`) or left ragged.
"""
import numpy as np
from uncertain.core import RepresentationSet, ConfigError
from uncertain.registry import register_encoder


def unit_rows(source, short_samples):
    """Each draw's units, padded to the horizon when short_samples == "pad"."""
    return tuple(s.units + ((source.unit.pad_symbol,) * (source.horizon-len(s.units))
                           if short_samples == "pad" else ()) for s in source.sequences)


@register_encoder("tokens")
class TokensEncoder:
    output_type = "units"
    supported_units = {"token", "word", "sentence"}
    descriptor = {"encoder": "tokens", "version": "1"}

    def __init__(self, backend=None):
        pass

    def encode(self, source, short_samples):
        return RepresentationSet(unit_rows(source, short_samples), source,
                                 {**self.descriptor, "short_samples": short_samples}, self.output_type)


@register_encoder("strings")
class StringsEncoder(TokensEncoder):
    output_type = "string"

    def __init__(self, backend=None):
        self.backend = backend
        self.descriptor = {"encoder": "strings", "version": "1", "backend": backend.descriptor}

    def encode(self, source, short_samples):
        rows = unit_rows(source, short_samples)
        reps = tuple(" ".join(x) for x in rows)
        if source.unit.name == "token":
            reps = tuple(self.backend.decode(row.tokens[:source.horizon]) +
                         (source.unit.pad_symbol*(source.horizon-len(row.units))
                          if short_samples == "pad" else "") for row in source.sequences)
        return RepresentationSet(reps, source,
                                 {**self.descriptor, "short_samples": short_samples}, self.output_type)


@register_encoder("layer")
class LayerEncoder:
    output_type = "vector"
    supported_units = {"token", "word", "sentence"}

    def __init__(self, layer=0, pooling="mean", context_mode="conditioned", backend=None):
        if backend is None or not hasattr(backend, "hidden"):
            raise ConfigError("layer encoder requires a backend with hidden representations")
        if type(layer) is not int or pooling not in {"mean", "first", "last", "sum"}:
            raise ConfigError("layer must be integer; pooling must be mean/first/last/sum")
        if context_mode not in {"conditioned", "isolated"}:
            raise ConfigError("context_mode must be conditioned or isolated")
        self.backend, self.layer, self.pooling, self.context_mode = backend, layer, pooling, context_mode
        self.descriptor = {"encoder": "layer", "version": "1", "layer": layer,
                           "pooling": pooling, "context_mode": context_mode,
                           "backend": backend.descriptor}

    def encode(self, source, short_samples):
        if short_samples == "pad":
            raise ConfigError("layer encoder has no semantic terminal embedding; use ragged/drop")
        reps = []
        for row in source.sequences:
            end = row.unit_spans[-1][1] if row.unit_spans else 0
            hidden = self.backend.hidden(source.context if self.context_mode == "conditioned" else (),
                                         row.tokens[:end], self.layer)
            if not row.unit_spans:
                raise ConfigError("layer encoding an empty completed sequence has no defined vector")
            unit_vectors = np.array([hidden[a:b].mean(axis=0) for a,b in row.unit_spans])
            if self.pooling in {"first", "last"}:
                vector = unit_vectors[0 if self.pooling == "first" else -1]
            elif self.pooling == "sum":
                vector = unit_vectors.sum(axis=0)
            else:
                vector = unit_vectors.mean(axis=0)
            reps.append(vector)
        return RepresentationSet(tuple(reps), source, self.descriptor, self.output_type)

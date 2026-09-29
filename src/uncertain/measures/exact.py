"""Finite-vocabulary expectations for one-token observed measures.

All positive-probability outcomes are retained, including EOS. Outcomes carry
probability weights; they are not an unweighted synthetic sample. Energy compares
the model distribution with the deterministic observed-token distribution.
"""
import numpy as np
from uncertain.core import AlternativeSet, ConfigError
from uncertain.data import Stimulus
from uncertain.distances import Cosine, Discrete
from uncertain.statistics import Energy


def finite_expectation(measure, stimulus):
    p = np.asarray(measure._probabilities(stimulus.context), dtype=float)
    if (p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0)
            or not np.isclose(p.sum(), 1, atol=1e-12)):
        raise ConfigError("Exact enumeration requires normalized finite probabilities")
    target = stimulus.target[0]
    if not 0 <= target < len(p):
        raise ConfigError("Target token outside vocabulary")
    support = np.flatnonzero(p)
    weights = p[support]
    backend, unit = measure.sampler.backend, measure.spec.unit
    outcomes = tuple(Stimulus(stimulus.context, (int(t),)).observed(backend, unit)
                     for t in support)
    # Observed sequences initially share the default ID; enumerated outcomes need
    # distinct identities. These are support indices, never sampled draw IDs.
    from dataclasses import replace
    outcomes = tuple(replace(row, sample_id=i) for i, row in enumerate(outcomes))
    source = AlternativeSet(outcomes, stimulus.context, unit, 1,
                            {"enumeration": "positive_probability_support"})
    observed = AlternativeSet((stimulus.observed(backend, unit),), stimulus.context,
                               unit, 1, deterministic=True)
    a = measure.encoder.encode(source, measure.spec.short_samples).reps
    b = measure.encoder.encode(observed, measure.spec.short_samples).reps[0]
    distance = measure.distance
    energy = type(measure.statistic) is Energy

    def checked(x, y):
        value = float(distance(x, y))
        if (not np.isfinite(value) or value < distance.range[0]-1e-10
                or value > distance.range[1]+1e-10):
            raise ConfigError("Distance returned a nonfinite or out-of-range value")
        return value

    cross = float(sum(w * checked(x, b) for w, x in zip(weights, a, strict=True)))
    algorithm = "weighted_vocabulary_sum"
    if energy:
        if type(distance) is Cosine:
            vectors = np.asarray(a, dtype=float)
            norms = np.linalg.norm(vectors, axis=1)
            target_norm = np.linalg.norm(b)
            if np.any(norms == 0) or target_norm == 0:
                raise ConfigError("Cosine distance is undefined for a zero vector")
            mean = weights @ (vectors / norms[:, None])
            # Energy for cosine is the squared distance between normalized means.
            raw = float(np.square(mean - np.asarray(b)/target_norm).sum())
            algorithm = "cosine_weighted_mean_identity"
        elif type(distance) is Discrete:
            masses: dict[tuple, float] = {}
            for w, rep in zip(weights, a, strict=True):
                masses[rep] = masses.get(rep, 0.) + float(w)
            raw = float(1 - 2*masses.get(b, 0.) + sum(v*v for v in masses.values()))
            algorithm = "discrete_probability_mass_identity"
        else:
            # Generic exact energy can require |V|^2 evaluations. This loop avoids
            # materializing that matrix. Custom negative-type distances remain exact.
            within = 0.
            for i, x in enumerate(a):
                within += float(weights[i]) * sum(float(w)*checked(x, y)
                                                  for w, y in zip(weights, a, strict=True))
            raw = 2*cross - within - checked(b, b)
            algorithm = "weighted_pairwise_enumeration"
        if raw < -1e-9:
            raise ConfigError("Negative population energy: check the distance contract")
        raw = max(0., raw)
    else:
        raw = cross
    return raw, {"algorithm": algorithm, "support_size": len(support),
                 "enumerated_probability_mass": float(weights.sum()),
                 "sampling_performed": False}

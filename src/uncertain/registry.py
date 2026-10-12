"""Versioned extension points. Registration has no model loading side effects."""
from functools import partial
from uncertain.core import ConfigError
from uncertain.utils import canonical_json
from typing import Any, Callable

_REGISTRY: dict[str, dict[tuple[str, str], Callable[..., Any]]] = {
    kind: {} for kind in ("encoder", "distance", "statistic", "warping", "unit", "sampler")}


def register(kind, name, *, version="1"):
    def decorator(factory):
        key = (name, version)
        if key in _REGISTRY[kind]:
            raise ConfigError(f"{kind} {name}@{version} already registered")
        _REGISTRY[kind][key] = factory
        return factory
    return decorator


def resolve(kind, name, config=None, *, version="1", **runtime):
    try:
        factory = _REGISTRY[kind][(name, version)]
    except KeyError as exc:
        raise ConfigError(f"Unknown {kind}: {name}@{version}") from exc
    component = adapt(kind, factory(**dict(config or {}), **runtime), name, version)
    if not hasattr(component, "descriptor"):
        raise ConfigError(f"{kind} {name} must declare a stable descriptor")
    canonical_json(component.descriptor)
    return component


class _Plugin:
    """Library-protocol view of a §7-shaped plugin. Declarations it lacks get the
    defaults passed here; everything else falls through to the wrapped object."""

    def __init__(self, inner, **defaults):
        self._inner = inner
        for key, value in defaults.items():
            if not hasattr(inner, key):
                setattr(self, key, value)

    def __getattr__(self, attr):
        if attr == "_inner":  # not yet set (copy/pickle): don't recurse
            raise AttributeError(attr)
        return getattr(self._inner, attr)

    def __call__(self, *args):
        return self._inner(*args)


class _SpecStatistic(_Plugin):
    """§7 statistic: `range` + `__call__(A, B)`, lifted with a whole-alternative bootstrap."""
    supports_uncertainty = True

    def range_for(self, distance):
        return tuple(float(x) for x in self._inner.range)

    def estimate(self, a, b, distance, estimator, seeds):
        from uncertain.statistics import weights, bootstrap_result
        value = float(self._inner(a, b))
        # ponytail: one user call per replicate (R=bootstrap, default 199); vectorize
        # only if a plugin statistic is ever the bottleneck.
        boots = [float(self._inner(_resampled(a, wa), _resampled(b, wb)))
                 for wa, wb in zip(weights(a.source, estimator.bootstrap, seeds[0]),
                                   weights(b.source, estimator.bootstrap, seeds[1]), strict=True)]
        return bootstrap_result(value, boots, a, b, estimator)


def _resampled(reps, multiplicity):
    """The bootstrap draw `multiplicity` as a RepresentationSet (sample IDs renumbered)."""
    from dataclasses import replace
    from uncertain.core.values import RepresentationSet
    import numpy as np
    idx: np.ndarray = np.repeat(np.arange(len(reps.reps)), multiplicity.astype(int))
    rows = tuple(replace(reps.source.sequences[i], sample_id=k) for k, i in enumerate(idx))
    return RepresentationSet(tuple(reps.reps[i] for i in idx), replace(reps.source, sequences=rows),
                             reps.descriptor, reps.input_type)


class _SpecEncoder(_Plugin):
    """§7 encoder: `output_type` + `encode(set)`. The short-sample policy is not passed;
    `drop` is applied before encoding, the others are the encoder's own business."""

    def encode(self, source, short_samples):
        return self._inner.encode(source)


class _SpecWarping(_Plugin):
    """§7 warping: `domain`, `__call__(x)`, `gap(raw, n)`. SE by central difference."""
    n: int | None = None  # set by the Measure that resolves it: gap() needs the set size

    def apply(self, estimate):
        f, x = self._inner, estimate.value
        ends = [f(v) for v in (estimate.ci_low, estimate.ci_high) if v is not None]
        low, high = (min(ends), max(ends)) if len(ends) == 2 else (None, None)
        se = None
        if estimate.se is not None:
            h = 1e-6 * max(1., abs(x))
            se = abs(f(x + h) - f(x - h)) / (2 * h) * estimate.se
        bias = float(self._inner.gap(x, self.n)) if hasattr(self._inner, "gap") and self.n else None
        return f(x), se, low, high, bias


def adapt(kind, component, name, version="1"):
    """Wrap a §7-shaped distance/statistic/encoder/warping in the library protocol.
    Components that already implement it are returned unchanged."""
    if kind not in ("distance", "statistic", "encoder", "warping"):
        return component
    descriptor = {kind: name, "version": version, "protocol": "spec-7"}
    if kind == "statistic" and not hasattr(component, "estimate"):
        return _SpecStatistic(component, descriptor=descriptor)
    if kind == "encoder":
        import inspect
        if len(inspect.signature(component.encode).parameters) == 1:
            units = {n for n, _ in _REGISTRY["unit"]}
            return _SpecEncoder(component, descriptor=descriptor, supported_units=units)
    if kind == "warping" and not hasattr(component, "apply"):
        return _SpecWarping(component, descriptor=descriptor)
    defaults: dict[str, Any] = {"descriptor": descriptor}
    if kind == "distance":
        # Undeclared properties are not assumed: energy refuses a distance that
        # doesn't declare itself a negative-type dissimilarity.
        defaults.update(negative_type=False, kind="dissimilarity")
    if all(hasattr(component, k) for k in defaults):
        return component
    return _Plugin(component, **defaults)


def register_encoder(name, **kw):
    return register("encoder", name, **kw)


def register_distance(name, **kw):
    return register("distance", name, **kw)


def register_statistic(name, **kw):
    return register("statistic", name, **kw)


def register_warping(name, **kw):
    return register("warping", name, **kw)


def register_unit(name, **kw):
    return register("unit", name, **kw)


def register_sampler(name, **kw):
    return register("sampler", name, **kw)


def unit_boundaries():
    """{unit name: the boundary detector it requires} over every registered unit."""
    return {name: resolve("unit", name, version=version).boundary
            for name, version in _REGISTRY["unit"]}


class _BuiltinUnit:
    def __init__(self, name, boundary, detect):
        self.boundary, self.detect = boundary, detect
        self.descriptor = {"unit": name, "boundary": boundary, "version": "1"}


def _builtin_units():
    # Deferred: alignment resolves units through this registry.
    from uncertain.alignment import word_boundaries, sentence_boundaries
    return (("token", "token:v1", None), ("word", "whitespace:v1", word_boundaries),
            ("sentence", "punctuation:v1", sentence_boundaries))


for _name, _boundary, _detect in _builtin_units():
    register_unit(_name)(partial(_BuiltinUnit, _name, _boundary, _detect))

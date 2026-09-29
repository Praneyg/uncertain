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
    component = factory(**dict(config or {}), **runtime)
    if not hasattr(component, "descriptor"):
        raise ConfigError(f"{kind} {name} must declare a stable descriptor")
    canonical_json(component.descriptor)
    return component


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

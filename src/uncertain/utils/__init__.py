"""Canonical identity and random streams; never use Python hash for provenance."""
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from collections.abc import Mapping
import numpy as np


def canonical(value):
    if is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot canonically serialize {type(value).__name__}")


def canonical_json(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def seed_for(*parts):
    return int(digest(parts)[:16], 16)


def immutable_array(value, dtype):
    """Immutable bytes backing prevents callers re-enabling the writable flag."""
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


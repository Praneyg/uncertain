"""Versioned content-addressed alternatives with atomic, non-pickle persistence."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import tempfile
import time
from collections import OrderedDict
from uncertain.core import AlternativeSet, Sequence, UnitSpec
from uncertain.utils import canonical, canonical_json, digest


class StoreError(RuntimeError):
    pass


def read_bytes_retry(path):
    for attempt in range(6):
        try:
            return Path(path).read_bytes()
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.01 * 2**attempt)
    raise AssertionError("unreachable")


def publish_immutable(path, content):
    """Create a complete immutable artifact exactly once, never replace a reader's file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        try:
            # A same-filesystem hard link publishes complete bytes atomically.
            # Unlike replacing an existing path, this cannot race existing readers.
            os.link(temp, path)
        except FileExistsError:
            if read_bytes_retry(path) != content:
                raise StoreError(f"immutable artifact conflict: {path}") from None
    finally:
        os.unlink(temp)


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(6):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                # Windows may refuse replacement while another reader has the
                # destination open. Identical concurrent publication is success.
                if path.exists() and read_bytes_retry(path) == content:
                    return
                if attempt == 5:
                    raise
                time.sleep(.01 * 2**attempt)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def deserialize(payload):
    d = dict(payload)
    d["sequences"] = tuple(Sequence(**s) for s in d["sequences"])
    d["unit"] = UnitSpec(**d["unit"])
    return AlternativeSet(**d)


class AlternativeStore:
    schema_version = 1

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._sets = {}
        self._requests = {}
        self.hits = 0
        self.misses = 0

    def save(self, alternatives):
        key = alternatives.content_hash
        self._sets[key] = alternatives
        if self.path is not None:
            path = self.path / "alternatives" / (key+".json")
            if path.exists():
                self.load(key)  # fail explicitly on existing corruption
            else:
                data = {"schema_version": self.schema_version, "content_hash": key,
                        "payload": canonical(alternatives)}
                publish_immutable(path, canonical_json(data).encode("utf-8"))
        return key

    def load(self, key):
        if not len(key) == 64 or any(c not in "0123456789abcdef" for c in key):
            raise StoreError("invalid artifact identity")
        if self.path is None:
            return self._sets[key]
        try:
            envelope = json.loads(read_bytes_retry(self.path/"alternatives"/(key+".json")))
            if envelope["schema_version"] != self.schema_version:
                raise StoreError("unsupported alternatives schema version")
            if envelope["content_hash"] != key or digest(envelope["payload"]) != key:
                raise StoreError("artifact checksum mismatch")
            value = deserialize(envelope["payload"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreError(f"Cannot load alternatives {key}: {exc}") from exc
        self._sets[key] = value
        return value

    def _candidates(self, stream, n):
        candidates = {h: key for (s, size, h), key in self._requests.items() if s == stream and size == n}
        if self.path is not None:
            for path in (self.path/"requests"/stream).glob(f"{n}-*.json"):
                try:
                    row = json.loads(read_bytes_retry(path))
                    if row["stream"] != stream or row["n"] != n:
                        raise StoreError("request index identity mismatch")
                    candidates[row["horizon"]] = row["artifact"]
                except (ValueError, KeyError, OSError) as exc:
                    raise StoreError(f"corrupt request index: {path}") from exc
        return candidates

    def available(self, sampler, context, horizon, spec, unit):
        stream = sampler.stream_key(context, spec, unit)
        return any(h == horizon or (sampler.prefix_consistent and h >= horizon)
                   for h in self._candidates(stream, spec.n))

    def get_or_sample(self, sampler, context, horizon, spec, unit):
        stream = sampler.stream_key(context, spec, unit)
        candidates = self._candidates(stream, spec.n)
        compatible = sorted(h for h in candidates if h == horizon or
                            (sampler.prefix_consistent and h >= horizon))

        def stored(h):
            value = self.load(candidates[h])
            if (value.provenance.get("stream") != stream or value.context != tuple(context)
                    or value.unit != unit or len(value) != spec.n or value.horizon != h):
                raise StoreError("request index points to incompatible alternatives")
            return value

        if compatible:
            self.hits += 1
            value = stored(compatible[0])
            return value if value.horizon == horizon else value.truncate(horizon)
        self.misses += 1
        shorter = [h for h in candidates if h < horizon]
        value = None
        if shorter and sampler.prefix_consistent and hasattr(sampler, "deepen"):
            # Same stream, shorter horizon: resume its rows instead of redrawing them.
            value = sampler.deepen(stored(max(shorter)), horizon)
        if value is None:
            value = sampler.sample(context, horizon, spec, unit)
        key = self.save(value)
        self._requests[(stream, spec.n, horizon)] = key
        if self.path is not None:
            request = dict(stream=stream, n=spec.n, horizon=horizon, artifact=key)
            publish_immutable(self.path/"requests"/stream/f"{spec.n}-{horizon}.json",
                         canonical_json(request).encode("utf-8"))
        return value


@dataclass
class RunMemo:
    """Bounded count of derived encodings, never persisted."""
    max_entries: int = 32

    def __post_init__(self):
        if self.max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.values = OrderedDict()

    def get_or_compute(self, key, fn):
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        value = fn()
        self.values[key] = value
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return value

    def clear(self):
        self.values.clear()

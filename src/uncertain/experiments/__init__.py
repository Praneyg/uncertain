"""Sweeps: one base Measure over a grid of axis values, planned so each sample stream
is drawn once at its largest horizon, then written as a tidy table plus manifest."""
from dataclasses import dataclass
from functools import cache
from itertools import product
from pathlib import Path
import csv
import json
import math
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError
from uncertain.core import ConfigError, StatisticSpec
from uncertain.utils import canonical, digest
from uncertain.store import atomic_write, RunMemo
from uncertain._version import __version__


def git_sha():
    """None outside a repository; provenance stops at the library version there."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def json_safe(value):
    value = canonical(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


@dataclass
class SweepResult:
    """One row per (stimulus, cell); `manifest` records specs, environment and artifacts."""
    rows: list[dict]
    manifest: dict

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install uncertain-altsets[tables] for dataframe export") from exc
        frame = pd.DataFrame(self.rows)
        # Explicitly nullable, even when every datum is absent.
        for name in ("datum", "se", "ci_low", "ci_high", "corrected_value", "warping_bias"):
            if name in frame:
                frame[name] = frame[name].astype("Float64")
        return frame

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            import io
            out = io.StringIO(newline="")
            if self.rows:
                writer = csv.DictWriter(out, fieldnames=list(self.rows[0]))
                writer.writeheader()
                for row in self.rows:
                    safe = json_safe(row)
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                                     else v for k,v in safe.items()})
            atomic_write(path, out.getvalue().encode("utf-8"))
        elif path.suffix == ".parquet":
            df = self.to_dataframe()
            for key in ("diagnostics", "artifact_ids"):
                if key in df:
                    df[key] = df[key].map(lambda x: json.dumps(json_safe(x), sort_keys=True))
            df.to_parquet(path, index=False)
        elif path.suffix == ".json":
            atomic_write(path, json.dumps(json_safe(self.rows), indent=2, ensure_ascii=False).encode("utf-8"))
        else:
            raise ValueError("output must be CSV, Parquet or JSON")
        atomic_write(path.with_suffix(".manifest.json"),
                     json.dumps(json_safe(self.manifest), indent=2, sort_keys=True).encode("utf-8"))
        return path


class Sweep:
    """`Sweep(measure, horizon=[1, 2], warping=[...])` (product grid) or `cells=[{...}]`."""

    def __init__(self, base, *, cells=None, **axes):
        allowed = {"horizon", "representation", "level", "distance", "distance_config",
                   "statistic", "warping"}
        if set(axes)-allowed:
            raise ConfigError(f"unsupported sweep axes: {sorted(set(axes)-allowed)}; one unit per run")
        if cells is not None and axes:
            raise ConfigError("supply cells or axes, not both")
        self.base = base
        configurations = cells if cells is not None else [dict(zip(axes, values, strict=True))
                          for values in product(*(axes[k] for k in axes))]
        self.measures, errors = [], []
        for i, config in enumerate(configurations):
            changes = dict(config)  # MeasureSpec normalizes string axis values, level included
            if isinstance(changes.get("statistic"), dict):
                changes["statistic"] = StatisticSpec(**changes["statistic"])
            if "unit" in changes and changes["unit"] != base.spec.unit:
                raise ConfigError("all cells must use the same unit")
            try:
                self.measures.append(base.replace(**changes))
            except (ConfigError, TypeError) as exc:
                errors.append(f"cell {i} {config}: {exc}")
        if errors:
            raise ConfigError("invalid sweep:\n"+"\n".join(errors))
        if not self.measures:
            raise ConfigError("sweep must contain at least one cell")
        hashes = [m.spec_hash for m in self.measures]
        if len(hashes) != len(set(hashes)):
            raise ConfigError("duplicate measure cells")

    def plan(self, stimuli):
        stimuli = tuple(stimuli)
        requests: dict[tuple, tuple] = {}
        for m in self.measures:
            for item in stimuli:
                for context, horizon in m.requirements(item):
                    stream = m.sampler.stream_key(context, m.spec.sampling, m.spec.unit)
                    key = (stream, m.spec.sampling.n, None if m.sampler.prefix_consistent else horizon)
                    if key not in requests or horizon > requests[key][2]:
                        requests[key] = (m, context, horizon)
        return SweepPlan(tuple(self.measures), stimuli, tuple(requests.values()))

    def run(self, stimuli):
        return self.plan(stimuli).run()


@dataclass
class SweepPlan:
    """Validated cells x stimuli plus the deduplicated sample requests; `run()` executes."""
    measures: tuple
    stimuli: tuple
    requests: tuple

    def describe(self):
        return {"cells": len(self.measures), "items": len(self.stimuli),
                "rows": len(self.measures)*len(self.stimuli),
                "sample_requests": len(self.requests),
                "anticipated_nested_work": any(m.spec.mode == "anticipatory" for m in self.measures),
                "cache_hits": sum(m.store.available(m.sampler, c, h, m.spec.sampling, m.spec.unit)
                                  for m,c,h in self.requests),
                "horizons": [h for _,_,h in self.requests]}

    def run(self):
        memo = RunMemo()
        for m, context, horizon in self.requests:
            m.store.get_or_sample(m.sampler, context, horizon, m.spec.sampling, m.spec.unit)
        run_id = digest({"specs": [m.spec_hash for m in self.measures], "stimuli": self.stimuli})
        rows = []
        try:
            for item in self.stimuli:
                for m in self.measures:
                    previous = m.memo
                    m.memo = memo
                    try:
                        r = m.compute(item)
                    finally:
                        m.memo = previous
                    k = len(item.target) if m.spec.unit.name == "token" else len(
                        item.observed(m.sampler.backend, m.spec.unit).units)
                    rows.append(dict(
                        run_id=run_id, stimulus_id=item.stimulus_id, document_id=item.document_id,
                        target_start=item.position, target_end=item.position+k, unit=m.spec.unit.name,
                        context_hash=digest(item.context), transform_id=item.transform_id,
                        original_id=item.original_id, datum=item.datum,
                        spec_hash=r.spec_hash, horizon=m.spec.horizon,
                        model_id=m.spec.sampling.model.model_id, model_revision=m.spec.sampling.model.revision,
                        representation=m.spec.representation.name, distance=m.spec.distance,
                        statistic=m.spec.statistic.name, warping=m.spec.warping,
                        short_samples=m.spec.short_samples, probability=m.spec.probability,
                        mode=m.spec.mode, value=r.value, raw_value=r.raw_value,
                        corrected_value=r.corrected_value, se=r.se, ci_low=r.ci_low, ci_high=r.ci_high,
                        confidence=r.confidence, n_requested=r.n_requested,
                        n_a_used=r.n_a_used, n_b_used=r.n_b_used, method=r.method, status=r.status,
                        artifact_ids=r.artifact_ids, warping_bias=r.warping_bias,
                        diagnostics=canonical(r.diagnostics)))
        finally:
            memo.clear()
        environment = {"python": platform.python_version(), "platform": platform.platform()}
        for package in ("numpy", "torch", "transformers", "pandas", "pyarrow"):
            try:
                environment[package] = version(package)
            except PackageNotFoundError:
                pass
        manifest = {"schema_version": 1, "engine": __version__, "run_id": run_id,
                    "git_sha": git_sha(), "environment": environment,
                    "seeds": sorted({m.spec.sampling.seed for m in self.measures}),
                    "specifications": {
                        m.spec_hash: {"spec": canonical(m.spec), "backend": canonical(m.sampler.backend.descriptor),
                                      "sampler": canonical(m.sampler.descriptor),
                                      "components": [canonical(c.descriptor) for c in
                                                     (m.encoder, m.distance, m.statistic, m.warping)]}
                        for m in self.measures},
                    "artifacts": sorted({key for row in rows for key in row["artifact_ids"]}),
                    "plan": self.describe()}
        return _tidy(SweepResult(rows, manifest))


def _tidy(result):
    """§5: ``df = grid.run(...)  # tidy``. With pandas installed the result is a DataFrame
    that still carries rows, manifest and write(); without it, the dependency-free SweepResult."""
    try:
        import pandas as pd
    except ImportError:
        return result
    frame = _sweep_frame(pd)(result.to_dataframe())
    frame.rows, frame.manifest = result.rows, result.manifest
    return frame


@cache
def _sweep_frame(pd):
    class SweepFrame(pd.DataFrame):  # type: ignore[name-defined]
        _metadata = ["rows", "manifest"]

        @property
        def _constructor(self):
            return pd.DataFrame  # slices and derived frames are plain DataFrames

        def to_dataframe(self):
            return self.copy()

        def write(self, path):
            return SweepResult(self.rows, self.manifest).write(path)
    return SweepFrame

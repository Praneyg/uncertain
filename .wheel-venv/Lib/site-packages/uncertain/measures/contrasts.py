"""Contrasts preserve aligned bootstrap draws when measures share alternatives."""
from dataclasses import dataclass
import math
import numpy as np
from uncertain.core import ConfigError
from uncertain.statistics import normal_interval


@dataclass(frozen=True)
class ContrastResult:
    value: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    status: str
    left_spec_hash: str
    right_spec_hash: str


def contrast(left, right):
    """Compute left-right; intervals are approximate, and never assume shared draws independent.

    Pass results from the same engine's deterministic bootstrap settings. Dropped
    subsets of a shared stream must have identical IDs to align resampling.
    """
    if left.confidence != right.confidence:
        raise ConfigError("contrasts require the same confidence level")
    for side_a in ("a", "b"):
        for side_b in ("a", "b"):
            key_a, key_b = f"resampling_seed_{side_a}", f"resampling_seed_{side_b}"
            if key_a in left.diagnostics and left.diagnostics[key_a] == right.diagnostics.get(key_b):
                if left.diagnostics.get(f"sample_ids_{side_a}") != right.diagnostics.get(f"sample_ids_{side_b}"):
                    raise ConfigError("shared streams with different retained samples require joint resampling")
    value = left.value-right.value
    if not math.isfinite(value):
        return ContrastResult(value, None, None, None, "nonfinite_contrast", left.spec_hash, right.spec_hash)
    if left.method == "exact" and right.method == "exact":
        return ContrastResult(value, 0., value, value, "exact", left.spec_hash, right.spec_hash)
    if left.se is None or right.se is None:
        return ContrastResult(value, None, None, None, "joint_uncertainty_unavailable",
                              left.spec_hash, right.spec_hash)
    a, b = np.array(left.resamples), np.array(right.resamples)
    if left.method == "exact" and len(b):
        a = np.full(len(b), left.value)
    if right.method == "exact" and len(a):
        b = np.full(len(a), right.value)
    if len(a) and len(a) == len(b) and np.all(np.isfinite(a-b)):
        values = a-b
        se = float(values.std(ddof=1))
        low, high = normal_interval(value, se, left.confidence)
        status = "paired_bootstrap_approximate"
    else:
        se, low, high, status = None, None, None, "joint_uncertainty_unavailable"
    return ContrastResult(value, se, low, high, status, left.spec_hash, right.spec_hash)

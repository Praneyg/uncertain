from dataclasses import dataclass, field
from collections.abc import Mapping
from uncertain.core.specs import freeze


@dataclass(frozen=True)
class StatisticEstimate:
    value: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    status: str = "ok"
    bootstrap_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class Result:
    value: float
    raw_value: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    n_requested: int
    n_a_used: int
    n_b_used: int
    spec_hash: str
    method: str
    status: str
    confidence: float
    artifact_ids: tuple[str, ...] = ()
    warping_bias: float | None = None
    corrected_value: float | None = None
    diagnostics: Mapping = field(default_factory=dict)
    resamples: tuple[float, ...] = field(default=(), repr=False, compare=True)

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))

    @property
    def n(self):
        return self.n_a_used

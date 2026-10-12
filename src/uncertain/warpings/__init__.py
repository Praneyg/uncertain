import math
from uncertain.core import ConfigError
from uncertain.registry import register_warping


@register_warping("identity")
class Identity:
    descriptor = {"warping": "identity", "version": "1"}
    domain = (-math.inf, math.inf)

    def apply(self, estimate):
        return estimate.value, estimate.se, estimate.ci_low, estimate.ci_high, 0.


def neglog(x):
    if x < 0 or math.isnan(x):
        raise ValueError("negative log requires a nonnegative input")
    return -math.log(x) if x else math.inf


@register_warping("neglog")
class NegLog:
    descriptor = {"warping": "neglog", "version": "1", "zero": "infinite", "base": "e"}
    domain = (0., math.inf)

    def apply(self, estimate):
        x = estimate.value
        if x < 0:
            raise ConfigError("neglog received a negative statistic")
        low = neglog(estimate.ci_high) if estimate.ci_high is not None else None
        # A normal-approximation interval can dip below 0; the statistic cannot.
        high = neglog(max(estimate.ci_low, 0.)) if estimate.ci_low is not None else None
        se = estimate.se/x if x and estimate.se is not None else None
        if not x and estimate.se is not None:
            # -log 0 = inf. Every resample at 0 => no spread; any resample above 0 => unbounded.
            se = 0. if estimate.se == 0 else math.inf
        bias = estimate.se**2/(2*x*x) if x and estimate.se is not None else None
        return neglog(x), se, low, high, bias

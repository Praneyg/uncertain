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
        high = neglog(estimate.ci_low) if estimate.ci_low is not None else None
        se = estimate.se/x if x and estimate.se is not None else None
        bias = estimate.se**2/(2*x*x) if x and estimate.se is not None else None
        return neglog(x), se, low, high, bias

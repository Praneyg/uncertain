"""Set statistics with whole-alternative resampling, never independent pair errors.

A statistic reduces two RepresentationSets (A, B) under a distance to a
StatisticEstimate. Built-ins: `mean` (E d(a, b)), `energy` (2E d(a,b) - E d(a,a') -
E d(b,b'), V or U form) and `min` (directed nearest-neighbour). Each declares
`range_for(distance)` for the warping-domain gate and `truncation_bound` for §4.7.
Bootstrap replicates reweight whole alternatives on each side with seeded
multinomial weights, so cells sharing a stream share resamples.
"""
from statistics import NormalDist
from dataclasses import replace
import numpy as np
from uncertain.core import StatisticEstimate, ConfigError
from uncertain.registry import register_statistic
from uncertain.store import RunMemo
from uncertain.utils import canonical_json


def normal_interval(value, se, confidence):
    """Two-sided normal-approximation interval; (None, None) without an SE."""
    if se is None:
        return None, None
    z = NormalDist().inv_cdf((1+confidence)/2)
    return value-z*se, value+z*se


def sample_mean(values, confidence):
    """Mean with its iid standard error and normal interval."""
    values = np.asarray(values, dtype=float)
    value = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None
    low, high = normal_interval(value, se, confidence)
    return StatisticEstimate(value, se, low, high, "ok" if se is not None else "insufficient_samples")


def binomial(hits, n, confidence):
    """Wilson interval remains informative at both boundaries."""
    p = hits/n
    z = NormalDist().inv_cdf((1+confidence)/2)
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    rad = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    low = 0. if hits == 0 else max(0., mid-rad)
    high = 1. if hits == n else min(1., mid+rad)
    se = np.sqrt(p*(1-p)/(n-1)) if n > 1 and 0 < hits < n else None
    status = "ok" if se is not None else ("zero_hits" if not hits else "boundary_probability")
    return StatisticEstimate(float(p), None if se is None else float(se), low, high, status)


def _span(distance):
    return float(distance.range[1])-float(distance.range[0])


def pairwise_blocks(a, b, distance, block_size=64):
    """Yield (i, j, block) tiles of d(a, b), checked finite and inside distance.range."""
    for i in range(0, len(a), block_size):
        for j in range(0, len(b), block_size):
            x, y = a[i:i+block_size], b[j:j+block_size]
            block = np.array([[distance(u, v) for v in y] for u in x], dtype=float)
            if not np.all(np.isfinite(block)):
                raise ValueError("distance returned non-finite values")
            if np.any(block < distance.range[0]-1e-10) or np.any(block > distance.range[1]+1e-10):
                raise ValueError("distance returned a value outside its declared range")
            yield i, j, block


def pairwise(a, b, distance):
    """Dense matrix; the three built-in estimators stream blocks, `lifted` needs it whole."""
    out = np.empty((len(a.reps), len(b.reps)))
    for i,j,block in pairwise_blocks(a.reps, b.reps, distance):
        out[i:i+block.shape[0], j:j+block.shape[1]] = block
    return out


def weights(source, replicates, seed):
    """(replicates, n) bootstrap multiplicities; all ones for a deterministic operand."""
    n = len(source)
    if source.deterministic:
        return np.ones((replicates, n))
    # Same source and seed across sweep cells => same resample index multiplicities.
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1/n), size=replicates).astype(float)


def weighted_mean(a, b, distance, wa, wb, block_size):
    """Mean of d over A x B, plus its value under each bootstrap weight pair."""
    total = 0.
    bootstrap = np.zeros(len(wa))
    for i,j,block in pairwise_blocks(a.reps, b.reps, distance, block_size):
        total += block.sum()
        bootstrap += np.einsum("ri,ij,rj->r", wa[:, i:i+block.shape[0]], block,
                               wb[:, j:j+block.shape[1]], optimize=True)
    scale = len(a.reps)*len(b.reps)
    return float(total/scale), bootstrap/scale


# ponytail: process-global bounded LRU, not per-run. Upgrade path is threading the
# caller's RunMemo through estimate() if cross-run staleness ever matters.
_WITHIN_BLOCKS = RunMemo(max_entries=32)


def within_block(s, distance, w, block_size, seed, replicates):
    """d(s,s) is shared by every comparison involving s, and by s as its own continuation.

    The key is every input weighted_mean reads: s.content_hash covers the ordered source
    sequences and the encoder descriptor (so reps are fixed), the distance descriptor covers
    the callable, and (seed, replicates) with the source's length and determinism -- both
    inside content_hash -- fix `w` exactly.
    """
    key = (s.content_hash, canonical_json(distance.descriptor), block_size, seed, replicates)

    def compute():
        value, boot = weighted_mean(s, s, distance, w, w, block_size)
        boot.flags.writeable = False  # a shared entry must not be mutated in place
        return value, boot
    return _WITHIN_BLOCKS.get_or_compute(key, compute)


@register_statistic("mean")
class Mean:
    supports_uncertainty = True

    def __init__(self, block_size=64):
        if type(block_size) is not int or block_size < 1:
            raise ConfigError("block_size must be a positive integer")
        self.block_size = block_size
        self.descriptor = {"statistic": "mean", "version": "1", "block_size": block_size}

    def range_for(self, distance):
        """§4.5's gate calls this; a mean of in-range values stays in range."""
        return float(distance.range[0]), float(distance.range[1])

    def estimates_probability(self, distance):
        """A mean of a [0,1] similarity is a probability: exactly for an indicator such as
        prefix_indicator, similarity-weighted for a soft one. Energy and min are not."""
        return (type(self) is Mean and distance.kind == "similarity"
                and 0 <= distance.range[0] and distance.range[1] <= 1)

    def truncation_bound(self, open_a, open_b, distance):
        """§4.7: the most the statistic can move if the draws still open at H (fractions
        open_a, open_b; an EOS stop is complete) were continued. Only pairs touching an
        open draw move, each by at most the distance's span."""
        return _span(distance)*(1-(1-open_a)*(1-open_b))

    def estimate(self, a, b, distance, estimator, seeds):
        if (distance.descriptor.get("distance") == "prefix_indicator" and
                all(x[:distance.prefix_length] == b.reps[0][:distance.prefix_length] for x in b.reps)):
            values = [distance(x, b.reps[0]) for x in a.reps]
            result = binomial(sum(values), len(values), estimator.confidence)
            wa = weights(a.source, estimator.bootstrap, seeds[0])
            return replace(result, bootstrap_values=tuple(wa @ values / len(values)))
        if b.source.deterministic:
            values = [distance(x, b.reps[0]) for x in a.reps]
            if distance.descriptor.get("distance") == "prefix_indicator":
                return binomial(sum(values), len(values), estimator.confidence)
            if a.source.deterministic:
                return StatisticEstimate(values[0], 0., values[0], values[0])
            result = sample_mean(values, estimator.confidence)
            wa = weights(a.source, estimator.bootstrap, seeds[0])
            return replace(result, bootstrap_values=tuple(wa @ values / len(values)))
        wa = weights(a.source, estimator.bootstrap, seeds[0])
        wb = weights(b.source, estimator.bootstrap, seeds[1])
        value, boots = weighted_mean(a, b, distance, wa, wb, self.block_size)
        return bootstrap_result(value, boots, a, b, estimator)


def bootstrap_result(value, boots, a, b, estimator):
    """Bootstrap SE and percentile interval around a point value."""
    if any(len(s) < 2 and not s.deterministic for s in (a.source, b.source)):
        return StatisticEstimate(value, None, None, None, "insufficient_samples")
    se = float(np.std(boots, ddof=1))
    tail = (1-estimator.confidence)/2
    low = float(np.quantile(boots, tail))
    high = float(np.quantile(boots, 1-tail))
    return StatisticEstimate(value, se, float(low), float(high), "bootstrap_approximate",
                             tuple(float(x) for x in boots))


@register_statistic("energy")
class Energy(Mean):

    def __init__(self, variant="v", block_size=64):
        super().__init__(block_size)
        if variant not in {"v", "u"}:
            raise ConfigError("energy variant must be v or u")
        self.variant = variant
        self.descriptor = {"statistic": "energy", "version": "1", "variant": variant,
                           "block_size": block_size}

    def range_for(self, distance):
        # V >= 0 always. U removes the diagonal and is unbiased, so it dips below 0 when
        # the sets agree: each within term is a mean over distinct pairs, at most D.
        high = 2*float(distance.range[1])
        return (-high if self.variant == "u" else 0.), high

    def truncation_bound(self, open_a, open_b, distance):
        # 2*cross - within_A - within_B. A within mean moves on at most 2*open of its
        # pairs (union bound; also holds for U's off-diagonal mean).
        cross = 1-(1-open_a)*(1-open_b)
        return _span(distance)*(2*cross + min(1., 2*open_a) + min(1., 2*open_b))

    def estimate(self, a, b, distance, estimator, seeds):
        if not distance.negative_type or distance.kind != "dissimilarity":
            raise ConfigError("energy requires a declared negative-type dissimilarity")
        wa = weights(a.source, estimator.bootstrap, seeds[0])
        wb = weights(b.source, estimator.bootstrap, seeds[1])
        cross, bcross = weighted_mean(a, b, distance, wa, wb, self.block_size)
        within = []
        for s, w, seed in ((a, wa, seeds[0]), (b, wb, seeds[1])):
            if s.source.deterministic:
                within.append((0., np.zeros(estimator.bootstrap)))
                continue
            value, boot = within_block(s, distance, w, self.block_size, seed, estimator.bootstrap)
            if self.variant == "u":
                n = len(s.source)
                if n < 2:
                    raise ConfigError("U-energy requires >=2 draws on each random side")
                # d(x,x)=0; bootstrap duplicates still count as distinct sample positions.
                value, boot = value*n/(n-1), boot*n/(n-1)
            within.append((value, boot))
        value = 2*cross-within[0][0]-within[1][0]
        boots = 2*bcross-within[0][1]-within[1][1]
        result = bootstrap_result(float(value), boots, a, b, estimator)
        bound = distance.range[1]
        if np.isfinite(bound):
            # Bounded differences: replacing one draw changes energy by <=4D/n.
            # V has positive bias <=D/n + D/m; U is unbiased. Deterministic
            # operands contribute neither sampling sensitivity nor V bias.
            inverse_n = sum(1/len(x.source) for x in (a,b) if not x.source.deterministic)
            radius = np.sqrt(8*bound**2*inverse_n*np.log(2/(1-estimator.confidence)))
            bias_bound = bound*inverse_n if self.variant == "v" else 0.
            return replace(result, ci_low=max(self.range_for(distance)[0], value-radius-bias_bound),
                           ci_high=min(2*bound, value+radius),
                           status="bounded_energy_ci_bootstrap_se")
        # The ordinary bootstrap can fail at the equal-distribution degeneracy.
        # Preserve its SE approximation without claiming a valid population CI.
        return replace(result, ci_low=None, ci_high=None, status="unbounded_energy_ci_unavailable")


@register_statistic("min")
class DirectedMin(Mean):
    def __init__(self, block_size=64):
        super().__init__(block_size)
        self.descriptor = {"statistic": "min", "version": "1", "direction": "B_queries_A",
                           "block_size": block_size}

    def truncation_bound(self, open_a, open_b, distance):
        # Each b's minimum ranges over all of A: one open draw in A can move every term.
        return _span(distance)*(1. if open_a else open_b)

    def estimate(self, a, b, distance, estimator, seeds):
        matrix = pairwise(a, b, distance)
        value = float(matrix.min(axis=0).mean())
        # Whole-alternative bootstrap: a resample of A drops the rows it didn't draw, so
        # each b's minimum is taken over the drawn A rows only. Not a smooth mean, so the
        # SE is approximate (bootstrap_approximate), but it is never omitted.
        wa = weights(a.source, estimator.bootstrap, seeds[0])
        wb = weights(b.source, estimator.bootstrap, seeds[1])
        boots = np.array([np.where(ra[:, None] > 0, matrix, np.inf).min(axis=0) @ rb / len(rb)
                          for ra, rb in zip(wa, wb, strict=True)])
        return bootstrap_result(value, boots, a, b, estimator)


class Lifted:
    """§4.6's automatic lift: a factorable statistic is a reduction over the pairwise matrix.

    `reduction` must be range-preserving (mean, min, max, a quantile) -- `range_for` reports
    the distance's own range. A reduction that leaves it, such as a sum, needs its own class.
    """
    supports_uncertainty = False

    def __init__(self, distance, reduction, name=None):
        name = name or getattr(reduction, "__name__", "")
        if not name.isidentifier():
            # Two unnamed lambdas would otherwise share a descriptor and silently collide.
            raise ConfigError("lifted requires a named reduction or an explicit name=")
        self.distance, self.reduction = distance, reduction
        self.descriptor = {"statistic": f"lifted_{name}", "version": "1",
                           "reduction": name, "distance": dict(distance.descriptor)}

    def range_for(self, distance):
        return float(distance.range[0]), float(distance.range[1])

    def estimate(self, a, b, distance, estimator, seeds):
        if dict(distance.descriptor) != dict(self.distance.descriptor):
            raise ConfigError("lifted statistic is bound to a different distance")
        value = float(self.reduction(pairwise(a, b, distance)))
        return StatisticEstimate(value, None, None, None, "uncertainty_unavailable_lifted")


def lifted(distance, reduction, name=None):
    """A statistic that applies `reduction` to the full pairwise matrix (§4.6)."""
    return Lifted(distance, reduction, name)

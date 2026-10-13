"""Test-only plugin: one distance, statistic, encoder and warping, registered through the
public `uncertain.register_*` decorators only. No core module is imported or edited."""
import math
import numpy as np
import uncertain

DIST_NAME, STAT_NAME, ENC_NAME, WARP_NAME = "eval_l1", "eval_mean_len", "eval_lengths", "eval_affine"
SOURCE_STAT_NAME = STAT_NAME


@uncertain.register_distance(DIST_NAME)
class L1:
    input_type, range = "vector", (0.0, math.inf)
    kind, negative_type, symmetric = "dissimilarity", True, True
    descriptor = {"distance": DIST_NAME, "version": "1"}

    def __call__(self, a, b):
        return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


@uncertain.register_encoder(ENC_NAME)
class Lengths:
    output_type = "vector"
    supported_units = {"token", "word", "sentence"}
    descriptor = {"encoder": ENC_NAME, "version": "1"}

    def __init__(self, backend=None):
        pass

    def encode(self, source, short_samples):
        reps = tuple(np.array([len(s.tokens), len(s.units)], dtype=float) for s in source.sequences)
        return uncertain.RepresentationSet(reps, source, self.descriptor, self.output_type)


def source_text_statistic_value(alternatives):
    """Mean character length of each alternative's unit strings -- reads only A.source."""
    return float(np.mean([len("".join(s.units)) for s in alternatives.sequences]))


@uncertain.register_statistic(STAT_NAME)
class MeanLength:
    """Non-factoring: never builds a distance matrix; reads raw sequences from A.source."""
    supports_uncertainty = True
    descriptor = {"statistic": STAT_NAME, "version": "1"}

    def range_for(self, distance):
        return (0.0, math.inf)

    def estimate(self, a, b, distance, estimator, seeds):
        values = [len("".join(s.units)) for s in a.source.sequences]
        value = source_text_statistic_value(a.source)
        se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
        return uncertain.StatisticEstimate(value, se, None, None, "ok")


@uncertain.register_warping(WARP_NAME)
class Affine:
    descriptor = {"warping": WARP_NAME, "version": "1"}
    domain = (-math.inf, math.inf)

    def apply(self, e):
        return 2 * e.value + 1, None if e.se is None else 2 * e.se, None, None, 0.0

"""Built-in pointwise scores. Each declares `input_type` (what encoder it pairs with),
`range`, `kind` (similarity or dissimilarity), `negative_type` (required by energy)
and `symmetric`."""
import numpy as np
from uncertain.core import ConfigError
from uncertain.registry import register_distance


@register_distance("prefix_indicator")
class PrefixIndicator:
    input_type = "units"
    range = (0., 1.)
    kind = "similarity"
    negative_type = False
    symmetric = False

    def __init__(self, prefix_length=1):
        if type(prefix_length) is not int or prefix_length < 1:
            raise ConfigError("prefix_length must be a positive integer")
        self.prefix_length = prefix_length
        self.descriptor = {"distance": "prefix_indicator", "version": "1",
                           "prefix_length": prefix_length}

    def __call__(self, a, b):
        k = self.prefix_length
        return float(len(a) >= k and len(b) >= k and a[:k] == b[:k])


@register_distance("ngram_overlap")
class NgramOverlap:
    input_type = "units"
    range = (0., 1.)
    kind = "similarity"
    negative_type = False
    symmetric = True

    def __init__(self, ngram=1):
        if type(ngram) is not int or ngram < 1:
            raise ConfigError("ngram must be a positive integer")
        self.ngram = ngram
        self.descriptor = {"distance": "ngram_overlap", "version": "1", "ngram": ngram,
                           "normalization": "set_jaccard"}

    def __call__(self, a, b):
        n = self.ngram
        x, y = {a[i:i+n] for i in range(len(a)-n+1)}, {b[i:i+n] for i in range(len(b)-n+1)}
        return len(x & y) / len(x | y) if x or y else 1.


@register_distance("ngram_distance")
class NgramDistance(NgramOverlap):
    kind = "dissimilarity"
    negative_type = False  # no stronger guarantee asserted by this implementation

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.descriptor["distance"] = "ngram_distance"

    def __call__(self, a, b):
        return 1. - super().__call__(a, b)


@register_distance("discrete")
class Discrete:
    input_type = "units"
    range = (0., 1.)
    kind = "dissimilarity"
    negative_type = True
    symmetric = True
    descriptor = {"distance": "discrete", "version": "1"}

    def __call__(self, a, b):
        return float(a != b)


@register_distance("cosine")
class Cosine:
    input_type = "vector"
    range = (0., 2.)
    kind = "dissimilarity"
    negative_type = True
    symmetric = True
    descriptor = {"distance": "cosine", "version": "1", "zero_vector": "error",
                  "distribution_separating": False}

    def __call__(self, a, b):
        scale = np.linalg.norm(a)*np.linalg.norm(b)
        if scale == 0:
            raise ValueError("cosine distance is undefined for a zero vector")
        return float(np.clip(1-np.dot(a, b)/scale, 0, 2))


@register_distance("euclidean")
class Euclidean(Cosine):
    range = (0., float("inf"))
    descriptor = {"distance": "euclidean", "version": "1", "distribution_separating": True}

    def __call__(self, a, b):
        return float(np.linalg.norm(a-b))


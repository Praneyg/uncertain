"""Illustrative empirical entailment-cluster entropy, not a published preset.

All bidirectional edges are computed, then undirected connected components define
clusters. Transitive closure is explicit. The statistic reads source text and
does not reduce a pointwise distance matrix. Missing-cluster uncertainty is not
identifiable from this finite sample; return an explicit unavailable status.
"""
import math
from collections import Counter
from uncertain import StatisticEstimate


class EntailmentEntropy:
    supports_uncertainty = True

    def __init__(self, entails, *, model_id):
        self.entails = entails
        self.descriptor = {"statistic": "example:entailment_entropy", "version": "1",
                           "model_id": model_id, "clusters": "bidirectional_connected_components"}

    def range_for(self, distance):
        """Shannon entropy of the cluster partition: 0 to log(n), unbounded in n."""
        return 0., math.inf

    def estimate(self, a, b, distance, estimator, seeds):
        texts = a.reps
        parent = list(range(len(texts)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(texts)):
            for j in range(i):
                if self.entails(texts[i], texts[j]) and self.entails(texts[j], texts[i]):
                    parent[root(i)] = root(j)
        counts = Counter(root(i) for i in range(len(texts)))
        p = [count/len(texts) for count in counts.values()]
        value = -sum(x*math.log(x) for x in p)
        return StatisticEstimate(value, None, None, None, "cluster_uncertainty_unavailable")


class StringMismatch:
    input_type = "string"
    range = (0., 1.)
    kind = "dissimilarity"
    negative_type = True
    symmetric = True
    descriptor = {"distance": "example:string_mismatch", "version": "1"}

    def __call__(self, a, b):
        return float(a != b)

"""The library's configurable scientific API; runs offline on a toy model."""
from uncertain import Measure, ToyBackend, Stimulus, Sampler, register_distance


@register_distance("example_soft_similarity")
class SoftSimilarity:
    """Illustrative graded similarity; substitute a meaningful registered scorer."""
    input_type = "units"
    kind = "similarity"
    range = (0., 1.)
    negative_type = False
    descriptor = {"distance": "example_soft_similarity", "version": "1"}

    def __call__(self, a, b):
        return 1. if a == b else .2


def main():
    backend = ToyBackend()  # P(a)=0.75, P(b)=0.25
    target = Stimulus(target=(1,))
    measure = Measure.from_config(
        backend,
        unit="token",
        horizon=1,
        representation="tokens",
        distance="discrete",
        statistic="energy",
        warping="identity",
        estimator="auto",
        sampler=Sampler(backend),
        n=512,
        seed=42,
        generation={"temperature": 1., "top_k": 0, "top_p": 1.},
        short_samples="compare_ragged",
    )
    result = measure.compute(target)
    print(f"Token / H=1 energy: {result.value:.6f}, method={result.method}")
    assert result.method == "exact" and measure.sampler.calls == 0

    # Same definition over longer alternatives automatically selects Monte Carlo.
    result = measure.replace(horizon=3).compute(target)
    print(f"Token / H=3 energy: {result.value:.6f}, method={result.method}, SE={result.se}")
    assert result.method == "mc"

    # A user-defined similarity followed by negative log: sum over the entire
    # next-token vocabulary, with probability weights, rather than drawing samples.
    adjusted = Measure.from_config(backend, distance="example_soft_similarity",
                                   statistic="mean", warping="neglog", n=512, short_samples="compare_ragged")
    result = adjusted.compute(target)
    print(f"Similarity-adjusted surprisal: {result.value:.6f}, method={result.method}")
    assert result.method == "exact" and adjusted.sampler.calls == 0


if __name__ == "__main__":
    main()

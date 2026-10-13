"""Category G: do the §5 API examples run as written? (model name swapped for a local model)."""
import pytest

from uncertain import HFSampler, Measure, Sweep, Stimulus

KW = dict(unit="token", representation="layer:2", distance="cosine", statistic="energy",
          warping="identity", short_samples="compare_ragged")


def _measure(backend):
    return Measure.from_config(backend, n=4, horizon=2, **KW)


@pytest.mark.req("35")
def test_spec_hfsampler_signature(tiny_gpt2):
    # §5: HFSampler("gpt2", n=32, horizon=5, top_k=0, seed=0)
    HFSampler(tiny_gpt2, n=4, horizon=2, top_k=0, seed=0)


@pytest.mark.req("35")
def test_spec_measure_keyword_constructor(tiny_gpt2):
    # §5: Measure(sampler=..., unit=..., representation=..., distance=..., statistic=...,
    #             warping=..., short_samples=...). Requirement 35: n comes from the sampler,
    # HFSampler(..., n=, horizon=, top_k=0, seed=0); with no n anywhere §4.4 still refuses.
    m = Measure(sampler=HFSampler(tiny_gpt2, n=4, horizon=2, top_k=0, seed=0), **KW)
    assert m.spec.sampling.n == 4 and m.spec.horizon == 2 and m.spec.distance == "cosine"
    assert m.spec_hash == _measure(tiny_gpt2).spec_hash
    with pytest.raises(TypeError, match="'n'"):
        Measure(sampler=HFSampler(tiny_gpt2), **KW)


@pytest.mark.req("35")
def test_spec_replace_to_surprisal(tiny_gpt2):
    m = _measure(tiny_gpt2)
    s = m.replace(horizon=1, unit="token", level="tokens", distance="prefix_indicator",
                  statistic="mean", warping="neglog")
    assert s.spec.warping == "neglog" and s.spec.representation.name == "tokens"


@pytest.mark.req("35", "9b")
def test_spec_custom_distance_injection(tiny_gpt2):
    # §5 verbatim shape: only input_type and range declared.
    class MyDistance:
        input_type, range = "vector", (0.0, 2.0)

        def __call__(self, a, b):
            return 0.0
    from uncertain import ConfigError
    m = _measure(tiny_gpt2)
    # The base measure uses energy, which §4.6 restricts to declared negative-type
    # dissimilarities; §5's MyDistance declares neither, so energy must refuse it ...
    with pytest.raises(ConfigError, match="negative-type"):
        m.replace(distance=MyDistance())
    # ... and any statistic without that requirement accepts the injected object.
    r = m.replace(distance=MyDistance(), statistic="mean").compute(Stimulus((1, 2), (3,)))
    assert r.value == 0.0


@pytest.mark.req("35")
def test_spec_sweep_returns_dataframe(tiny_gpt2, measure):
    m = _measure(tiny_gpt2)
    grid = Sweep(base=m, horizon=[1, 2], level=["layer:0", "layer:2"], distance=["cosine"],
                 statistic=["energy", "mean"])
    out = grid.run([Stimulus((1, 2), (3,))])
    measure("returned_type", type(out).__name__)
    import pandas as pd
    assert isinstance(out, pd.DataFrame)


@pytest.mark.req("35")
def test_spec_literal_grid_is_type_consistent(tiny_gpt2, measure):
    # §5 literal grid: level ["layer:0","layer:8"] x distance ["cosine","ngram_overlap"]. Under §4.5
    # typing, layer x ngram_overlap is invalid -- the spec's own example conflicts with its own rule.
    from uncertain import ConfigError
    m = _measure(tiny_gpt2)
    with pytest.raises(ConfigError) as info:
        Sweep(base=m, horizon=[1, 2], level=["layer:0", "layer:2"], distance=["cosine", "ngram_overlap"],
              statistic=["energy", "mean"])
    measure("message_head", str(info.value).splitlines()[:2])

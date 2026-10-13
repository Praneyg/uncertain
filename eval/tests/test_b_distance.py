"""Category B: distance interface, registration, validation, axioms, shipped components."""
import importlib.util
import inspect
import math
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import uncertain
from uncertain import (AlternativeSet, ConfigError, EstimatorSpec, Measure, RepresentationSet,
                       Sequence, Stimulus, UnitSpec)
from uncertain.registry import resolve
from uncertain.statistics import Mean, Energy, DirectedMin

from conftest import make_measure
from toy_lm import ToyLM

EST = EstimatorSpec(method="mc", bootstrap=29)
ROOT = Path(__file__).resolve().parents[2]
words = st.lists(st.sampled_from(["a", "b", "c"]), min_size=0, max_size=4).map(tuple)


def rep_set(reps, deterministic=False, tag="x"):
    seqs = tuple(Sequence((0,), (-.1,), (-.1,), ("0",), ((0, 1),), sample_id=i) for i in range(len(reps)))
    src = AlternativeSet(seqs, (len(reps),), UnitSpec(), 1, deterministic=deterministic)
    return RepresentationSet(tuple(reps), src, {"enc": tag, "reps": repr(reps)}, "units")


# --------------------------------------------------------------------------- criterion 8
@pytest.mark.req("8")
@settings(max_examples=200, deadline=None)
@given(a_rows=st.lists(words, min_size=2, max_size=8), u=words,
       name=st.sampled_from(["discrete", "ngram_overlap", "ngram_distance", "prefix_indicator"]))
def test_set_unit_and_unit_unit_agree_with_pointwise(a_rows, u, name):
    d = resolve("distance", name)
    A = rep_set(a_rows, tag="A")
    U = rep_set([u], deterministic=True, tag="U")
    set_unit = Mean().estimate(A, U, d, EST, (1, 2)).value
    assert abs(set_unit - np.mean([d(x, u) for x in a_rows])) <= 1e-12
    a1 = rep_set([a_rows[0]], deterministic=True, tag="a1")
    assert abs(Mean().estimate(a1, U, d, EST, (1, 2)).value - d(a_rows[0], u)) <= 1e-12


@pytest.mark.req("8")
@pytest.mark.parametrize("name", ["discrete", "ngram_overlap", "ngram_distance", "prefix_indicator",
                                  "cosine", "euclidean"])
def test_distances_have_single_pointwise_signature(name):
    d = resolve("distance", name)
    params = list(inspect.signature(type(d).__call__).parameters)
    assert params == ["self", "a", "b"]
    assert not any("arity" in k for k in vars(type(d)))


# --------------------------------------------------------------------------- criterion 9
@pytest.mark.req("9")
def test_registered_components_resolve_and_run():
    import plugin_components as pc
    lm = ToyLM([.5, .3, .2])
    m = make_measure(lm, n=32, horizon=2, representation=pc.ENC_NAME, distance=pc.DIST_NAME,
                     statistic="mean", warping=pc.WARP_NAME)
    r = m.compute(Stimulus((0,), (1,)))
    assert math.isfinite(r.value) and r.se is not None
    assert type(m.encoder).__name__ == "Lengths" and type(m.distance).__name__ == "L1"
    assert type(m.warping).__name__ == "Affine"
    assert r.value == 2 * r.raw_value + 1
    m2 = make_measure(lm, n=32, horizon=2, representation=pc.ENC_NAME, distance=pc.DIST_NAME,
                      statistic=pc.STAT_NAME)
    assert math.isfinite(m2.compute(Stimulus((0,), (1,))).value)


@pytest.mark.req("9")
def test_plugin_module_touches_only_public_namespace():
    src = (Path(__file__).parent / "plugin_components.py").read_text()
    imports = [line.strip() for line in src.splitlines() if line.strip().startswith(("import ", "from "))]
    assert all(line in ("import math", "import numpy as np", "import uncertain") for line in imports), imports


# --------------------------------------------------------------------------- criterion 9b
# Components written exactly to the PDF's protocols (§4.5, §4.6, §7 table) -- nothing more declared.
def _spec_components():
    class SpecDistance:                       # §4.5 / §7: input_type, range, __call__(a, b)
        input_type, range = "units", (0.0, 1.0)

        def __call__(self, a, b):
            return float(a != b)

    class SpecStatistic:                      # §4.6 / §7: range, __call__(A, B)
        range = (0.0, 1.0)

        def __call__(self, A, B):
            return float(np.mean([a != b for a in A.reps for b in B.reps]))

    class SpecEncoder:                        # §4.5 / §7: output_type, encode(set)
        output_type = "units"

        def __init__(self, backend=None):
            pass

        def encode(self, s):
            return RepresentationSet(tuple(x.units for x in s.sequences), s, {"enc": "spec"}, "units")

    class SpecWarping:                        # §4.7 / §7: domain, __call__(x), gap
        domain = (-math.inf, math.inf)

        def __call__(self, x):
            return x

        def gap(self, raw, n):
            return 0.0
    return dict(distance=SpecDistance, statistic=SpecStatistic, encoder=SpecEncoder, warping=SpecWarping)


@pytest.mark.req("9b")
@pytest.mark.parametrize("kind", ["distance", "statistic", "encoder", "warping"])
def test_spec_protocol_component_registers_and_runs(kind, measure):
    cls = _spec_components()[kind]
    name = f"eval_spec_{kind}"
    getattr(uncertain, f"register_{kind}")(name)(cls)
    field = {"distance": "distance", "statistic": "statistic", "encoder": "representation",
             "warping": "warping"}[kind]
    defaults = dict(representation="tokens", distance="discrete", statistic="mean", warping="identity")
    lm = ToyLM([.5, .3, .2])
    try:
        r = make_measure(lm, n=16, horizon=2, **{**defaults, field: name}).compute(Stimulus((0,), (1,)))
    except Exception as exc:
        measure("error", f"{type(exc).__name__}: {exc}")
        raise
    assert math.isfinite(r.value)


# --------------------------------------------------------------------------- criterion 29
CASES = [
    ("encoder/distance mismatch", dict(representation="tokens", distance="cosine"), ("units", "vector")),
    ("energy needs negative type", dict(distance="ngram_overlap", statistic="energy"), ("energy",)),
    ("unit unsupported by encoder", "unit", ("unit",)),
]


@pytest.mark.req("29")
@pytest.mark.parametrize("label,kw,needles", CASES, ids=[c[0] for c in CASES])
def test_config_errors_before_sampling(label, kw, needles, measure):
    lm = ToyLM([.5, .5])
    if kw == "unit":
        class TokenOnly:
            output_type, supported_units = "units", {"token"}
            descriptor = {"encoder": "eval-token-only", "version": "1"}

            def encode(self, s, short):  # pragma: no cover - never reached
                raise AssertionError
        base = make_measure(lm, n=4, representation="tokens", distance="discrete", statistic="mean")
        call = lambda: Measure(base.spec.replace(unit="word"), sampler=base.sampler, encoder=TokenOnly())  # noqa: E731
    else:
        call = lambda: make_measure(lm, n=4, **{"statistic": "mean", **kw})  # noqa: E731
    with pytest.raises(ConfigError) as info:
        call()
    measure("message", str(info.value))
    assert lm.calls == 0
    assert all(n in str(info.value) for n in needles), str(info.value)


# --------------------------------------------------------------------------- criterion 40
@pytest.mark.req("40")
def test_warping_domain_violation_rejected():
    import plugin_components as pc
    lm = ToyLM([.5, .5])
    with pytest.raises(ConfigError) as info:  # L1 range (0, inf) is fine; force a negative-range stat
        class Signed(Mean):
            descriptor = {"statistic": "eval-signed", "version": "1"}

            def range_for(self, distance):
                return (-1.0, 1.0)
        base = make_measure(lm, n=4, representation=pc.ENC_NAME, distance=pc.DIST_NAME, statistic="mean")
        Measure(base.spec.replace(warping="neglog"), sampler=base.sampler, statistic=Signed())
    assert "neglog" in str(info.value) and lm.calls == 0


@pytest.mark.req("40")
def test_neglog_over_range_including_zero_is_accepted(measure):
    # §4.5 says "f = -log with a statistic whose range includes 0 fails"; the surprisal preset
    # (§5) is exactly -log over a [0,1] mean. Record what the library does.
    lm = ToyLM([.5, .5])
    m = make_measure(lm, n=4, representation="tokens", distance="prefix_indicator", statistic="mean",
                     warping="neglog")
    measure("accepted", True)
    assert m.statistic.range_for(m.distance) == (0.0, 1.0)


# --------------------------------------------------------------------------- criterion 36
DISSIM = ["discrete", "ngram_distance"]


@pytest.mark.req("36")
@settings(max_examples=300, deadline=None)
@given(x=words, y=words, z=words, name=st.sampled_from(DISSIM))
def test_metric_axioms(x, y, z, name):
    d = resolve("distance", name)
    assert d(x, x) == 0
    assert d(x, y) >= 0 and d(x, y) == d(y, x)
    assert d(x, z) <= d(x, y) + d(y, z) + 1e-12


@pytest.mark.req("36")
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 10**6), dim=st.integers(2, 5))
def test_cosine_axioms(seed, dim):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=dim), rng.normal(size=dim)
    d = resolve("distance", "cosine")
    assert abs(d(x, x)) <= 1e-12 and d(x, y) >= 0 and abs(d(x, y) - d(y, x)) <= 1e-12


@pytest.mark.req("36")
@settings(max_examples=150, deadline=None)
@given(a=st.lists(words, min_size=2, max_size=8), b=st.lists(words, min_size=2, max_size=8),
       perm_seed=st.integers(0, 10**6), stat=st.sampled_from(["mean", "energy", "min"]))
def test_statistics_are_permutation_invariant(a, b, perm_seed, stat):
    rng = np.random.default_rng(perm_seed)
    pa, pb = [a[i] for i in rng.permutation(len(a))], [b[i] for i in rng.permutation(len(b))]
    s = {"mean": Mean, "energy": Energy, "min": DirectedMin}[stat]()
    d = resolve("distance", "discrete")
    v1 = s.estimate(rep_set(a, tag="a"), rep_set(b, tag="b"), d, EST, (1, 2)).value
    v2 = s.estimate(rep_set(pa, tag="pa"), rep_set(pb, tag="pb"), d, EST, (1, 2)).value
    assert abs(v1 - v2) <= 1e-12


# --------------------------------------------------------------------------- criterion 37
@pytest.mark.req("37")
@pytest.mark.parametrize("kind,name", [("distance", "prefix_indicator"), ("distance", "ngram_overlap"),
                                       ("distance", "cosine"), ("statistic", "mean"),
                                       ("statistic", "min"), ("statistic", "energy")])
def test_shipped_components_resolve(kind, name):
    assert resolve(kind, name) is not None


@pytest.mark.req("37")
@pytest.mark.parametrize("plugin", ["pos", "tree_edit", "entailment"])
def test_plugin_examples_live_outside_core_and_import(plugin):
    path = ROOT / "examples" / plugin / "plugin.py"
    assert path.exists() and "src" not in path.parts
    spec = importlib.util.spec_from_file_location(f"eval_example_{plugin}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    core = Path(uncertain.__file__).parent
    assert not str(path).startswith(str(core))

import pytest
from uncertain import Measure, Stimulus, ToyBackend


def _computed():
    backend = ToyBackend(("a", "b", "<eos>"), {(): (.6, .3, .1)}, eos_token_id=2)
    m = Measure.from_config(backend, horizon=2, n=40, statistic="mean",
                            short_samples="compare_ragged")
    return m, m.compute(Stimulus(target=(0,)))


def test_alternatives_and_summary():
    m, r = _computed()
    sets = m.alternatives(r)
    assert sets[0].content_hash == r.artifact_ids[0]
    rows = sets[0]._grouped(top=100)
    assert sum(count for _, count, _, _ in rows) == 40
    assert rows[0][1] == max(count for _, count, _, _ in rows)
    text = sets[0].summary(top=3)
    assert text.startswith("40 alternatives") and len(text.splitlines()) == 2 + min(3, len(rows))
    labels = {label for label, *_ in sets[0]._grouped(100, m.sampler.backend.decode)}
    assert labels <= {"aa", "ab", "ba", "bb", "a<eos>", "b<eos>", "<eos>"}


def test_plot():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    m, r = _computed()
    ax = m.alternatives(r)[0].plot(top=5)
    assert len(ax.patches) == min(5, len(m.alternatives(r)[0]._grouped(100)))


def test_sentence_summary_mc_and_exact():
    from uncertain import positions
    backend = ToyBackend(("a", "b", "<eos>"), {(): (.6, .3, .1)}, eos_token_id=2)
    for horizon, kwargs in ((2, dict(statistic="mean")), (1, dict(warping="neglog"))):
        m = Measure.from_config(backend, horizon=horizon, n=40, short_samples="compare_ragged", **kwargs)
        lines = m.sentence_summary(positions((0, 1, 0)), top=2).splitlines()
        assert len(lines) == 4 and "'a'" in lines[1]
    assert "60%" in lines[1]  # exact path lists p(next token)

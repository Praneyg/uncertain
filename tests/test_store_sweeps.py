from dataclasses import replace
import json
import numpy as np
import pytest
from uncertain import (AlternativeStore, Sampler, Measure, MeasureSpec, Sweep, Stimulus,
                       StimulusTransform, positions, ConfigError, load_dataset, EstimatorSpec)
from uncertain.store import StoreError


def test_roundtrip_and_corruption(tmp_path, backend, sampling):
    sampler = Sampler(backend)
    a = sampler.sample((1,), 3, replace(sampling, n=9))
    store = AlternativeStore(tmp_path)
    key = store.save(a)
    b = AlternativeStore(tmp_path).load(key)
    assert b.content_hash == a.content_hash and b.sequences == a.sequences
    data = json.loads((tmp_path/"alternatives"/(key+".json")).read_text())
    data["payload"]["sequences"][0]["tokens"][0] ^= 1
    (tmp_path/"alternatives"/(key+".json")).write_text(json.dumps(data))
    with pytest.raises(StoreError, match="checksum"):
        store.load(key)


def test_sweep_groups_contexts_and_horizons(tmp_path, backend, sampling):
    m = Measure(MeasureSpec(sampling, 1, "compare_ragged"), sampler=Sampler(backend),
                store=AlternativeStore(tmp_path))
    items = positions((0,1,0))
    sweep = Sweep(m, horizon=[1,2,4], warping=["identity", "neglog"])
    plan = sweep.plan(items)
    assert plan.describe()["sample_requests"] == 4
    assert m.sampler.calls == 0
    result = plan.run()
    assert m.sampler.calls == 4 and len(result.rows) == 18
    assert all(r["datum"] is None for r in result.rows)
    fresh = Measure(m.spec, sampler=Sampler(backend), store=AlternativeStore(tmp_path))
    replay = Sweep(fresh, horizon=[1,2,4], warping=["identity", "neglog"]).run(items)
    assert fresh.sampler.calls == 0
    assert replay.rows == result.rows
    memory = Measure(m.spec, sampler=Sampler(backend))
    ram = Sweep(memory, horizon=[1,2,4], warping=["identity", "neglog"]).run(items)
    assert ram.rows == result.rows


def test_invalid_grid_before_work(make_measure):
    m = make_measure()
    with pytest.raises(ConfigError, match="cell"):
        Sweep(m, distance=["prefix_indicator", "cosine"])
    assert m.sampler.calls == 0
    with pytest.raises(ConfigError, match="one unit"):
        Sweep(m, unit=["word"])


def test_transform_provenance_and_data_loading(tmp_path):
    item = Stimulus((0,1,2,3), (1,), "i", datum=30.)
    assert StimulusTransform("drop")(item).context == ()
    assert StimulusTransform("truncate", count=2)(item).context == (2,3)
    assert StimulusTransform("shuffle", seed=2)(item) == StimulusTransform("shuffle", seed=2)(item)
    derived = StimulusTransform("replace", replacement=(9,))(item)
    assert derived.original_id == "i" and derived.datum == 30.
    path = tmp_path/"items.jsonl"
    path.write_text(json.dumps({"context": [0], "target": [1], "datum": 23})+"\n")
    assert load_dataset(path)[0].datum == 23


def test_output_formats(tmp_path, make_measure):
    result = Sweep(make_measure()).run(positions((0,1), data=[1.2,2.3]))
    result.write(tmp_path/"scores.csv")
    result.write(tmp_path/"scores.json")
    assert (tmp_path/"scores.manifest.json").exists()
    assert len(json.loads((tmp_path/"scores.json").read_text())) == 2
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    result.write(tmp_path/"scores.parquet")
    frame = pd.read_parquet(tmp_path/"scores.parquet")
    np.testing.assert_allclose(frame.datum, [1.2,2.3])


def test_exact_sweep_has_no_sampling(make_measure):
    m = make_measure(estimator=EstimatorSpec(method="exact"))
    result = Sweep(m, horizon=[1,2]).run(positions((0,1)))
    assert m.sampler.calls == 0 and len(result.rows) == 4


def test_content_keys_change_after_anchor(make_measure, backend):
    from uncertain import AlternativeSet, UnitSpec
    m = make_measure(horizon=3)
    a = m._get((0,), 2)
    observed = Stimulus((), (0,)).observed(backend, UnitSpec())
    b = AlternativeSet.anchor(observed, a, ())
    assert m._encode(a).content_hash != m._encode(b).content_hash



# §1.4 and §10: the dataframe schema is the contract. Pinned here so a dropped, renamed
# or reordered column fails loudly instead of breaking every downstream consumer silently.
SWEEP_COLUMNS = (
    "run_id", "stimulus_id", "document_id", "target_start", "target_end", "unit",
    "context_hash", "transform_id", "original_id", "datum", "spec_hash", "horizon",
    "model_id", "model_revision", "representation", "distance", "statistic", "warping",
    "short_samples", "probability", "mode", "value", "raw_value", "corrected_value", "se",
    "ci_low", "ci_high", "confidence", "n_requested", "n_a_used", "n_b_used", "method",
    "status", "artifact_ids", "warping_bias", "diagnostics")


def test_dataframe_schema_is_pinned_in_both_pairing_modes(make_measure):
    unpaired = Sweep(make_measure()).run(positions((0,1)))
    paired = Sweep(make_measure()).run(positions((0,1), data=[1.2,2.3]))
    for result in (unpaired, paired):
        assert all(tuple(row) == SWEEP_COLUMNS for row in result.rows)
    # The datum column is present either way; unpaired items leave it null rather than
    # emitting a narrower schema.
    assert [r["datum"] for r in unpaired.rows] == [None, None]
    assert [r["datum"] for r in paired.rows] == [1.2, 2.3]
    pytest.importorskip("pandas")
    assert tuple(paired.to_dataframe().columns) == SWEEP_COLUMNS
    assert str(paired.to_dataframe().datum.dtype) == "Float64"

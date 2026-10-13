"""Category D: set size, extend, top_k, axis order / log-prob alignment, optional persistence."""
import math
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

import uncertain
import uncertain.samplers as samplers_module
from uncertain import (AlternativeSet, AlternativeStore, Measure, Sampler, SamplingSpec, Sequence,
                       Stimulus, UnitSpec)

from conftest import make_measure
from toy_lm import ToyLM

SRC = Path(uncertain.__file__).resolve().parent


# --------------------------------------------------------------------------- criterion 15
@pytest.mark.req("15")
def test_set_size_is_required_and_honoured():
    lm = ToyLM([.5, .5])
    with pytest.raises(TypeError):
        Measure.from_config(lm, short_samples="compare_ragged", sampler=Sampler(lm))
    for n in (2, 17, 100):
        r = make_measure(lm, n=n, representation="tokens", distance="discrete", statistic="mean").compute(
            Stimulus((0,), (1,)))
        assert r.n_requested == n == r.n_a_used == r.n


# --------------------------------------------------------------------------- criterion 16
@pytest.mark.req("16")
@pytest.mark.parametrize("n1,n2", [(8, 8), (16, 5), (1, 31)])
def test_extend_generates_only_new_samples(n1, n2, monkeypatch, measure):
    lm = ToyLM([.4, .3, .2, .1])
    sampler = Sampler(lm)
    spec = SamplingSpec(lm.model_spec, n=n1, seed=7)
    first = sampler.sample((0,), 1, spec, UnitSpec())
    requests = []
    original = Sampler.sample

    def spy(self, context, horizon, spec, unit=None, *, start=0):
        requests.append((spec.n, start))
        return original(self, context, horizon, spec, unit, start=start)
    monkeypatch.setattr(Sampler, "sample", spy)
    calls = lm.calls
    grown = sampler.extend(first, n2)
    measure("backend_calls", lm.calls - calls); measure("sample_requests", requests)
    # H=1 token unit: at most one forward per new draw. The sampler caches p(.|context),
    # so all draws here share one cached forward; `requests` is what proves no redraw.
    assert lm.calls - calls <= n2
    assert requests == [(n2, n1)]
    assert grown.sequences[:n1] == first.sequences
    assert len(grown) == n1 + n2
    monkeypatch.undo()
    direct = Sampler(lm).sample((0,), 1, SamplingSpec(lm.model_spec, n=n1 + n2, seed=7), UnitSpec())
    assert grown.sequences == direct.sequences


# --------------------------------------------------------------------------- criterion 17
@pytest.mark.req("17")
def test_top_k_explicit_and_off_by_default(monkeypatch, measure):
    seen = []
    original = samplers_module.sampling_probabilities

    def spy(p, generation):
        seen.append(dict(generation))
        return original(p, generation)
    monkeypatch.setattr(samplers_module, "sampling_probabilities", spy)
    lm = ToyLM([.5, .5])
    a = Sampler(lm).sample((0,), 3, SamplingSpec(lm.model_spec, n=4, seed=0), UnitSpec())
    assert seen and all("top_k" in g and g["top_k"] == 0 for g in seen)
    assert a.provenance["sampling"]["generation"]["top_k"] == 0
    assert SamplingSpec(lm.model_spec, n=1, seed=0).generation["top_k"] == 0


@pytest.mark.req("17")
@pytest.mark.parametrize("top_k,expect_tail", [(None, True), (50, False)])
def test_tail_beyond_top_50_is_sampled(top_k, expect_tail, measure):
    V, n = 60, 6000
    p = 0.97 ** np.arange(V)
    p /= p.sum()
    tail = float(np.sort(p)[:V - 50].sum())
    lm = ToyLM(p)
    gen = {} if top_k is None else {"top_k": top_k}
    a = Sampler(lm).sample((0,), 1, SamplingSpec(lm.model_spec, n=n, seed=3, generation=gen), UnitSpec())
    top50 = set(np.argsort(-p)[:50].tolist())
    freq = sum(s.tokens[0] not in top50 for s in a.sequences) / n
    measure("tail_mass", tail); measure("tail_freq", freq)
    if expect_tail:
        assert abs(freq - tail) <= 4 * math.sqrt(tail * (1 - tail) / n)
    else:
        assert freq == 0


# --------------------------------------------------------------------------- criterion 18
class _FakeTransformer:
    def __call__(self, ids, past_key_values=None, use_cache=True):
        import torch
        step = 0 if past_key_values is None else past_key_values + 1
        B = ids.shape[0]
        h = torch.zeros(B, 1, 2, dtype=torch.float64)
        h[:, 0, 0] = torch.arange(B, dtype=torch.float64)
        h[:, 0, 1] = step
        return SimpleNamespace(last_hidden_state=h, past_key_values=step)


def _fake_gpt2(B, V, T):
    """Batched 'GPT-2' whose logits table (T, B, V) has a distinct value in every cell."""
    import torch

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="gpt2", n_positions=128, bos_token_id=0,
                                          eos_token_id=V - 1, hidden_size=2)
            self.table = torch.sin(torch.arange(T * B * V, dtype=torch.float64) * 0.7).mul(3).reshape(T, B, V)
            self.transformer = _FakeTransformer()

        def lm_head(self, h):
            rows, steps = h[:, 0].long(), h[:, 1].long()
            return self.table[steps, rows]
    return Fake()


@pytest.mark.req("18")
@pytest.mark.parametrize("B,V,T", [(3, 7, 5), (4, 9, 3), (2, 11, 6)])
def test_batched_logits_rows_stay_with_their_sample(B, V, T, measure):
    # Edge case: non-square logits (batch != vocab != steps).
    torch = pytest.importorskip("torch")
    from uncertain import HFBackend
    from uncertain.emnlp2024 import StudySampler
    model = _fake_gpt2(B, V, T)
    backend = HFBackend(model=model, bos_token_id=0, eos_token_id=V - 1)
    a = StudySampler(backend, batch_size=B).sample((1,), T, SamplingSpec(backend.model_spec, n=B, seed=0))
    expected = torch.log_softmax(model.table.transpose(0, 1), dim=-1).numpy()  # (B, T, V) by named swap
    worst = 0.
    for row in a.sequences:
        j = row.sample_id
        for s, (tok, lp) in enumerate(zip(row.tokens, row.model_logprobs)):
            worst = max(worst, abs(lp - expected[j, s, tok]))
    measure("max_abs_err", worst)
    assert worst <= 1e-12


@pytest.mark.req("18")
def test_sampler_logprobs_match_teacher_forced(tiny_gpt2, measure):
    spec = SamplingSpec(tiny_gpt2.model_spec, n=6, seed=1, token_budget=8)
    a = Sampler(tiny_gpt2).sample((1, 2, 3), 6, spec, UnitSpec())
    worst = max(float(np.abs(np.array(r.model_logprobs) - tiny_gpt2.score((1, 2, 3), r.tokens)).max())
                for r in a.sequences)
    measure("max_abs_err", worst)
    assert worst <= 1e-5


@pytest.mark.req("18", "23")
def test_study_sampler_batched_logprobs_match_unbatched_teacher_forced(tiny_gpt2, measure):
    from uncertain.emnlp2024 import StudySampler
    spec = SamplingSpec(tiny_gpt2.model_spec, n=5, seed=2, token_budget=8)
    a = StudySampler(tiny_gpt2, batch_size=3).sample((1, 2, 3), 6, spec)
    worst = max(float(np.abs(np.array(r.model_logprobs) - tiny_gpt2.score((1, 2, 3), r.tokens)).max())
                for r in a.sequences)
    measure("max_abs_err", worst)
    assert worst <= 1e-4


@pytest.mark.req("18")
@pytest.mark.parametrize("n,width", [(3, 5), (5, 2), (4, 4)])
def test_arrays_are_sample_by_step(n, width):
    seqs = tuple(Sequence(tuple(10 * i + j for j in range(width)),
                          tuple(-(10 * i + j + 1) / 100 for j in range(width)), (-.1,) * width,
                          tuple(str(j) for j in range(width)), tuple((j, j + 1) for j in range(width)),
                          sample_id=i) for i in range(n))
    arr = AlternativeSet(seqs, (), UnitSpec(), width).arrays()
    assert arr["tokens"].shape == (n, width)
    for i in range(n):
        for j in range(width):
            assert arr["tokens"][i, j] == 10 * i + j
            assert arr["model_logprobs"][i, j] == -(10 * i + j + 1) / 100


@pytest.mark.req("18")
def test_no_reshape_based_axis_reordering_in_src(measure):
    hits = []
    for path in SRC.rglob("*.py"):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.(reshape|view|permute|swapaxes)\(|\.T\b|transpose\(", line):
                hits.append(f"{path.relative_to(SRC)}:{no}: {line.strip()}")
    measure("hits", hits)
    # The only allowed occurrence: the same-shape immutable-buffer round trip in utils.
    assert all(h.startswith(str(Path("utils/__init__.py"))) and "reshape(a.shape)" in h for h in hits), hits


# --------------------------------------------------------------------------- criterion 19
def _files(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())


@pytest.mark.req("19")
def test_persistence_off_by_default_and_results_identical(scratch, monkeypatch, measure):
    lm = ToyLM([.4, .3, .2, .1])
    work = scratch / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    kw = dict(n=32, horizon=2, representation="tokens", distance="discrete", statistic="energy")
    r_mem = make_measure(lm, **kw).compute(Stimulus((0,), (1,)))
    assert _files(work) == []
    disk = scratch / "store"
    r_disk = make_measure(lm, store=AlternativeStore(disk), **kw).compute(Stimulus((0,), (1,)))
    files = _files(disk)
    measure("files", files)
    assert files and all(f.split("\\")[0].split("/")[0] in {"alternatives", "requests"} for f in files)
    assert r_mem == r_disk
    r_reload = make_measure(lm, store=AlternativeStore(disk), **kw).compute(Stimulus((0,), (1,)))
    assert r_reload == r_mem

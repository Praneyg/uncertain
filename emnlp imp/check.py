"""Self-check for plugins.py and the identity-derived pairwise measures (GPT-2, offline)."""
import os
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from uncertain import HFBackend, Measure, Sampler, SamplingSpec, Stimulus, AlternativeStore
from uncertain.emnlp2024 import cosine_u
from plugins import BatchedSampler
from run_estimates import MODELS, measures, compute_token

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
backend = HFBackend(MODELS["gpt2-small"][0], revision=MODELS["gpt2-small"][1],
                    bos_token_id=50256, eos_token_id=50256, device=device)
ctxs = [(50256,) + backend.encode(t) for t in ("The cat sat on the", "Once upon a", "He said that")]

# (a) batched draws == reference sampler draws, same streams
spec = SamplingSpec(backend.model_spec, n=16, seed=3)
ref, fast = Sampler(backend), BatchedSampler(backend, batch_size=8)
fast.stream_key = ref.stream_key  # same RNG streams: isolates the draw algorithm
same = total = 0
for c in ctxs:
    a, b = ref.sample(c, 3, spec), fast.sample(c, 3, spec)
    assert a.provenance["stream"] == b.provenance["stream"]
    for x, y in zip(a.sequences, b.sequences):
        total += 1
        same += x.tokens == y.tokens
        if x.tokens == y.tokens:
            assert np.allclose(x.model_logprobs, y.model_logprobs, atol=1e-3)
print(f"(a) identical draws {same}/{total}")
assert same >= total - 1  # float32 batched-vs-single logits may flip a near-tie

fast = BatchedSampler(backend, batch_size=8)  # own stream identity from here on

# (b) layer0_local == hidden_states[0] of the bare sequence, mean over non-EOS
torch = backend._load()
m = Measure.from_config(backend, sampler=fast, representation="layer0_local", distance="cosine",
                        statistic="mean", estimator="mc", n=16, seed=3, short_samples="compare_ragged")
a = fast.sample(ctxs[0], 3, spec)
reps = m.encoder.encode(a, "compare_ragged").reps
with torch.inference_mode():
    for r, rep in zip(a.sequences, reps):
        h = backend._model(torch.tensor([r.tokens], device=device), output_hidden_states=True).hidden_states[0][0]
        keep = torch.tensor(r.tokens, device=device) != 50256
        assert np.allclose(h[keep].mean(0).double().cpu().numpy(), rep, atol=1e-5)
print("(b) layer0_local matches hidden_states[0]")

# (c) within_u == off-diagonal pairwise mean == 2*mean - U-energy (built-in route), same samples
from uncertain import StatisticSpec
ms = measures(backend, fast, AlternativeStore(), n=64, horizon=5)
s = Stimulus(ctxs[1], (backend.encode(" time")[0],))
ref = Stimulus(s.context, (50256,))
row = compute_token(ms, s, backend)
a5 = ms["seq_within"].alternatives(ms["seq_within"].compute(ref))[0]
a1 = a5.truncate(1)
enc = ms["iv"].encoder
assert np.isclose(row["expected_decontextualised"], cosine_u(enc.encode(a1, "x").reps), atol=1e-9)
assert np.isclose(row["expected_seq_decontextualised"], cosine_u(enc.encode(a5, "x").reps), atol=1e-9)
u = StatisticSpec("energy", {"variant": "u"})
for m, stim, got in ((ms["iv"], s, row["expected_decontextualised"]),
                     (ms["iv"].replace(horizon=5), ref, row["expected_seq_decontextualised"])):
    energy = m.replace(statistic=u).compute(stim)
    assert np.isclose(got, 2 * m.compute(stim).value - energy.value, atol=1e-9)
    w = m.replace(statistic="within_u").compute(stim)
    # bootstrap replicates equal the U-energy within-A replicates (same library weights)
    assert np.allclose(2 * np.array(m.compute(stim).resamples) - np.array(energy.resamples),
                       w.resamples, atol=1e-9)
print("(c) within_u == cosine_u == 2*mean - U-energy:",
      round(row["expected_decontextualised"], 4), round(row["expected_seq_decontextualised"], 4))
# (d) sum p^2 shortcut == library exact discrete energy - 1 + 2 p(y)
e = ms["energy_discrete"].compute(s).value
assert np.isclose(row["expected_prob"], e - 1 + 2 * row["prob"], atol=1e-9)
# (e) cosine_fast == built-in cosine
from uncertain.distances import Cosine
from plugins import CosineFast
x, y = reps[0], reps[1]
assert np.isclose(Cosine()(x, y), CosineFast()(x, y), atol=1e-12)
print("(d) sum p^2 == exact energy identity; (e) cosine_fast == cosine")
print("all checks passed")

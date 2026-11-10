"""EMNLP 2024 generalized-surprisal estimates, computed with uncertain.Measure.

Per text: every token position is a Stimulus (BOS + prefix -> token). Token scores are
aggregated to whitespace words with the original study's rules (sum / product / first).

    python "emnlp imp/run_estimates.py" --model gpt2-small [--limit 3]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from uncertain import AlternativeStore, EstimatorSpec, HFBackend, Measure, StatisticSpec
from uncertain.data import positions
from plugins import BatchedSampler, LN2  # also registers layer0_local

ORIGINAL = Path(r"C:\Users\praney\OneDrive\Desktop\generalized-surprisal")
MODELS = {  # label: (hub id, pinned commit)
    "gpt2-small": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "neo-125m": ("EleutherAI/gpt-neo-125m", "21def0189f5705e2521767faed922f1f15e7d7db"),
}
# Word aggregation, as in the original word_score().
AGGREGATE = {"surprisal": "sum", "prob": "prod", "entropy": "first", "expected_prob": "first",
             "decontextualised": "sum", "expected_decontextualised": "first",
             "expected_seq_decontextualised": "first", "sequence_entropy": "first"}


def measures(backend, sampler, store, n=512, horizon=5, seed=0):
    base = Measure.from_config(backend, sampler=sampler, store=store, n=n, seed=seed,
                               short_samples="compare_ragged", horizon=1,
                               distance="prefix_indicator", statistic="mean", estimator="exact")
    cos = dict(representation="layer0_local", distance="cosine_fast", estimator=EstimatorSpec(method="mc"))
    return {
        "surprisal": base.replace(warping="neglog"),
        "prob": base,
        "entropy": base.replace(mode="entropy"),
        "energy_discrete": base.replace(distance="discrete", statistic="energy"),
        # H=5 first; the store then serves H=1 by truncating the same draws.
        "seq_entropy": base.replace(mode="entropy", horizon=horizon, estimator=EstimatorSpec(method="mc")),
        "seq_within": base.replace(horizon=horizon, statistic="within_u", **cos),
        "iv": base.replace(**cos),
        "iv_within": base.replace(statistic="within_u", **cos),
    }


def compute_token(ms, s, backend):
    r = {k: ms[k].compute(s) for k in ("surprisal", "prob", "entropy")}
    # within_u ignores B; an EOS target keeps H=5 from sampling a continuation of y.
    ref = type(s)(s.context, (backend.eos_token_id,))
    seq = {k: ms[k].compute(ref) for k in ("seq_entropy", "seq_within")}
    iv, iv_w = ms["iv"].compute(s), ms["iv_within"].compute(s)
    a = ms["seq_entropy"].alternatives(seq["seq_entropy"])[0]
    return {
        "surprisal": r["surprisal"].value / LN2,
        "prob": r["prob"].value,
        "entropy": r["entropy"].value,  # nats, as the original
        # sum p^2. Library route: exact discrete energy - 1 + 2 p(y) (asserted equal in
        # check.py), but its vocabulary enumeration costs ~6 s/token, so read p directly.
        "expected_prob": float(np.square(backend.probabilities(s.context)).sum()),
        "decontextualised": iv.value,
        "decontextualised_se": iv.se,
        "expected_decontextualised": iv_w.value,
        "expected_decontextualised_se": iv_w.se,
        "expected_seq_decontextualised": seq["seq_within"].value,
        "expected_seq_decontextualised_se": seq["seq_within"].se,
        "sequence_entropy": seq["seq_entropy"].value / LN2,
        "sequence_entropy_se": seq["seq_entropy"].se / LN2,
        "first_eos_draws": sum(x.tokens[:1] == (backend.eos_token_id,) for x in a.sequences),
        "later_eos_draws": sum(x.stop_reason == "eos" and len(x.tokens) > 1 for x in a.sequences),
    }


def word_index(text, offsets):
    """Index of the whitespace word each token ends in."""
    ends, pos = [], 0
    for w in text.split():
        pos = text.index(w, pos) + len(w)
        ends.append(pos)
    out, k = [], 0
    for _, e in offsets:
        while ends[k] < e:
            k += 1
        out.append(k)
    return out


def aggregate(tokens):
    words = []
    for k, group in pd.DataFrame(tokens).groupby("word_index", sort=True):
        row = {"sent_id": group.sent_id.iloc[0], "word_index": k, "n_tokens": len(group)}
        for metric, how in AGGREGATE.items():
            v = group[metric]
            row[metric] = v.sum() if how == "sum" else v.prod() if how == "prod" else v.iloc[0]
        words.append(row)
    return words


def append(path, rows):
    old = pd.read_csv(path, keep_default_na=False) if path.exists() else None
    pd.concat([old, pd.DataFrame(rows)]).to_csv(path, index=False)  # union of columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    if os.name == "nt":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep mid-run
        __import__("ctypes").windll.kernel32.SetThreadExecutionState(0x80000001)
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    model_id, revision = MODELS[args.model]
    backend = HFBackend(model_id, revision=revision, bos_token_id=50256, eos_token_id=50256,
                        device=args.device)
    backend._load()
    sampler = BatchedSampler(backend)
    texts = pd.read_csv(ORIGINAL / "data/texts/devarda/aligned_texts.csv")
    if args.limit:
        texts = texts.head(args.limit)
    words_path = out / "word_estimates.csv"
    done = set(pd.read_csv(words_path).sent_id) if words_path.exists() else set()
    tokens_path = out / "token_estimates.csv"
    if tokens_path.exists():  # drop token rows of a text killed before its word rows were written
        t = pd.read_csv(tokens_path, keep_default_na=False)
        t[t.sent_id.isin(done)].to_csv(tokens_path, index=False)
    t0, n_tokens = time.time(), 0
    for i, (sid, text) in enumerate(zip(texts.id, texts.text)):
        if sid in done:
            continue
        enc = backend._tokenizer(text, return_offsets_mapping=True)
        ids = tuple(enc["input_ids"])
        assert ids == backend.encode(text)
        wi = word_index(text, enc["offset_mapping"])
        ms = measures(backend, sampler, AlternativeStore(), n=args.n, horizon=args.horizon)
        tokens = []
        for s, k in zip(positions(ids, context=(backend.bos_token_id,), document_id=str(sid)), wi):
            row = {"sent_id": sid, "token_index": s.position, "token": backend.decode(s.target),
                   "word_index": k, **compute_token(ms, s, backend)}
            tokens.append(row)
        append(out / "token_estimates.csv", tokens)
        append(words_path, aggregate(tokens))
        n_tokens += len(tokens)
        el = time.time() - t0
        print(f"[{i+1}/{len(texts)}] id={sid} tokens={len(tokens)} {el/n_tokens:.2f}s/token "
              f"elapsed={el/60:.1f}m rss={__import__('psutil').Process().memory_info().rss/2**30:.2f}GB",
              flush=True)
    manifest = {"model": model_id, "revision": revision, "n": args.n, "horizon": args.horizon,
                "seed": 0, "temperature": 1.0, "top_k": 0, "first_step_eos": "not suppressed (raw p)",
                "sampler": sampler.descriptor, "backend": backend.descriptor,
                "encoder": measures(backend, sampler, AlternativeStore())["iv"].encoder.descriptor,
                "runtime_s_this_invocation": time.time() - t0}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

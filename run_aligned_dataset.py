"""Score aligned.csv using the uncertain library and a pinned, cached GPT-2.

Run with the Python interpreter from the uncertain virtual environment.
All source rows and columns are retained. Scores use natural logarithms (nats).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(
        r"C:\Users\praney\OneDrive\Desktop\generalized-surprisal\data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepend-bos", action="store_true",
                        help="Prepend BOS to every context, matching the study's ExactScorer")
    parser.add_argument("--per-sentence", action="store_true",
                        help="Score each sentence in one forward pass and slice word spans; "
                             "requires --prepend-bos (every word then sees the same BOS)")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every row")
    args = parser.parse_args()
    if args.threads < 1 or args.limit < 0:
        parser.error("threads must be positive and limit must be nonnegative")
    if args.per_sentence and not args.prepend_bos:
        parser.error("--per-sentence requires --prepend-bos")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / (
            "replication/exact" if args.prepend_bos else "aligned_gpt2")

    # Enforce local-only execution; fail clearly if the pinned model is absent.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import uncertain
    from uncertain import EstimatorSpec, HFBackend, Measure, Sampler, SamplingSpec
    from uncertain import Stimulus, surprisal

    torch.set_num_threads(args.threads)
    source = args.data_dir / "psyling" / "aligned.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not {"item", "word"}.issubset(original_columns):
        raise ValueError("aligned.csv must contain item and word columns")
    total_source_rows = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = HFBackend(args.model, revision=args.revision, device="cpu")
    sampler = Sampler(backend)
    sampling = SamplingSpec(model=backend.model_spec, n=1, seed=42)
    measures = {}
    sentences = {}  # sentence text -> (token IDs, per-token log-probs after BOS)
    sentence_rows = 0
    extra = ["uncertain_source_row", "uncertain_context_tokens", "uncertain_target_tokens",
             "uncertain_target_ids", "uncertain_surprisal_nats", "uncertain_surprisal_bits",
             "uncertain_probability", "uncertain_method", "uncertain_status",
             "uncertain_error", "uncertain_spec_hash", "uncertain_seconds"]
    if set(extra).intersection(original_columns):
        raise ValueError("Source already contains output columns")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    statuses = Counter()
    token_counts = Counter()
    values = []
    reference_deltas = []
    score_path = args.output_dir / "scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=original_columns + extra)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            record = dict(row)
            record.update(dict.fromkeys(extra, ""))
            record["uncertain_source_row"] = index
            tick = time.perf_counter()
            try:
                # csv preserves the literal word 'None', unlike pandas' default NA parser.
                context = row["item"].rstrip()
                word = row["word"].lstrip()
                if not word:
                    raise ValueError("Empty target")
                target = (" " if context else "") + word
                stimulus = Stimulus.from_text(context, target, backend,
                                              stimulus_id=f"aligned:{index}")
                if args.prepend_bos:
                    bos = backend._tokenizer.bos_token_id
                    if bos is None:
                        raise ValueError("Notebook convention requires a BOS token")
                    stimulus = replace(stimulus, context=(bos,) + stimulus.context)
                horizon = len(stimulus.target)
                record["uncertain_context_tokens"] = len(stimulus.context)
                record["uncertain_target_tokens"] = horizon
                record["uncertain_target_ids"] = json.dumps(stimulus.target)
                token_counts[horizon] += 1
                span = None
                if args.per_sentence:
                    # Causal LM: one forward over the sentence yields every prefix's
                    # next-token distribution, so word i reuses the pass for words < i.
                    sentence = row["sentence"]
                    if sentence not in sentences:
                        ids = backend.encode(sentence)
                        sentences[sentence] = ids, backend.score(stimulus.context[:1], ids)
                    ids, logprobs = sentences[sentence]
                    start = len(stimulus.context) - 1  # context starts with BOS
                    # A word that tokenizes differently inside the sentence is scored alone.
                    if ids[:start+horizon] == stimulus.context[1:] + stimulus.target:
                        span = logprobs[start:start+horizon]
                if span is not None:
                    value = -float(span.sum())
                    record.update(uncertain_probability=math.exp(-value),
                                  uncertain_method="exact_sentence", uncertain_status="ok")
                    sentence_rows += 1
                else:
                    if horizon not in measures:
                        measures[horizon] = Measure(surprisal(
                            sampling, horizon=horizon, short_samples="compare_ragged",
                            estimator=EstimatorSpec(method="exact")), sampler=sampler)
                    result = measures[horizon].compute(stimulus)
                    value = result.value
                    record.update(uncertain_probability=result.raw_value,
                                  uncertain_method=result.method, uncertain_status=result.status,
                                  uncertain_spec_hash=result.spec_hash)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite surprisal")
                record.update(uncertain_surprisal_nats=value,
                              uncertain_surprisal_bits=value / math.log(2))
                values.append(value)
                if not args.prepend_bos and horizon == 1 and row.get("is_start_end") == "0":
                    try:
                        reference = float(row.get("s_GPT2", ""))
                        if math.isfinite(reference):
                            reference_deltas.append(abs(value - reference))
                    except ValueError:
                        pass
            except Exception as exc:
                record["uncertain_status"] = "error"
                record["uncertain_error"] = f"{type(exc).__name__}: {exc}"
            record["uncertain_seconds"] = time.perf_counter() - tick
            statuses[record["uncertain_status"]] += 1
            writer.writerow(record)
            if index % 100 == 0 or index == len(rows):
                handle.flush()
                print(f"Processed {index}/{len(rows)} in {time.perf_counter()-started:.1f}s; "
                      f"statuses={dict(statuses)}", flush=True)

    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": total_source_rows, "processed_rows": len(rows),
        "filter": "none", "limit": args.limit,
        "model": args.model, "revision": args.revision,
        "uncertain_version": uncertain.__version__, "torch_version": torch.__version__,
        "device": "cpu", "threads": args.threads,
        "method": "exact", "sampling_calls": sampler.calls,
        "per_sentence": args.per_sentence, "per_sentence_rows": sentence_rows,
        "sentence_forwards": len(sentences),
        "prepend_bos": args.prepend_bos,
        "definition": "P(target token sequence | context); surprisal=-ln(P)",
        "text_mapping": "item.rstrip() + one space + word.lstrip(); no space for empty context",
        "parsing": "csv.DictReader preserves literal None and other text values",
        "boundary_rows": "retained; multi-token targets use joint token-sequence probability",
        "started_at_utc": started_at, "elapsed_seconds": time.perf_counter()-started,
        "statuses": dict(statuses), "target_token_counts": dict(token_counts),
        "mean_surprisal_nats": sum(values)/len(values) if values else None,
        "comparison_to_s_GPT2": {
            "scope": "single-token non-boundary targets with finite reference",
            "rows": len(reference_deltas),
            "mean_absolute_error_nats": (sum(reference_deltas)/len(reference_deltas)
                                         if reference_deltas else None),
            "max_absolute_error_nats": max(reference_deltas, default=None),
        },
        "measure_specs": {str(h): m.spec.to_dict() for h, m in measures.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "measure_specs"}, indent=2))
    print(f"Results: {score_path.resolve()}", flush=True)
    if statuses.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

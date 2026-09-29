"""Compare exact scores, run a sampling pilot, or expand it to the full study subset.

Examples: python run_replication.py compare
          python run_replication.py sample --rows 3
          python run_replication.py sample --rows 0 --shard 0 --shards 8
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"C:\Users\praney\OneDrive\Desktop\generalized-surprisal\data")
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
KEYS = ("sent_id", "context_length", "word")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    from uncertain.store import atomic_write
    import io
    if not rows:
        raise ValueError("Cannot write an empty results table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode("utf-8"))


def write_json(path, value):
    from uncertain.store import atomic_write
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))


def key(row):
    return tuple(row[k] for k in KEYS)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_rows(args):
    original = read_csv(args.data_dir / "psyling/aligned.csv")
    filtered = [r for r in original if not any(c in r['word'] for c in ".,'")]
    reference = read_csv(args.data_dir / "psyling/aligned_with_estimates_v2.csv")
    expected = {key(r) for r in reference}
    if len(expected) != len(reference) or {key(r) for r in filtered} != expected:
        raise ValueError("Notebook membership does not match the documented punctuation filter")
    if len({key(r) for r in filtered}) != len(filtered):
        raise ValueError("Duplicate observation keys")
    return filtered, reference


def compare(args):
    filtered, reference = analysis_rows(args)
    exact_dir = args.output_dir / "exact"
    exact = read_csv(exact_dir / "scores.csv")
    manifest = json.loads((exact_dir / "manifest.json").read_text())
    if not manifest.get("prepend_bos") or manifest['model'] != 'openai-community/gpt2':
        raise ValueError("Comparison requires GPT-2 exact scores with --prepend-bos")
    if manifest['source_sha256'] != checksum(args.data_dir / "psyling/aligned.csv"):
        raise ValueError("Source checksum changed since scoring")
    index = {key(r): r for r in exact}
    if len(index) != len(exact):
        raise ValueError("Duplicate exact-score keys")
    output = []
    for ref in reference:
        row = index[key(ref)]
        if row['uncertain_status'] != 'ok':
            raise ValueError(f"Failed exact score at {key(row)}")
        output.append({**row,
                       'surprisal_gpt2_small': row['uncertain_surprisal_bits'],
                       'prob_gpt2_small': row['uncertain_probability'],
                       'reference_surprisal_bits': ref['surprisal_gpt2_small'],
                       'reference_probability': ref['prob_gpt2_small']})
    report = {'rows': len(filtered), 'join_keys': KEYS,
              'reference_sha256': checksum(args.data_dir/'psyling/aligned_with_estimates_v2.csv'),
              'exact_scores_sha256': checksum(exact_dir/'scores.csv'), 'metrics': {}}
    for ours, ref, tol in [('surprisal_gpt2_small', 'reference_surprisal_bits', 1e-3),
                           ('prob_gpt2_small', 'reference_probability', 1e-4)]:
        a = np.array([float(r[ours]) for r in output])
        b = np.array([float(r[ref]) for r in output])
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("Nonfinite comparison values")
        delta = np.abs(a-b)
        report['metrics'][ours] = {'mean_absolute_error': float(delta.mean()),
                                   'max_absolute_error': float(delta.max()),
                                   'pearson': float(np.corrcoef(a, b)[0, 1]),
                                   'absolute_tolerance': tol,
                                   'rows_outside_tolerance': int(np.sum(delta > tol))}
    report['passed'] = all(m['rows_outside_tolerance'] == 0 for m in report['metrics'].values())
    write_csv(exact_dir/'notebook_scores.csv', output)
    write_json(exact_dir/'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit("Exact replication comparison failed")


def sample(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import uncertain
    from uncertain import HFBackend, AlternativeStore, SamplingSpec, Stimulus, UnitSpec
    from uncertain.emnlp2024 import StudySampler, GPT2StudyEncoder, estimate_study, normalize
    from uncertain.utils import digest, seed_for
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    filtered, _ = analysis_rows(args)
    exact_dir = args.output_dir/'exact'
    comparison = json.loads((exact_dir/'comparison.json').read_text())
    if not comparison['passed']:
        raise ValueError("Exact comparison must pass before sampling")
    exact = {key(r): r for r in read_csv(exact_dir/'notebook_scores.csv')}
    if comparison['exact_scores_sha256'] != checksum(exact_dir/'scores.csv'):
        raise ValueError("Rerun exact comparison after changing exact results")
    if comparison['reference_sha256'] != checksum(args.data_dir/'psyling/aligned_with_estimates_v2.csv'):
        raise ValueError("Rerun exact comparison after changing the notebook reference")
    if args.rows:
        # Stratify on exact surprisal, deterministically, with no manual cherry-picking.
        ordered = sorted(filtered, key=lambda r: float(exact[key(r)]['surprisal_gpt2_small']))
        chosen = np.linspace(0, len(ordered)-1, min(args.rows, len(ordered)), dtype=int)
        filtered = [ordered[int(i)] for i in chosen]
    filtered = filtered[args.shard::args.shards]
    if not filtered:
        raise ValueError("No observations selected")
    backend = HFBackend('openai-community/gpt2', revision=REVISION, device=args.device,
                        bos_token_id=50256, eos_token_id=50256)
    sampler = StudySampler(backend, batch_size=args.batch_size)
    encoder = GPT2StudyEncoder(backend)
    package_dir = Path(uncertain.__file__).resolve().parent
    package_sources = {str(p.relative_to(package_dir)): checksum(p)
                       for p in sorted(package_dir.rglob('*.py'))}
    config = {'model': canonical_model(backend), 'sampler': sampler.descriptor,
              'backend': backend.descriptor, 'bootstrap': args.bootstrap,
              'sample_sizes': args.samples, 'horizons': args.horizons, 'seeds': args.seeds,
              'rows': [key(r) for r in filtered], 'shard': args.shard, 'shards': args.shards,
              'data_sha256': checksum(args.data_dir/'psyling/aligned.csv'),
              'notebook_scores_sha256': checksum(exact_dir/'notebook_scores.csv'),
              'source_sha256': checksum(ROOT/'src/uncertain/emnlp2024.py'),
              'imported_package_sources': package_sources,
              'runner_sha256': checksum(__file__),
              'uncertain_version': uncertain.__version__, 'threads': args.threads}
    from uncertain.utils import canonical
    config = canonical(config)
    run_dir = args.output_dir/'sampling'/digest(config)[:16]
    write_json(run_dir/'config.json', config)
    started = time.perf_counter()
    outputs = []
    checks = []
    oracles = []
    completed = 0
    cache_hits = 0
    total = sum(len(backend.encode(r['item'].rstrip()+' '+r['word'].lstrip()))
                - len(backend.encode(r['item'].rstrip())) for r in filtered)*len(args.seeds)
    for row in filtered:
        stimulus = Stimulus.from_text(row['item'].rstrip(), ' '+row['word'].lstrip(), backend)
        stimulus = replace(stimulus, context=(50256,)+stimulus.context)
        for offset, target in enumerate(stimulus.target):
            context = stimulus.context + stimulus.target[:offset]
            # Analytic next-symbol oracles under the actual proposal q.
            p = backend.probabilities(context)
            q = p.copy()
            q[50256] = 0
            q /= q.sum()
            z = normalize(encoder.encode_tokens(np.arange(len(p))))
            mean_z = q @ z
            target_z = z[target]
            analytic = {'when_first_token': float(q[target]),
                        'decontextualised': float(1-mean_z@target_z),
                        'expected_decontextualised': float(1-mean_z@mean_z),
                        'entropy_sampling_bits': float(-np.sum(q[q>0]*np.log2(q[q>0]))),
                        'cross_entropy_model_bits': float(-np.sum(q[q>0]*np.log2(p[q>0])))}
            oracles.append({'sent_id': row['sent_id'], 'context_length': row['context_length'],
                            'word': row['word'], 'token_offset': offset,
                            'raw_model_probability': float(p[target]),
                            'raw_model_entropy_nats': float(-np.sum(p[p>0]*np.log(p[p>0]))),
                            'raw_model_expected_probability': float(p@p),
                            'first_eos_probability': float(p[50256]),
                            **{f'analytic_q_{k}': v for k, v in analytic.items()}})
            del z
            for seed in args.seeds:
                spec = SamplingSpec(backend.model_spec, max(args.samples), seed,
                                    token_budget=max(args.horizons))
                # Per-context store bounds retained in-memory artifacts in full runs.
                store = AlternativeStore(args.output_dir/'cache')
                hit = store.available(sampler, context, max(args.horizons), spec, UnitSpec())
                cache_hits += int(hit)
                a = store.get_or_sample(sampler, context, max(args.horizons), spec, UnitSpec())
                artifact = a.content_hash
                # Independent full-forward oracle for sampled log probabilities.
                teacher_error = max(float(np.max(np.abs(
                    backend.score(context, r.tokens) - np.asarray(r.model_logprobs))))
                    for r in a.sequences[:3])
                if teacher_error > 2e-4:
                    raise ValueError(f"Teacher-forced probability mismatch: {teacher_error}")
                checks.append({'sent_id': row['sent_id'], 'context_length': row['context_length'],
                               'word': row['word'], 'token_offset': offset, 'seed': seed,
                               'teacher_max_abs_error_nats': teacher_error,
                               'first_eos_count': sum(r.tokens[0] == 50256 for r in a.sequences),
                               'later_eos_count': sum(r.stop_reason == 'eos' for r in a.sequences),
                               'cache_hit': hit, 'artifact': artifact})
                for h in args.horizons:
                    for n in args.samples:
                        cut = a.truncate(h).select(range(n))
                        estimates = estimate_study(cut, target, encoder, bootstrap=args.bootstrap,
                                                   seed=seed_for(seed, context, n, 'bootstrap'))
                        for metric, result in estimates.items():
                            if h != min(args.horizons) and metric in {
                                'when_first_token', 'mc_surprisal_bits', 'decontextualised',
                                'expected_decontextualised'}:
                                continue
                            oracle = analytic.get(metric) if h == 1 or metric in {
                                'when_first_token', 'decontextualised', 'expected_decontextualised'} else None
                            outputs.append({'sent_id': row['sent_id'],
                                'context_length': row['context_length'], 'word': row['word'],
                                'token_offset': offset, 'target_token_count': len(stimulus.target),
                                'target_id': target, 'seed': seed, 'n': n, 'horizon': h,
                                'metric': metric, 'value': result.value, 'se': result.se,
                                'ci_low': result.ci_low, 'ci_high': result.ci_high,
                                'status': result.status, 'analytic_q_value': oracle,
                                'absolute_error': abs(result.value-oracle) if oracle is not None else None,
                                'analytic_in_interval': (result.ci_low <= oracle <= result.ci_high
                                                         if oracle is not None else None),
                                'artifact': artifact})
                completed += 1
                # Atomic checkpoints preserve already computed scores after interruption.
                write_csv(run_dir/'scores.csv', outputs)
                write_csv(run_dir/'validation.csv', checks)
                print(f"Sampling {completed}/{total}; {time.perf_counter()-started:.1f}s; "
                      f"cache hits {cache_hits}; output {run_dir.name}", flush=True)
    summary = {'completed': True, 'observations': len(filtered), 'token_seed_jobs': completed,
               'result_rows': len(outputs), 'cache_hits': cache_hits, 'sampler_calls': sampler.calls,
               'elapsed_seconds': time.perf_counter()-started,
               'teacher_max_abs_error_nats': max(c['teacher_max_abs_error_nats'] for c in checks),
               'first_eos_count': sum(c['first_eos_count'] for c in checks),
               'later_eos_count': sum(c['later_eos_count'] for c in checks),
               'scope': 'GPT-2 token-level estimates; full means all 1487 analysis observations',
               'uncertainty': '95% intervals; pairwise bounded CI and whole-alternative bootstrap SE',
               'sampling_law': 'q=p conditioned on first token not EOS; stop at subsequent EOS',
               'historical_bitwise_replication': False}
    diagnostics = []
    for metric in sorted({r['metric'] for r in outputs}):
        for n in args.samples:
            rr = [r for r in outputs if r['metric'] == metric and r['n'] == n
                  and r['analytic_q_value'] is not None]
            if rr:
                diagnostics.append({'metric': metric, 'n': n, 'comparisons': len(rr),
                                    'mean_absolute_error': float(np.mean([r['absolute_error'] for r in rr])),
                                    'interval_coverage': float(np.mean([r['analytic_in_interval'] for r in rr]))})
    write_csv(run_dir/'convergence.csv', diagnostics)
    write_csv(run_dir/'oracles.csv', oracles)
    write_json(run_dir/'summary.json', summary)
    write_json(args.output_dir/'sampling/latest.json', {
        'run_id': run_dir.name, 'scores_sha256': checksum(run_dir/'scores.csv'),
        'summary': summary})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Run directory: {run_dir}", flush=True)


def canonical_model(backend):
    return {'model_id': backend.model_spec.model_id, 'revision': backend.model_spec.revision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['compare', 'sample'])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'replication')
    parser.add_argument('--rows', type=int, default=3, help='Stratified pilot rows; 0 means all')
    parser.add_argument('--samples', type=int, nargs='+', default=[4,8,16,32,64,128,256,512])
    parser.add_argument('--horizons', type=int, nargs='+', default=[1,5,10,15])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1])
    parser.add_argument('--bootstrap', type=int, default=199)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if (args.rows < 0 or min(args.samples) < 2 or min(args.horizons) < 1
            or args.bootstrap < 20 or args.threads < 1 or args.batch_size < 1
            or not 0 <= args.shard < args.shards):
        parser.error('Invalid experiment dimensions')
    args.samples = sorted(set(args.samples))
    args.horizons = sorted(set(args.horizons))
    args.seeds = sorted(set(args.seeds))
    if args.command == 'compare':
        compare(args)
    else:
        sample(args)


if __name__ == '__main__':
    main()

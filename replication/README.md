# Replication workflow

Run these commands in PowerShell from `C:\Users\praney\OneDrive\Desktop\uncertain`.
The GPT-2 snapshot is cached; the runners enforce offline Hugging Face loading.

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe run_aligned_dataset.py --prepend-bos
.\.venv\Scripts\python.exe run_replication.py compare
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_replication.py sample --rows 3
```

Exact outputs:

- `exact/scores.csv`: all 1,726 scored observations, with BOS.
- `exact/notebook_scores.csv`: 1,487 matching notebook observations; compatible
  `surprisal_gpt2_small` (bits) and `prob_gpt2_small` column names.
- `exact/comparison.json`: complete exact-score comparison and pass/fail thresholds.
- `exact/manifest.json`: source checksum and execution settings.

Sampling outputs are under `sampling/<configuration-hash>/`:
`sampling/latest.json` identifies the latest completed run and its score checksum.

- `config.json`: immutable model revision, dataset/source checksums, selected rows,
  seeds, sample sizes, horizons, batch size and execution settings.
- `scores.csv`: token-level estimates, SEs, confidence intervals, analytic q values,
  and alternative-artifact identifiers. Infinity at zero Monte Carlo hits is explicit.
- `validation.csv`: teacher-forced log-probability checks, EOS counts, cache reuse.
- `convergence.csv`: errors and interval coverage against analytic next-symbol values.
  A small pilot is a diagnostic, not a statistical coverage study.
- `oracles.csv`: analytic q expectations, raw-model entropy and expected probability,
  and the probability mass excluded by first-step EOS suppression.
- `summary.json`: created on successful completion. An interrupted run may have
  partial CSV checkpoints but no completed summary.

Alternative artifacts are stored in `cache/`. Repeat the same sampling command to
verify regeneration-free replay. Results use the same cached samples across N/H.

To expand to every notebook observation (long-running on CPU):

```powershell
.\.venv\Scripts\python.exe run_replication.py sample --rows 0 --seeds 0
```

For partitioned execution, launch each shard index from 0 to M-1; for example:

```powershell
.\.venv\Scripts\python.exe run_replication.py sample --rows 0 --seeds 0 --shard 0 --shards 8
```

GPU execution is available with `--device cuda` in a CUDA-enabled PyTorch environment.
The current virtual environment uses CPU PyTorch. Device/batching changes create a
distinct execution identity and samples need not be bitwise equal across hardware.

The same-sample original-helper comparison additionally needs SciPy and access to
the original source; run `validate_study_reference.py` in such an environment.
It performs no model downloads. See `docs/replication-plan.md` for definitions,
intentional differences, multi-token aggregation and remaining scope.

The original-helper validation was executed with the system `python`, which has
SciPy installed, whereas the project virtual environment does not. To repeat it
on this machine after setting `PYTHONPATH=src`:

```powershell
python validate_study_reference.py
```

The full filtered corpus contains 1,493 target-token contexts. The cold CPU pilot
took about 392 seconds for eight context/seed jobs. A simple extrapolation is about
20 hours per seed for the full corpus, before accounting for context/cache reuse
and differing context lengths. Use this as a planning estimate, not a benchmark.

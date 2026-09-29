# Replication progress — September 9, 2026

## Completed

1. **Exact-score replication across the full analysis subset.** Scored all 1,726
   observations with the notebook's BOS convention. All 1,487 notebook observations
   pass the comparison thresholds. Surprisal MAE is 0.00001821 bits; maximum error
   is 0.00009932 bits. Probability MAE is 0.000000873; maximum error is 0.00001072.

2. **Study-specific estimator implementation and validation.** Implemented observed
   information value, expected next-token information value, expected sequence
   information value, Monte Carlo probability/surprisal, and stopped-sequence
   entropy/cross-entropy with explicit sampling-law labels. Original helper
   functions agree on identical samples: maximum point-estimator difference
   0.000000068 in the tiny-model oracle. Tests cover EOS, duplicate draws, actual
   layer-0 embeddings, off-diagonal means, whole-alternative bootstrap, sampling
   replay/extension, teacher-forced probabilities and distribution-label safeguards.

3. **Executed a reproducible sampling pilot and implemented expansion.** Three
   observations (including one multi-token word), two seeds, eight sample sizes
   from 4 through 512, and four horizons (1, 5, 10, 15) produced 1,024 result rows.
   The eight token/seed jobs took approximately 392 seconds on a cold cache.
   Sampled log probabilities agree with independent full-forward scoring within
   0.000026 nats. There were zero first-step EOS draws and 23 later EOS terminations.

At N=512, mean information-value error against vocabulary-wide analytic q
expectations was 0.000530, compared with 0.005536 at N=4. This is a small pilot,
not a general estimator-coverage claim. Zero-hit surprisal remains explicitly
infinite. Pairwise confidence intervals account for dependent distances.

## Verification

- Full current suite: **78 passed**, no failures or skips.
- Ruff passes for source, tests, examples and the three replication runners.
- Mypy passes for all 22 source files.
- The new module imports from the project's source in the existing virtual environment.
- Cache replay uses all eight cached alternative sets and makes **zero sampler calls**.
  Replay scores are byte-identical to the cold pilot. Replays took 12–15 seconds.
- `reference_validation.json` records the same-sample original-helper check and its
  environment, which uses system Python/SciPy. The sampling pilot uses the project's
  CPU virtual environment.

## Files to inspect

- `exact/comparison.json` — full-table exact validation.
- `exact/notebook_scores.csv` — exact scores aligned with the notebook's rows/names.
- `reference_validation.json` — original-helper point-estimator comparison.
- `sampling/latest.json` — latest completed run ID and score checksum.
- `sampling/47f8d9725607cf9d/` — final source-version pilot replay, results and diagnostics.
- `sampling/e16d0d9a3e0404ff/` — initial cold pilot used for the runtime measurement.
- `../docs/replication-plan.md` — definitions, compatibility decisions and acceptance gates.
- `README.md` — exact, pilot, replay and full-dataset commands.

## Remaining execution and research scope

The full-dataset sampling run is configured but **has not been executed**. It covers
1,493 target-token contexts across 1,487 observations. A rough extrapolation from
the pilot is about 20 CPU hours per seed, with substantial uncertainty from context
lengths and cache reuse. The runner supports observation shards and CUDA environments.

Sampling results are at token resolution; the notebook's word-level aggregation,
lagged predictors and downstream regressions remain outside this pilot. GPT-Neo
also needs its own sampler/encoder audit. Historical padding-log-probability bugs,
epsilon-smoothed Monte Carlo surprisal and the original dependent-row bootstrap
are not silently treated as equivalent to the corrected implementations.

The resulting claim is **full exact-score validation plus audited study estimators
and an executed GPT-2 sampling pilot**, not full reproduction of the paper.

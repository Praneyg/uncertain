# Replication plan: stages 1–3

The reference is the local `generalized-surprisal` implementation and the data
used by `analysis/emnlp2024/analysis.ipynb`. This workflow targets the pinned GPT-2
small model. It does not claim bitwise reproduction of historical random samples.

## 1. Exact-score replication

Implemented in `run_aligned_dataset.py --prepend-bos` and
`run_replication.py compare`.

- Preserve the literal text `None` with CSV parsing.
- Prepend GPT-2 BOS (50256) to every context.
- Score the complete target token sequence with the existing library exact path.
- Retain all 1,726 observations in the complete output.
- Match the notebook's 1,487 observations by `(sent_id, context_length, word)`.
  Confirm that their membership equals exclusion of periods, commas, and apostrophes.
- Compare probability and surprisal in bits to `aligned_with_estimates_v2.csv`.
- Require every comparison to pass absolute tolerances of 0.001 bits and 0.0001
  probability. These tolerances account for float32 execution differences and
  were set before running the complete comparison.

Completed: all observations pass. See `replication/exact/comparison.json`.
The earlier no-BOS output in `aligned_gpt2` is retained as a distinct experiment.

## 2. Generalized-measure definitions and validation

Implemented in `uncertain.emnlp2024`, with dedicated tests and a same-sample
comparison script, `validate_study_reference.py`.

Let p be raw GPT-2 and q be p conditioned on the first token not being EOS.
Subsequent steps use raw p until EOS or the requested horizon. Let e(t) be GPT-2's
token embedding plus its position-zero embedding; d is cosine distance.

| Original name | Implemented definition | Compatibility |
|---|---|---|
| `prob`, `surprisal` | p(target sequence given BOS + context), -log2(p) | Exact scores validated across all analysis rows |
| `when_first_token` | Mean indicator of first sampled token matching target | Same next-token point formula; expectation under q |
| `neg_log2_when_first_token` | Original: -log2(frequency + 0.0001) | Library retains unsmoothed -log2(frequency), including infinity at zero; named `mc_surprisal_bits` |
| `decontextualised` | Mean d(e(sampled first token), e(target)) | Same point formula; NOT generic energy distance |
| `expected_decontextualised` | Mean d(e(U_i), e(U_j)) over i != j | Same off-diagonal U-statistic; duplicate draws remain distinct indices |
| `expected_seq_decontextualised` | Same U-statistic over mean-pooled layer-0 sequence embeddings | Local positions start at zero; EOS excluded from pool |
| `sequence_entropy` | Original averages raw-model sequence surprisal on q samples | Corrected stopped-sequence counterpart named `cross_entropy_model_bits` |
| additional `entropy_sampling_bits` | Mean -log2(q(sequence)) | Entropy of the actual proposal law, not labeled as raw-p entropy |
| exact `entropy`, `expected_prob` | -sum p log p (nats), sum p squared | Definitions identified; not columns in the initial exact-score run |

Scientific distinctions that remain explicit:

- The generic library's `mode="anticipatory"` is a nested expectation; it is not
  used as a substitute for these off-diagonal study estimators.
- `StudySampler` is rejected by generic `Measure` to prevent accidentally labeling
  its conditional q samples as raw-model p samples. Use `estimate_study` with it.
- Original GPT-2 layer 0 includes positional embeddings. Raw word embedding lookup
  alone and the generic context-conditioned layer encoder are not equivalent.
- The historical sampling code can include padded EOS log probabilities after a
  sequence has ended. This implementation stops scoring at the first EOS. It
  records termination frequency rather than preserving that artifact.
- The original pairwise bootstrap resamples dependent row means. This implementation
  resamples complete alternatives and recomputes both axes. Reported bootstrap SE
  is approximate; bounded-difference confidence intervals remain nonzero even when
  all observed embeddings coincide. Legacy bootstrap variances are not a target.
- For mean information value and sequence (cross-)entropy, intervals use the
  library's normal approximation. Probability uses Wilson intervals, including
  at zero hits; surprisal transforms those interval endpoints.

Validation gates: independent teacher-forced log probabilities; p/q normalization;
EOS termination; fixed-width batched sample extension and horizon reuse; actual
GPT-2 hidden-state-zero oracles; explicit pairwise matrix enumeration; bootstrap
resampling on both axes; duplicate and degenerate cases; original point-estimator
helpers on identical draws with duplicate/EOS cases.

## 3. Sampling pilot and expansion

Implemented in `run_replication.py sample`.

- Default pilot: three observations chosen at evenly spaced exact-surprisal ranks,
  two seeds (0 and 1), N = 4, 8, 16, 32, 64, 128, 256, 512 and H = 1, 5, 10, 15.
- Sample N_max continuations at H_max once per context/seed; reuse indexed samples
  across smaller N and H. Grid cells are therefore correlated, not independent
  replications. Seeds provide independent runs.
- Use fixed-width batches and a KV cache to avoid one full forward pass per draw
  per token. Batch size, model revision and execution settings enter cache identity.
- For next-symbol probability, information value and expected information value,
  compute vocabulary-wide analytic q expectations as numerical oracles. Also check
  one-token entropy and raw-model cross-entropy against their analytic q values.
- Save complete alternative artifacts, per-cell uncertainty, EOS counts, sampled
  log-probability checks, convergence diagnostics, source checksums and configuration.
- Write atomic score checkpoints. Repeating a run reuses cached alternatives.
- Expand with `--rows 0`; optional `--shard K --shards M` partitions observations
  across jobs. Changing source or execution configuration creates a new run ID.

Sampling outputs are at target-token resolution, with explicit `token_offset` and
`target_token_count`. Every token of a multi-token target is sampled in its own
conditional context. Before replacing the notebook's word-level sampling columns,
aggregate responsive information values as sums and anticipatory metrics at the
first target token, as in the original scorer. The original sum of first-token hit
frequencies across subtokens is not a joint word probability. Word-level confidence
intervals and lagged predictors require separate aggregation; they are not claimed
by the token-level pilot.

Completion boundary: the pilot exercises the notebook's sample sizes and horizons.
The full-dataset sampling grid and the second model (GPT-Neo) are not claimed as
completed by a GPT-2 pilot. Full-dataset execution is available through the runner;
adding GPT-Neo requires a model-specific batched sampler/encoder audit first.

# Generalized surprisal (EMNLP 2024) rebuilt on `uncertain`

This folder re-implements [rycolab/generalized-surprisal](https://github.com/rycolab/generalized-surprisal) (Giulianelli, Opedal & Cotterell, Findings of EMNLP 2024). Every estimator is computed through `uncertain.Measure`, not hand-written. The run uses the paper's main configuration: N=512 samples, H=5 tokens, temperature 1, no top-k. It covers GPT-2 small and GPT-Neo 125M on all 205 texts of the Aligned dataset (2,001 token contexts per model). The results are compared with the original estimates (`data/psyling/aligned_with_estimates_v2.csv`) and with the original cross-validation, which was rerun here on both sets of estimates.

**Result.**
- The four exact measures reproduce the original to within float32 noise.
- The four sampling measures correlate r = 0.974–0.999 with the original, with differences the size of Monte Carlo noise.
- Across all 494 (model × analysis × predicted variable × measure) regression cells, ΔR² correlates at r ≥ 0.9998 between original and new estimates.
- Significance at p < .05 agrees in 485 of 494 cells. The 9 that flip are all sampling measures with |ΔR²| < 0.001 and p near the threshold.
- The paper's qualitative conclusions are unchanged.

## How each measure is computed

All measures are `Measure.from_config(...)` / `.replace(...)` variants sharing one sampler and one `AlternativeStore`, evaluated per token on `Stimulus(context=BOS+prefix, target=(token,))`. Within a context, all four sampling measures use **the same 512 draws**. The H=1 measures receive the store's truncation of the H=5 set, which is what the original does: it uses the first token of each H=5 sample.

| Original column | Library configuration | Word aggregation |
|---|---|---|
| `surprisal` (bits) | exact, `prefix_indicator` + `mean` + `neglog`, ÷ ln 2 | sum |
| `prob` | exact, `prefix_indicator` + `mean` | product |
| `entropy` (nats) | exact, `mode="entropy"`, H=1 | first token |
| `expected_prob` (Σp²) | `backend.probabilities`; equals exact discrete energy − 1 + 2p(y) (asserted in `check.py`) | first token |
| `decontextualised` (information value) | MC, H=1, `layer0_local` + `cosine_fast` + `mean` | sum |
| `expected_decontextualised` | MC, H=1, `layer0_local` + `cosine_fast` + `within_u` | first token |
| `expected_seq_decontextualised` | MC, H=5, same with `within_u`, target set to EOS so no continuation is sampled | first token |
| `sequence_entropy` (bits) | MC, `mode="entropy"`, H=5, `probability="model"`, ÷ ln 2 | first token |

Word aggregation follows the original `word_score` rules. The join with the human data (`analysis.py`) ports `apply_estimates_to_dataset.py`, including its punctuation and clitic filter (1,487 rows) and its t−1/t−2 lags. The port keeps one quirk: a lag of −1 indexes the *last* word of the sentence, because the original uses Python list indexing. The CV is a port of `crossvalidation.ipynb` cells 9, 11 and 13:
- 100 seeds × 10-fold OLS on min-max-scaled predictors;
- test R², with the same NaN handling as `smf.ols` and pandas sums;
- significance from the paired permutation test of `analysis.ipynb`, with 10,000 resamples.

`np.linalg.lstsq` replaces the statsmodels formula API for speed; `check_ols` asserts that both give the same R². Cell 11 of the original notebook has a lost `for predicted_var in …` header; its evident intent is implemented.

### Plugins (`plugins.py`, registered through the library's extension API; `src/uncertain` is untouched)

| Component | Why it was needed | Check (`check.py`) |
|---|---|---|
| `BatchedSampler(Sampler)` | The reference sampler runs one full forward pass per prefix, with no KV cache (about 10 h per model). This sampler uses KV-cache batches of 512 with the same indexed per-draw RNG streams and the same inverse-CDF draw as `Generator.choice`. | 47/48 draws token-identical to `uncertain.Sampler` on shared streams; the one difference is a float32 near-tie |
| `layer0_local` encoder | The built-in `LayerEncoder(context_mode="isolated")` prepends BOS, so token positions start at 1. GPT-2's position-0 embedding is atypical, so this changes every cosine distance. The plugin uses `hidden_states[0]` of the bare alternative (positions from 0), mean over non-EOS tokens, as the original does. | Equals `model(ids).hidden_states[0]` to 1e-5 |
| `cosine_fast` distance | Same formula as `cosine` without per-call numpy overhead; `pairwise_blocks` calls the distance once per pair. | Equals `cosine` to 1e-12 |
| `within_u` statistic | The library has no target-free within-set statistic. `mode="anticipatory"` enumerates the full vocabulary with inner MC, which is infeasible here, and the U-energy route needs 512² Python distance calls. The plugin computes the off-diagonal mean in closed form, with bootstrap replicates from the library's own `weights`. | Equals `emnlp2024.cosine_u` and 2·mean − U-energy (built-in route), including identical bootstrap replicates |

## Results

### Estimates (1,487 analysis rows, new vs original)

| Measure | GPT-2 r (Pearson / Spearman) | GPT-2 MAE | Neo r (Pearson / Spearman) | Neo MAE |
|---|---|---|---|---|
| surprisal | 1.0000 / 1.0000 | 1.1e-5 bits (max 8.0e-5) | 1.0000 / 1.0000 | 7.5e-6 bits (max 8.2e-5) |
| prob | 1.0000 / 1.0000 | 5.0e-7 | 1.0000 / 1.0000 | 3.5e-7 |
| entropy | 1.0000 / 1.0000 | 5.3e-6 nats | 1.0000 / 1.0000 | 3.6e-6 nats |
| expected_prob | 1.0000 / 1.0000 | 6.9e-7 | 1.0000 / 1.0000 | 4.1e-7 |
| decontextualised | 0.9987 / 0.9986 | 0.0008 | 0.9967 / 0.9952 | 0.0003 |
| expected_decontextualised | 0.9963 / 0.9957 | 0.0013 | 0.9918 / 0.9880 | 0.0004 |
| expected_seq_decontextualised | 0.9888 / 0.9836 | 0.0008 | 0.9735 / 0.9550 | 0.0002 |
| sequence_entropy | 0.9893 / 0.9871 | 0.49 bits | 0.9886 / 0.9872 | 0.52 bits |

The sampling-measure differences are consistent with independent Monte Carlo draws. For example, the library's per-estimate SE for `sequence_entropy` has a median of 0.43 bits, and the MAE between two independent estimates is about 0.49 bits. Means agree to three decimals, except `sequence_entropy`, which is 0.06 bits (GPT-2) and 0.08 bits (Neo) lower than the original, about 0.2%. Rows with and without EOS draws show the same gap, so EOS handling is not the cause. The gap remains unexplained.

### Cross-validation: ΔR² on held-out folds

| Model | Analysis | Cells | Significant (orig / new) | Significance agrees | r(ΔR²) |
|---|---|---|---|---|---|
| GPT-2 | measure over base predictors | 104 | 64 / 62 | 102 | 0.9999 |
| GPT-2 | measure over base + surprisal | 65 | 25 / 25 | 65 | 0.9999 |
| GPT-2 | swap surprisal / entropy | 78 | 31 / 29 | 76 | 0.9999 |
| Neo | measure over base predictors | 104 | 69 / 70 | 103 | 1.0000 |
| Neo | measure over base + surprisal | 65 | 23 / 23 | 65 | 0.9998 |
| Neo | swap surprisal / entropy | 78 | 33 / 33 | 74 | 0.9999 |

Exact-measure cells have ΔR² identical to the original. The largest absolute ΔR² difference in any sampling cell is 0.005, for `sequence_entropy`.

**Headline cells, GPT-2, measure over base predictors** (ΔR² × 1000, original / new):

| Predicted variable | surprisal | prob | info. value | entropy | exp. info. value (next token) | exp. info. value (sequence) | sequence entropy |
|---|---|---|---|---|---|---|---|
| Cloze probability | 209.9 / 209.9 | 475.9 / 475.9 | 220.8 / 217.4 | 120.3 / 120.3 | 100.3 / 99.0 | 49.4 / 50.8 | 60.0 / 63.3 |
| Predictability rating | 303.7 / 303.7 | 161.3 / 161.3 | 141.6 / 141.7 | 80.0 / 80.0 | 47.6 / 48.6 | 49.8 / 51.0 | 59.4 / 61.2 |
| Cloze entropy | 107.0 / 107.0 | 218.9 / 218.9 | 153.4 / 150.1 | 268.2 / 268.2 | 195.9 / 194.5 | 138.7 / 137.9 | 147.9 / 152.9 |
| Self-paced RT | 31.6 / 31.6 | 6.7 / 6.7 | 23.2 / 23.9 | 5.1 / 5.1 | 5.7 / 6.3 | 2.9 / 3.2 | 6.9 / 7.3 |
| N400 | 20.2 / 20.2 | 10.6 / 10.6 | **27.7 / 27.8** | 3.7 / 3.7 | 5.8 / 5.9 | 1.9 / 1.1 | −0.2 / −0.4 |
| First-pass RT | 5.2 / 5.2 | 3.4 / 3.4 | 1.0 / 0.9 | −0.6 / −0.6 | −1.1 / −1.0 | 0.9 / 1.0 | −0.7 / −0.2 |

Both sets of estimates support the paper's conclusions:
- Information value beats surprisal for the N400 with GPT-2.
- The anticipatory measures carry most of the signal for cloze entropy.
- Surprisal dominates reading times.

The full grid, with CIs and p-values, is in `results/cv_summary.csv` and `results/cv_comparison.csv`.

**The 9 cells where significance flips:** all have |ΔR²| ≤ 0.0007 and involve sampling measures. GPT-2:
- base: P600 and PNP with sequence expected information value (original significant);
- surprisal/entropy swap: right-bounded RT with next-token expected information value; ELAN with information value.

Neo:
- base: EPNP with information value (new significant);
- surprisal/entropy swap: first-pass RT with information value; first-pass and right-bounded RT with sequence expected information value; ELAN with information value.

## Deliberate differences from the original

1. **Sampling law.** The library samples raw p. The original forbade EOS as the first token (`begin_suppress_tokens`). Out of 1.02M draws per model, 233 (GPT-2) and 106 (Neo) began with EOS. An EOS-only alternative is represented by EOS's own embedding.
2. **Sequence entropy stops at EOS.** The original also summed the log-probabilities of padding EOS tokens after a sequence ended.
3. **Different random draws.** Agreement is therefore statistical, not bitwise. The library additionally reports an SE per estimate (in `token_estimates.csv`). The original's per-row bootstrap summaries are not reproduced.

## Library findings for `uncertain`

- **No batched or KV-cache sampler.** The reference `Sampler` is too slow for N=512 over a corpus. `BatchedSampler` here is a drop-in candidate.
- **`LayerEncoder(context_mode="isolated")` shifts positions.** It prepends BOS (`HFBackend.hidden` → `_prefix`), so positions start at 1, and it runs a full forward pass per alternative.
- **`pairwise_blocks` loops in Python.** One distance call per pair: 524k calls per context for the two within-set terms. A vectorized `pairwise(X, Y)` hook on distances would remove this.
- **Slow exact enumeration.** `finite_expectation` for exact discrete energy builds one `Stimulus` per vocabulary item (about 6 s/token). There is also no Σp² (expected probability) preset.
- **No target-free within-set statistic.** The study's expected-information-value measures need E d(X, X′), not a nested anticipatory expectation.
- **Unvectorized cosine.** The built-in `cosine` uses `np.linalg.norm` and `np.clip` on scalars, about 10× slower per call than needed.

## Reproduce

Use the system Python (CUDA torch, scikit-learn) from the repo root. Runs are resumable per text. Run the two models one after the other: in parallel they time-slice the GPU and are about 6× slower.

```powershell
$env:PYTHONPATH='src'; $env:HF_HUB_OFFLINE='1'
python "emnlp imp\check.py"                       # plugin + identity checks
"emnlp imp\run_model.cmd" gpt2-small              # about 50 min on an RTX 3060 laptop
"emnlp imp\run_model.cmd" neo-125m                # about 58 min
python "emnlp imp\analysis.py"                    # join, CV on both sources, comparison (about 40 min)
```

Outputs are in `results/`:
- `<model>/token_estimates.csv` and `<model>/word_estimates.csv`, plus `manifest.json` (pinned revisions; sampler, encoder and backend descriptors);
- `aligned_with_estimates_uncertain.csv` (same column names as the original);
- `estimate_agreement.csv`;
- `cv_{original,uncertain}_<model>.csv` (per fold);
- `cv_summary.csv` and `cv_comparison.csv`.

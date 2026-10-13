# Conformance report: `uncertain` vs. the Library Design Proposal

Spec: `π-infoval-library (1).pdf`. Code: `src/uncertain` working tree at commit `656d364`, **including
uncommitted changes**. Model for integration: `openai-community/gpt2` @ `607a30d7…`, CPU, offline.
Every number below comes from `eval/results.json` / `eval/mutations.json` produced by the commands
under "How to rerun".

## 1. Verdict

**The implementation does not fully meet the spec.** The numerical core is sound. Every analytic
oracle test passes, including H = 1 information value, H > 1 operand composition, energy distance
properties, presets, closed form vs Monte Carlo on the toy model and on GPT-2, and log-prob
alignment against teacher-forced passes. Sampling, hashing, persistence and sweep reuse also behave
as specified. **42 requirements graded: 31 PASS, 7 FAIL, 3 AMBIGUOUS, 1 NOT IMPLEMENTED** (271 tests: 252 passed, 19 failed, 0 skipped; 419 s on CPU).

Blocking failures:
- **(14)** `truncate(H)` is not unit-aligned when one token spans the boundary between unit H and
  H+1. The sampled set then becomes invalid and a horizon sweep raises.
- **(11)** Word and sentence units crash mid-sampling with `AlignmentError` whenever a byte-level
  tokenizer (GPT-2) emits a multi-byte character split across tokens. Reproduced on GPT-2 with a
  Japanese context.
- **(20)** The shipped `min` statistic never returns a standard error, against R5.
- **(24)** `Measure.compute` re-generates S[t+1] when used position by position, instead of sampling
  it once at H_max as §6.1 specifies: 384 vs 288 backend calls on a 4-token sentence.

Non-blocking failures:
- **(35)** The §5 API examples do not run as written.
- **(9b)** Components written to the PDF's own protocol signatures cannot be registered.
- **(25)** Usability: the "level" axis scores 2/5.

## 2. Requirements matrix

| ID | Status | Evidence (one line) |
|---|---|---|
| 1 | PASS | `RepresentationSet.source is A`; a registered non-factoring statistic reading only `A.source` returns the hand-computed value exactly. |
| 2 | PASS | 60 hypothesis cases (V ∈ 2..8, one-hot cosine and discrete): IV within 4·SE of 1 − p[o]; anchor at H=1 is a deterministic singleton. |
| 3 | PASS | Spy statistic: A == `sample(c,H)`, B rows = observed + `sample(c+o,H−1)`; mean within 4·SE of exact enumeration for H∈{1,2,3} × 3 targets × 2 distances. |
| 4 | PASS | 250 + 100 generated set pairs: energy(A,A)=0, ≥0, symmetric, equals 2·M̄_AB − M̄_AA − M̄_BB to 1e-12. Same-p sets: energy/SE = 1.20, 0.55, 0.68 at n = 128, 512, 1024; CI lower bound 0 every time. |
| 5 | PASS | Same-p mean pairwise distance 0.764 / 0.756 / 0.752 vs analytic 1 − Σp² = 0.7542; CI low 0.741 / 0.748 / 0.743 > 0. |
| 6 | PASS | layer:0 vs layer:2 vectors differ; spec hashes differ; `level=` alias works; toy layer encodes one-hot exactly. |
| 7 | PASS | 2- and 3-unit targets within 4·SE of exact; span > H raises `ConfigError` with 0 backend calls. |
| 8 | PASS | 200 generated cases: mean(A,{u}) = mean_i d(a_i,u) and mean({a},{b}) = d(a,b) to 1e-12; all distances have signature `(self, a, b)`. |
| 9 | PASS | Test-only module registers distance/statistic/encoder/warping through `uncertain.register_*` only; all run by name. |
| 9b | FAIL | All four PDF-shaped components are rejected: `must declare a stable descriptor`. |
| 10 | PASS | `Sweep(unit=[…])`, a mixed-unit cell list, and an unknown unit all raise `ConfigError` mentioning "unit". |
| 11 | FAIL | Boundary stopping correct on toy (e.g. word H=3: 107 horizon / 92 EOS / 1 budget). Multi-byte split → `AlignmentError` (toy byte decoder and GPT-2 sentence #19). |
| 12 | PASS | Every EOS/budget stop has `effective_horizon` < H; `Result.diagnostics["effective_horizons"]` matches the set. |
| 13 | PASS | Omitting `short_samples` raises `TypeError`; bad value raises `ConfigError`; three policies give three spec and run hashes. |
| 14 | FAIL | `Sequence.truncate(1)` on tokens ("ab","d e") returns units ('abd','e'). Without spanning tokens: truncate == direct sample (40 cases); H-sweep {1,2,4} uses 851 backend calls, same as H=4 alone. |
| 15 | PASS | `from_config` without `n` raises; `n_requested == n_a_used == n` for n ∈ {2,17,100}. |
| 16 | PASS | `extend(S_n1, n2)` makes exactly n2 backend calls, one sample request (n2, start=n1), and equals `sample(n1+n2)`. |
| 17 | PASS | Every filter call sees `top_k=0`; V=60: tail beyond top 50 sampled at 0.0652 vs analytic mass 0.0682 (within 4·SE); with top_k=50, 0.0. |
| 18 | PASS | Fake batched model (B≠V≠T): max error 0.0; Sampler vs teacher-forced 2.7e-8 (tiny GPT-2), 4.6e-5 (GPT-2); StudySampler 1.6e-8; only `reshape` in src is the same-shape buffer round trip. |
| 19 | PASS | Default store writes 0 files; path store writes only `alternatives/`, `requests/`; Results equal with, without and after reload. |
| 20 | FAIL | `min` returns `se=None` for identity/neglog × H∈{1,2}; mean and energy return finite SE. |
| 21 | PASS | Toy: Spearman ρ = 0.982 (12 items, exact vs MC surprisal); GPT-2: ρ = 0.994 (20 items, mean \|Δ\| 0.085 nats); mean \|closed − MC\| 0.0372 → 0.0265 → 0.0115 → 0.0048 for n = 32 → 2048. |
| 21b | PASS | log SE vs log n slope = −0.487 (target −0.5). |
| 22 | PASS | Same seed ⇒ `==` Results; 3 processes (different `PYTHONHASHSEED`, persistence on/on/off) give byte-identical artifacts and equal rows; all 19 spec fields change the hash. |
| 23 | AMBIGUOUS | `block_size` 1/7/64: identical value and SE; batched StudySampler log-probs = unbatched teacher-forced (1.6e-8); StudySampler draws differ between batch_size 1 and 4 (batch size is in the stream key). |
| 24 | FAIL | Sweep: 5 sampling passes for 5 contexts (288 calls); grid of 8 cells costs the same as 1 cell (216 = 216); d(A,A) reused (768 → 512 distance calls). Sequential `compute`: 8 passes / 384 calls. |
| 25 | FAIL | LLM rubric: unit 3, horizon 3, level 2, distance 3, aggregation 3, set size 5, short-sample policy 4 (mean 3.3); threshold "all ≥ 3". |
| 26 | PASS | Presets vs analytic (z): surprisal 0.74, contextual prob. 0.73, next-unit entropy 2.05, sequence entropy H=2 0.66, IV 0.73, anticipatory IV 1.56. |
| 27 | PASS | Truncation bound by H = 1,2,4,8: 0.715, 0.516, 0.262, 0.059; `warping_bias` = CV²/2n within the 1e-9 tolerance in 6 cases (observed differences at float precision); `corrected_value` = value − bias. |
| 28 | PASS | `prefix_consistent=False`: `truncate` raises; Sweep over H∈{1,2} makes 3 separate sampling passes. |
| 29 | PASS | Type mismatch, energy with non-negative-type distance, unsupported unit: `ConfigError` with 0 backend calls. |
| 30 | PASS | `arrays()` shapes (n, T_max); provenance has model+revision, n, seed, generation (top_k=0); set has context, H, unit + boundary detector. |
| 31 | PASS | All Result fields present on 12 statistic × warping × H configs. |
| 32 | PASS | Runs with `datum=None`; all four transforms flagged (`transform_id` ≠ identity, `original_id` set). |
| 33 | PASS | rows = items × cells; schema present; `datum` is nullable `Float64`; cells of an item share artifact ids, with and without data. |
| 34 | PASS | CSV, Parquet and JSON plus manifest (`specifications`, `engine`, `git_sha`, `seeds`, `environment`); `from_dict(to_dict(s)) == s`. |
| 35 | FAIL | 4 of 5 §5 snippets fail: `HFSampler(…, n=)`, `Measure(sampler=…, unit=…)`, `replace(distance=MyDistance())`, `run()` → `SweepResult`, not a DataFrame. |
| 36 | PASS | 300 generated triples: metric axioms for discrete and ngram_distance; cosine identity/symmetry; mean, energy and min permutation-invariant (150 cases). |
| 37 | PASS | Six shipped components resolve; POS, tree-edit and entailment plugins live in `examples/` and import. |
| 38 | AMBIGUOUS | All four pooling values change vectors and hash; but pooling is unit→span, not token→unit. |
| 39 | NOT IMPLEMENTED | Golden test passes, but `gpt2-values.json` says "NOT recovered study rerun numbers". |
| 40 | AMBIGUOUS | Domain violations rejected before sampling; −log over a [0,1] range is accepted (needed by the surprisal preset). |

## 3. Failures

### F1 — truncation is not unit-aligned when a token spans two units (req 14)
- **Spec (§4.2):** "Truncation is unit-aligned. truncate(H) cuts at unit boundaries."
- **Code:** [src/uncertain/core/values.py:60-62](../src/uncertain/core/values.py#L60-L62). When unit H's span already
  ends at the last token, `Sequence.truncate` returns `self` unchanged, keeping units H+1…,
  because that same token also owns unit H+1. `AlternativeSet.__post_init__` then raises
  `effective horizon exceeds requested horizon` ([values.py:86-87](../src/uncertain/core/values.py#L86-L87)).
- **Minimal failing test:** `eval/tests/test_c_units.py::test_truncate_when_last_token_spans_units_minimal`.
  `Sequence(tokens=(0,3), units=("abd","e"), unit_spans=((0,2),(1,2)), …).truncate(1).units == ('abd','e')`.
  The hypothesis test `test_truncate_equals_direct_sample` found it from sampled data (word unit, H_max=2).
- **Fix:** only take the early return when nothing would be dropped:
  `if end == len(self.tokens) and horizon == len(self.units): return self`. Otherwise build the
  truncated record (tokens[:end], units[:H], spans[:H], char_spans[:H]).

### F2 — word/sentence sampling crashes on split multi-byte characters (req 11)
- **Spec (§4.2, §6.3):** for word/sentence units the sampler generates until H boundaries, EOS or
  budget, "token-ragged but unit-aligned". "Alignment is owned by the library".
- **Code:** [src/uncertain/alignment/__init__.py:53-57](../src/uncertain/alignment/__init__.py#L53-L57) raises when a
  decoded prefix is revised (a partial UTF-8 sequence decodes to U+FFFD, then to the full character).
  It is called per generated token from [samplers/__init__.py:82](../src/uncertain/samplers/__init__.py#L82), so
  one such sample aborts the whole `compute`. The README documents the rejection ("decoders that
  revise previous characters are rejected"), but for GPT-2 it happens mid-run on ordinary
  continuations, not at configuration time.
- **Minimal failing test:** `test_c_units.py::test_edge_multibyte_char_split_across_tokens[word]`
  (byte-level toy: "東" = two tokens). Also `test_integration_gpt2.py::test_pipeline_runs_on_gpt2[19-sentence]`.
- **Fix:** align on byte offsets (or decode with `errors="strict"` and treat an incomplete trailing
  sequence as "no boundary yet"). Assign a unit only once its characters are complete, instead of
  requiring every decoded prefix to be stable.

### F3 — `min` statistic has no standard error (req 20)
- **Spec (R5, §4.7, §10):** "returning every score with its standard error and the n that produced
  it"; "Uncertainty reporting: Mandatory, never a bare float".
- **Code:** [src/uncertain/statistics/__init__.py:216](../src/uncertain/statistics/__init__.py#L216) returns
  `StatisticEstimate(value, None, None, None, "uncertainty_unavailable_min")`. This is a deliberate,
  documented choice (README "Directed minima … return `se=None`").
- **Minimal failing test:** `test_e_estimates.py::test_every_score_has_se_and_n[min-identity-1]`.
- **Fix:** give `min` the same whole-alternative bootstrap the other statistics use: resample A's
  rows and B's rows, recompute the directed minimum, and take the SD. Mark it approximate in
  `status` rather than omitting it.

### F4 — position-by-position `compute` regenerates S[t+1] (req 24)
- **Spec (§1.3, §6.1 step 2):** "One sample set per position therefore serves the whole sentence";
  `S[t], S[t+1] = store.get_or_sample(key, [t, t+1], H_max)`.
- **Code:** [src/uncertain/measures/__init__.py:254](../src/uncertain/measures/__init__.py#L254) samples the
  continuation at `remaining = H − k`, not H_max. The store serves a request only from an artifact
  with horizon ≥ the request ([store/__init__.py:143](../src/uncertain/store/__init__.py#L143)). Position t+1's
  H-horizon request therefore cannot reuse position t's (H−1) continuation and samples again. `Sweep`
  avoids this by pre-planning max horizons per context.
- **Minimal failing test:** `test_f_repro.py::test_sequential_compute_over_positions_reuses_samples`
  (4-token sentence, H=2: 8 sampling passes / 384 backend calls vs 5 / 288 via `Sweep`). Values are
  identical either way, because per-draw seeding makes the H−1 draws prefixes of the H draws. Only
  compute is wasted.
- **Fix:** in `compute`, call `self._get(context, self.spec.horizon).truncate(remaining)` when the
  sampler is `prefix_consistent`, as §6.1 does.

### F5 — §5 API examples do not run (req 35)
- **Spec (§5):** `HFSampler("gpt2", n=32, horizon=5, top_k=0, seed=0)`;
  `Measure(sampler=…, unit=…, representation=…, …)`; `m.replace(distance=MyDistance())`;
  `df = grid.run(…)  # tidy`.
- **Code:** `HFSampler = Sampler` takes only a backend ([samplers/__init__.py:130](../src/uncertain/samplers/__init__.py#L130));
  `Measure.__init__(spec, *, sampler, …)` ([measures/__init__.py:58](../src/uncertain/measures/__init__.py#L58)); `replace`
  routes `distance` into the spec, so an object is looked up as a registry name
  (`Unknown distance: <…MyDistance object…>`); `Sweep.run` returns `SweepResult`
  ([experiments/__init__.py:125-126](../src/uncertain/experiments/__init__.py#L125-L126)). `docs/api.md:100` calls the
  constructor change "a deliberate deviation".
- **Minimal failing tests:** `eval/tests/test_g_api.py::test_spec_hfsampler_signature`,
  `::test_spec_measure_keyword_constructor`, `::test_spec_custom_distance_injection`, `::test_spec_sweep_returns_dataframe`.
- **Fix:** add thin adapters: `HFSampler(model_id, *, n, horizon, top_k=0, seed, revision)`, a
  keyword `Measure(sampler=…, **axes)` path delegating to `from_config`, object injection in
  `Measure.replace` (it already exists on `Measure(...)`), and `SweepResult.__dataframe__` or `run(...,
  as_frame=True)`. Or amend the spec to the implemented API.

### F6 — PDF-shaped plugin components are rejected (req 9b)
- **Spec (§7 table, §4.5–4.7):** distance declares `input_type, range`, implements `__call__(a,b)`;
  statistic `range`, `__call__(A,B)`; encoder `output_type`, `encode(set)`; warping `domain`,
  `__call__(x)`, `gap`.
- **Code:** the registry requires a `descriptor` ([registry.py:27-28](../src/uncertain/registry.py#L27-L28)). In a
  probe that added only a descriptor, the distance then worked, but the statistic failed
  (`statistic must supply an uncertainty contract`, [measures/__init__.py:96](../src/uncertain/measures/__init__.py#L96);
  the library protocol is `estimate(A,B,distance,estimator,seeds)` + `range_for`). The encoder
  failed (`supported_units` missing; library calls `encode(source, short_samples)`), and so did
  the warping (library calls `apply(estimate)`).
- **Minimal failing test:** `test_b_distance.py::test_spec_protocol_component_registers_and_runs[statistic]`.
- **Fix:** derive a default descriptor from the registered name and version, and wrap
  spec-protocol objects in adapters. For example, lift a `__call__(A,B)` statistic with a
  bootstrap SE, call `encode(s)` when it takes one argument, and lift `__call__`/`gap` into
  `apply`. Or update §7 to the implemented protocol.

### F7 — usability: representation names not discoverable (req 25)
- **Spec (§5, §1.4 "Small surface"):** "a reader who knows the framework can read a Measure and know
  exactly which measure it is".
- **Evidence:** `eval/usability_grading.md`. `Measure`, `compute`, `replace`, `Sweep` and the
  presets have no docstrings. The `from_config` docstring names no encoder, distance or statistic.
  `Unknown statistic: median@1` and `Unknown unit: morpheme` don't list the valid options.
- **Fix:** list the registered names in the `from_config` docstring and in "Unknown X" errors (the
  registry already holds them), and state that H counts units.

## 4. Ambiguities in the spec

| Topic | What is unclear | Interpretation tested |
|---|---|---|
| Distance arity (prompt item 8) | The prompt asks for three distance arities; §4.5 says one pointwise signature, no arity flag. | PDF: singleton-set statistics must agree with the pointwise distance (req 8). |
| Reshape regression (18) | §4.3 says "named rearrangement … checked at construction by a shape assertion and an alignment assertion". The library has no axis swap at all. | No reshape-based reordering in `src/`; per-row alignment through the only batched logits path; log-probs = teacher-forced (§9 oracle 2). |
| top_k spy (17) | The prompt says spy on `generate()`; the library never calls it. | Spy on `sampling_probabilities`; empirical tail test. |
| Batched vs unbatched (23) | Not in the PDF. | block_size invariance and batched-vs-unbatched log-probs pass. StudySampler draws change with batch size, which is documented, since batch size is in the stream key. |
| Pooling (38) | §6.3 pools tokenizer offsets into unit spans; the library always mean-pools tokens→unit and offers mean/first/last/sum across units. | Pooling choices are hashed and change the vectors. |
| −log domain (40) | §4.5 says −log over a statistic range containing 0 "fails in milliseconds"; the §5 surprisal preset is exactly that. | Real domain violations are rejected; −log over [0,1] is accepted with an explicit infinite zero-hit policy. |
| §5 sweep grid | Its literal grid (`layer:*` × `ngram_overlap`) is type-invalid under §4.5. | The library rejects the grid at construction (`test_spec_literal_grid_is_type_consistent`, passes); valid cells tested separately. |
| Short-sample "documented direction" (§9) | No direction is documented in the PDF or the repo. | The three policies give three different values and `drop` lowers n; direction not graded. |
| Energy "vanishes exactly" (§4.6) | The default V-statistic has positive bias ≈ (1−Σp²)(1/n+1/m) at P = Q. | Graded on CI ∋ 0 and \|value\| ≤ 4·SE (energy/SE = 1.20, 0.55, 0.68). An unbiased U variant exists (`variant="u"`). |
| Dependencies (§6.3) | The PDF lists torch, transformers, pandas and einops as core; the library's core is numpy-only (`docs/design-decisions.md:47-49`, "deliberate departure"). | Not graded. |

## 5. Eval-suite gaps

**Mutation check (`eval/mutations.json`): 9 / 9 caught.** That covers the four required mutations
plus five extras.

| Mutation | Caught by |
|---|---|
| reshape in StudySampler logits path | 4 / 4 selected tests (fake batched model, teacher-forced) |
| reshape in `AlternativeSet.arrays` | 4 / 4 |
| default `top_k=50` | 2 / 3 (spy + tail frequency) |
| energy without d(A,A), d(B,B) | 7 / 9 |
| `extend` regenerates everything | 3 / 3. Only the call-count spy can see it, since values are identical. |
| truncate by tokens, not units | 1 / 2 |
| short_samples not hashed | 2 / 3 |
| SE = sd/n | 6 / 8 |
| continuation sampled from the wrong context | 18 / 27 |

Known gaps and limits:
- The truncation mutation was first **not caught**: the only truncate-vs-sample test already fails
  on the real library (F1), which masks the mutant. A variant without spanning tokens now catches
  it. Other requirements that currently FAIL (11, 20, 24) have the same masking risk for future
  mutations.
- Criterion 25 was graded by the same model that wrote the suite, so it is not an independent
  judgment.
- GPT-2 integration uses n ≤ 8 and token budget ≤ 20 to stay within budget. Sentence-unit runs
  therefore often stop on the budget (effective horizon 0), and word/sentence IV values are mostly
  1.0 with SE 0. The run is a smoke test, not a statistical check. Oracle 1 on GPT-2 uses n = 1024
  with a memoised next-token distribution.
- Energy P = Q is tested at n ≤ 1024, not 2048: the pairwise distance loop is O(n²) in Python.
- Not tested: §9 "truncate(sample(H_max), H) matches sample(H) in distribution" for samplers whose
  per-draw seeding is not prefix-stable (the library makes it hold exactly); YAML config loading;
  the CLI; the `emnlp2024` study estimators beyond StudySampler log-prob alignment; GPU execution.
- Robustness note (not a spec failure): the process-global d(A,A) memo
  ([statistics/__init__.py:82-99](../src/uncertain/statistics/__init__.py#L82-L99)) keys on (source hash, encoder
  descriptor), not on the representations. Two different encodings sharing a descriptor would
  silently reuse a stale block. My first property test hit this and got a negative energy of −0.27.

## 6. How to rerun

```sh
# from the repo root, Windows Git Bash / PowerShell equivalent
.venv/Scripts/python.exe eval/grade.py      # full suite (~7 min CPU) -> eval/results.json
.venv/Scripts/python.exe eval/mutate.py     # mutation check            -> eval/mutations.json
.venv/Scripts/python.exe eval/usability_evidence.py   # regenerate evidence for criterion 25
# just the tests:
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest eval/tests -q
```

GPT-2 tests need `openai-community/gpt2` @ `607a30d783dfa663caf39e06633721c8d4cfcd7e` in the local
Hugging Face cache; without it they skip and are reported as NOT RUN.

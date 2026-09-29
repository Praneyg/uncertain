# Conformance audit: `uncertain` vs. the π-infoval library design proposal

Audited 2026-09-21 against `π-infoval-library (1).pdf` §1–§11 at `0.1.0a2`. Four independent
passes (core values · axes and extensibility · estimation and persistence · oracles and
runtime), reconciled here. Findings marked **[ran]** were verified by executing code;
everything else was verified by reading. No source or test file was modified.

## Status (updated 2026-09-21, after WP1–WP4)

The findings below are **kept verbatim** — the record is the audit's value. This section
says only what has changed since it was written. Suite: **120 passed, 0 failed, 0 skipped**;
`ruff check .` passes.

**Closed.** Gap 1 (horizon sweep on word/sentence units — `sample()` trims the lookahead
token; the sentence detector no longer revises units mid-generation). Gap 2 (energy's
within-set blocks memoised). Gap 3 (R4 — `tests/test_measures.py` now asserts exact ≈ mc at
large n, tolerance from the MC path's own standard error). Gap 4 (`register_unit`,
`register_sampler`; `boundaries()` dispatches through the registry). Gap 5 (string
`unit`/`representation` normalized on `replace`, `"layer:8"` parsed, `level=` alias,
`mode=` in `from_config`; `Measure(sampler=…, unit=…)` deliberately still raises — see
scope below). Gap 6 (sampler records `unterminated`; `_diagnostics` turns it into a real
`truncation_bound` wherever the distance range is finite). Gap 7 (`range_for(distance)` on
every statistic; `validate()` enforces `warping.domain ⊇ statistic.range_for(distance)`
generically). Gap 8 (`lifted()` ships and is tested). Gap 9 (§9 oracle 3 re-pinned on GPT-2
at a fixed model/context/seed in `tests/golden/gpt2-values.json`, per target token and in
order; the toy analytic values are kept alongside). Gap 10 (oracle 1's cross-item
correlation half exists: MC surprisal vs. GPT-2 teacher-forced surprisal, r ≥ 0.99 pinned,
measured 0.9975 at n=512). Gap 11 (the 36-column dataframe schema is pinned in both pairing
modes). Gap 12 (git SHA and a top-level `seeds` field in run manifests). Gap 13 (`pad`
exercised end-to-end against `drop` and `compare_ragged` with the direction asserted;
metric axioms on core's own distances; `effective_horizon` under EOS; energy-vanishes
rewritten onto two independent draws with the mean-pairwise-distance companion). The
transpose negative control named in the closing caveat now feeds the corrupted table back
through the oracle's own assertion inside `pytest.raises`.

**Open, with the reason.**

- *No `einops`, and torch/transformers/pandas stay out of the core install* (gap 14). The
  audit found no axis-permuting code left to ban; adding a dependency to satisfy a prose
  line is backwards. A lint ban on `.reshape` was cheap and shipped instead: ruff's
  `banned-api` covers `np.reshape`, and `tests/test_properties.py` covers bare
  `x.reshape(...)` in `src/` and `examples/`, exempting the one `frombuffer` round-trip.
  Revisit if an encoder reintroduces axis reordering.
- *Six distances rather than three* (gap 15). `discrete` is load-bearing. Removing
  `ngram_distance`/`euclidean` is a breaking change that buys only catalogue purity.
- *Two persisted file classes* (gap 16). §6.2's actual concern — not persisting encodings —
  is respected; the request→artifact index is what makes replay without generation work.
- *Flat `tests/`* (gap 17). Cosmetic.
- *`Result.spec_hash != MeasureSpec.spec_hash`.* Deliberate: the result's hash also covers
  resolved component descriptors, the backend and the engine version, which the spec alone
  does not.
- *No general truncation bound for unbounded distances.* Only the bounded-distance case
  gets a real number; elsewhere the honest `None` / `"no_general_bound"` stands. Guessing a
  bound is worse than not having one.
- *`lifted()` ships as a helper without rewriting `Mean`/`Energy`/`DirectedMin` onto it.*
  They carry bootstrap and CI logic a generic lifter would not capture; rewriting three
  working estimators to satisfy a helper's existence is risk without benefit.
- *Appendix A item 5, the unpaired output schema* (gap 18). Still per-position rows only,
  but no longer decided implicitly: `test_dataframe_schema_is_pinned_in_both_pairing_modes`
  pins the same 36 columns for paired and unpaired sweeps, with `datum` null rather than
  absent when no human data is supplied. Per-document summaries remain out of scope.
- *`mypy` is now reproducible, and green.* `mypy` is in the `dev` extra and installed;
  it reports **no issues across 41 source files** (`src/`, `tests/`, `examples/`). The
  first run reported 16 errors, 11 of them introduced by this work — `MeasureSpec.unit`
  and `.representation` were widened to `… | str` for string normalization and then read
  unguarded in `measures/__init__.py`. Those annotations were narrowed back to the
  concrete spec types (the stored value is never a `str` after `__post_init__`), and the
  five pre-existing numpy `var-annotated` requests were annotated. The "as do Ruff and
  mypy" claim that this audit found half-unsupported is now true and reproducible from
  the declared dependencies.

## Verdict

The library is a faithful implementation of the proposal's *architecture* and a partial
implementation of its *guarantees*. The central claim survives contact with measurement:
a 9-cell × 3-item sweep produced 27 result rows from 4 generation passes **[ran]**, exactly
one per distinct position, with cost flat as the grid doubled. Type checking before model
work is real and strict, the functional core is genuinely pure, the plugin API is open in
the way §7 demanded (all three demanding examples live outside core and none required
editing it), and the teacher-forced log-prob oracle that §9 calls "the one that was
missing" now exists and passes **[ran]**. 91 tests pass, zero skips, exactly as claimed.

What is not delivered is a specific and coherent class of things: the accounting and
amortisation promises. Truncation error is a permanent `None`. Energy's advertised block
reuse does not exist, so the default statistic costs 3× on every cell. The
`warping.domain ⊇ statistic.range` gate — one of the two checks §4.5 names when it calls
failing loudly "an explicit design value" — is a hardcoded name test. R4 is inverted: the
exact and Monte Carlo paths are each pinned to separately maintained analytic constants and
nothing asserts they agree. And the horizon sweep, the economic premise of the whole design,
is unsound for word and sentence units.

The proposal's own §5 code samples do not run as printed. That is worth stating plainly
because §5 is the document's contract with a reader.

## Requirements R1–R11

| | Requirement | Verdict | Evidence |
|---|---|---|---|
| R1 | Measure as a function of two alternative sets under configurable d/s/f, unit or span | CONFORMS | `measures/__init__.py:263-269`, span granularity `:200-203` |
| R2 | H ≥ 1 in the run's unit; H=1 not a separate code path | CONFORMS | `core/specs.py:144-145`; unit-counted stop `samplers/__init__.py:73-79` |
| R3 | Recover published special cases by configuration alone | **PARTIAL** | surprisal/probability/IV ran clean; entropy and anticipatory need `mode`, absent from `from_config`; `_anticipatory` is a separate method `measures/__init__.py:304-332` |
| R4 | Closed-form fast paths validated **against the Monte Carlo path** | **CONTRADICTS** | `tests/test_auto_configuration.py:94-105`, `tests/test_golden.py:7-13` — both pin independent analytic truth; no test asserts `exact ≈ mc` |
| R5 | Every score returns estimate + standard error + n | PARTIAL | `core/results.py:17-41` always carries n; `se=None` on six paths, worst `measures/__init__.py:260-262` |
| R6 | Sampling error and truncation error reported **separately**, plus warping gap | **CONTRADICTS** | `measures/__init__.py:301` hardcodes `truncation_bound: None`; gap half is correct |
| R7 | One unit per run; horizon/granularity/representation/alignment expressed in it | CONFORMS | `core/specs.py:71`, `measures/__init__.py:94`, `experiments/__init__.py:90` |
| R8 | Representational levels pluggable; rep set retains its sequences | CONFORMS | `registry.py:32`, `core/values.py:157` |
| R9 | Set-level statistics, energy distance among them | CONFORMS | `statistics/__init__.py:122`; formula exact at `:153` |
| R10 | Sweep driver over the axis grid, tidy dataframe | CONFORMS | `experiments/__init__.py:103-113` dedupes to one request per stream; `:141-166` one row per (item, cell) |
| R11 | One persisted artifact with full provenance, deterministic seeding | PARTIAL | `store/__init__.py:140-160`; key list complete, but a second request-index file class is also written `:156-159` |

## Architectural commitments (§4, §10)

| Commitment | Verdict | Evidence |
|---|---|---|
| One operand type; anchor at H=1 a singleton; no arity branching | CONFORMS | `core/values.py:70`, `:111`, `:116`; no arity flag in any of the six distances |
| Distances pointwise only, one signature | CONFORMS | `distances/__init__.py` — verified across all six |
| Representation set is the primitive; matrix derived | CONFORMS | `core/values.py:157`; `examples/entailment/plugin.py` reads `a.reps` directly |
| Type checking before any model work | PARTIAL | `measures/__init__.py:92-95` fires in `__init__`; but see domain-check row |
| `warping.domain ⊇ statistic.range` | **MISSING** | no statistic declares `range`; substituted by `measures/__init__.py:101-103` name test |
| `sampler.prefix_consistent or H == H_max` in VALIDATE | PARTIAL | enforced at `core/values.py:102-103`, not in the no-GPU gate |
| Energy = `2·mean(A,B) − mean(A,A) − mean(B,B)` | CONFORMS | `statistics/__init__.py:153` |
| Energy's within-set blocks reused across the sweep | **CONTRADICTS** | `statistics/__init__.py:145` recomputes inline; measured 1200 calls per cell, no amortisation **[ran]** |
| `lifted(distance, reduction)`; factorable stats lifted automatically | **MISSING** | no `lifted` symbol in `src/`; every statistic hand-rolls its reduction |
| `pairwise(A,B,d)` memoised helper | PARTIAL | `statistics/__init__.py:50-55` exists, unmemoised, bypassed by production code |
| Three distances, three statistics | PARTIAL | statistics exact; **six** distances shipped |
| Six registry decorators | **PARTIAL** | `register_sampler` and `register_unit` absent; `registry.py:6-7` has four kinds |
| Plugin examples ship outside core, none touching core | CONFORMS | `examples/{pos,tree_edit,entailment}/plugin.py`; import only public symbols |
| Canonical `(sample, step)` order, reshape banned in lint | PARTIAL | `pyproject.toml:38` selects `E4,E7,E9,F,B`; no reshape ban, no einops |
| Shape **and** alignment assertion at construction | PARTIAL | `core/values.py:30-47` structural only; semantic check lives in `tests/test_hf.py:21` |
| `n` user-specified, always | CONTRADICTS at the entry point | `core/specs.py:34` required; `measures/__init__.py:22` re-defaults `n=512, seed=0` |
| `short_samples` required, hashed, never conventional | CONTRADICTS at the entry point | `core/specs.py:132` required; `measures/__init__.py:22` re-defaults it |
| Frozen dataclasses; spec is cache key + run id + citable spec | PARTIAL | frozen and hashable, but `Result.spec_hash != MeasureSpec.spec_hash` |
| Uncertainty mandatory, never a bare float | PARTIAL | `Result` is always structured; `se` is `None` on six paths |
| No adaptive stopping / importance weighting / variance reduction | CONFORMS | grep clean; bootstrap CIs only |
| Functional core, imperative shell | CONFORMS | no I/O outside `samplers/` and `store/` |
| Store with no path writes nothing, identical results | CONFORMS | every write behind a `path is not None` guard; byte-identical across processes **[ran]** |

## Ranked gaps

### (a) Contradictions — the code does the opposite of a stated commitment

1. **The horizon sweep is unsound for word and sentence units.** `sample()` retains the
   boundary-triggering lookahead token (`samplers/__init__.py:83`, deliberately, per its own
   comment); `truncate()` cuts at the unit span end and drops it (`core/values.py:60`). So
   `truncate(sample(H_max), H) ≠ sample(H)` on any non-token unit. Reproduced directly
   **[ran]** with a spaced vocabulary: units identical, but tokens, per-token log-prob sums
   (−3.98 vs −5.18) and `content_hash` all diverge — the difference is exactly the lookahead
   token's log-prob. This is §4.2's "sample once at H_max, truncate for every smaller
   horizon" failing on precisely the units the unit abstraction was built for, and it is
   silent: unit-level distances still agree, so only log-prob-dependent measures and cache
   identity are affected.

   *Why no test caught it:* `tests/test_sampling.py:19-23` asserts pathwise identity but
   passes no `UnitSpec`, so it runs on token units, where one token is one unit and the two
   paths coincide by construction. The default `ToyBackend` vocabulary `("a","b")` contains
   no whitespace, so even a word-unit test against it would degenerate to a single unit and
   pass vacuously. Two agents initially disagreed here; both were right about what they ran.

2. **Energy's block reuse does not exist.** §4.6 and §10 both justify energy as the default
   on "two extra blocks per comparison, and both are reused… amortised across the sweep."
   Measured **[ran]**: identical repeat calls and cells sharing operand A each recompute
   `d(A,A)` from scratch — 1200 distance evaluations per cell, three times over.
   `RunMemo` memoises encodings only (`measures/__init__.py:219`); `Energy.estimate`
   computes `weighted_mean(s, s, …)` inline at `statistics/__init__.py:145` with no
   content-keyed cache. The default statistic costs 3× on every cell of every sweep.

3. **R4 is inverted.** The fast path and the MC path are each validated against separately
   maintained analytic constants — exactly the "two independently maintained
   implementations" R4 exists to prevent. No test asserts the two agree at large n.

4. **`register_unit` absent means adding a unit requires editing core.** Units are hardcoded
   in two places (`core/specs.py:77-80` and `measures/__init__.py:39-42`). §7's "Nothing in
   the core is edited, and the core never learns that the new component exists" is false for
   two of its own six rows. §7 itself sets the test: "If a component of that kind cannot be
   added without touching core, the plugin API is only nominally open." For distances,
   statistics, encoders and warpings it is genuinely open; for units and samplers it is not.

5. **§5's published examples do not run.** `Measure(sampler=…, unit=…)` raises `TypeError`
   (the constructor is `Measure.from_config`); `level=` is not a kwarg anywhere;
   `representation="layer:8"` raises `ConfigError` (no `name:arg` parsing, so §5's own
   `Sweep(level=["layer:0","layer:8"])` grid is unwritable); and
   `replace(unit="token")` / `replace(representation="tokens")` raise `AttributeError`
   because `MeasureSpec.__post_init__` normalizes a string `statistic` (`core/specs.py:152`)
   but not a string `unit` or `representation`, while `from_config` normalizes all three.
   String axis values work on the way in and break on `replace`.

### (b) Required by the spec, simply absent

6. **Truncation error.** `measures/__init__.py:301` emits `truncation_bound: None,
   truncation_status: "no_general_bound"` on every result, and the quantity §4.7 defines the
   bound from — unterminated probability mass — is never recorded by the sampler, which
   tracks only `stop_reason`. Half of a requirement the spec headlines ("reported separately
   because they shrink under different remedies") is a stub. The honest label is the only
   thing keeping it from being silent. The other half of R6 is correct: `NegLog` computes
   `se²/(2x²)`, algebraically identical to the specified CV²/2n, with optional correction off
   by default.

7. **The `warping.domain ⊇ statistic.range` gate.** No statistic declares `range`. The check
   is `warping == "neglog" and statistic.name in {"mean","energy"}`. A user-registered
   warping over a user-registered statistic gets no domain check at all —
   `tests/test_auto_configuration.py:116` exercises that unchecked path today. §4.5 names
   this check specifically, right after "Given that both known bugs in the prior code were
   silent, failing loudly on anything checkable is an explicit design value."

8. **`lifted()`.** Absent entirely; factorable statistics are lifted by hand, so adding one
   means writing the blocked reduction rather than calling a helper.

9. **Oracle 3 is not the oracle §9 specifies.** `tests/golden/toy-values.json` pins analytic
   constants of a two-symbol toy distribution — no model, no seed — and its own `source`
   field says "not study rerun values". §9 wants corrected re-run numbers on a fixed
   (model, context, seed) *precisely so "the bug that motivated the library cannot silently
   recur"*; analytic toy constants cannot catch a regression in the HF path, which is where
   the bug lived. `docs/validation-results.json` concedes the cause: the original code and
   corrected fixtures are unavailable.

10. **Oracle 1 is missing its cross-item correlation half.** The convergence check exists and
    is well built (`tests/test_measures.py:31-41`, halving over n ∈ {64, 2048}), but it runs
    on *probability* against a ToyBackend constant, not on surprisal against real LM
    surprisal across items. No correlation test exists in any form, so one of the two
    thresholds §9 requires pinned cannot be.

11. **No dataframe schema assertion.** §1.4 and §10 both say "the dataframe schema is the
    contract"; §9 requires the e2e sweep to assert it. Columns are derived at runtime from
    `list(self.rows[0])` (`experiments/__init__.py:50`) and pinned nowhere. A dropped or
    renamed column breaks every downstream consumer silently.

12. **Git SHA missing from run manifests** (`experiments/__init__.py:169-183`). Five of six
    §6.3 items present. For a library whose stated purpose is third-party re-analysis of
    published results, the provenance chain stops at the library version.

13. **Smaller test holes**, each named explicitly in §9: `pad` is never exercised
    end-to-end (only as a hash input and a rejection argument) though §4.2 insists the policy
    changes both log-prob sums and distances; metric axioms are tested only on the *plugin*
    `TreeEdit`, never on core's own distances; `eff_horizon` is asserted under token budget
    but not under EOS; energy-vanishes is tested as `estimate(a, a)` — the same set against
    itself, zero by construction — rather than two independent draws, and its companion half
    (mean pairwise bounded away from zero by the entropy) is absent, which is the entire
    stated reason energy is the default.

### (c) Scoping questions, not bugs

14. **Core dependencies are `numpy` only** (`pyproject.toml:11`); torch, transformers and
    pandas are extras and einops is absent from the project. This contradicts §6.3's named
    core list while satisfying the same paragraph's "nothing heavyweight in the core
    install" — the spec paragraph is internally inconsistent and the code picked the better
    half. Consequence worth knowing: a reader following §6.3 will `pip install` and hit
    `ImportError` in `Sweep.to_dataframe`. No `[nli]` extra exists.
15. **Six distances rather than three.** `discrete` is load-bearing — it is the `from_config`
    default and the only negative-type unit distance, so energy-over-tokens requires it, and
    the spec's own three-distance list cannot support its own default statistic.
    `ngram_distance` and `euclidean` are unforced surplus, each now a permanent test-suite
    and sweep-grid commitment per §10.
16. **Two persisted file classes**, not one: the alternatives plus a small request→artifact
    index (`store/__init__.py:156-159`). §6.2's actual concern — not persisting encodings —
    is respected.
17. **Flat `tests/`** instead of `{unit,oracles,golden,integration}`, and no `scripts/`
    (runners sit at repo root). Structural only — though it is why "does oracle 2 exist?"
    required reading three files rather than looking in one.
18. **Appendix A item 5 (unpaired output schema) was decided implicitly**: per-position rows
    only (`data/__init__.py:76-83`), no per-document summaries. The spec flags this as an
    open decision, so this is a choice made without being recorded as made.

## Claims in IMPLEMENTATION-STATUS.md that runtime verification did not support

- **"91 tests pass" — verified exactly** **[ran]**: 91 collected, 91 passed, 0 failed,
  0 skipped, 54.2s. Every optional-dependency test genuinely ran (torch, transformers,
  hypothesis, zss, pandas, pyarrow all installed).
- **"as do Ruff and mypy" — half unsupported** **[ran]**: ruff passes ("All checks
  passed!"); **mypy is not installed and appears in no extra** (`pyproject.toml:16` dev =
  pytest, hypothesis, ruff, build). `[tool.mypy]` config exists at `:40-43`, but nobody
  following the declared dependencies can reproduce the claim.
- **"Importing the package performs no downloads" — verified** **[ran]**: with fresh empty
  `HF_HOME`/`HUGGINGFACE_HUB_CACHE`/`TRANSFORMERS_CACHE` and `socket.connect` patched to
  raise, `import uncertain` completed in 243 ms with `torch` and `transformers` absent from
  `sys.modules`.
- **Sampling defaults — verified at the point of use** **[ran]**: realised generation dict is
  `{do_sample: True, temperature: 1.0, top_k: 0, top_p: 1.0, num_beams: 1}`, the filter is
  provably identity, and first-token EOS rate measured 0.49 on a 50/50 backend, i.e. no
  first-step suppression. There is no `GenerationConfig` to override it — `Sampler.sample`
  runs its own ancestral loop and never calls `model.generate`.
- **The §9 "written first, confirmed to fail before the refactor" process claim is
  unverifiable**: the working directory is not a git repository.

One caveat on the headline oracle: `tests/test_hf.py:26-29` builds the historically-buggy
transposed array and asserts only that it *differs from* the correct table — it never feeds
the corrupted array back through the oracle's own assertion. That proves the fixture is
discriminative, not that the comparison rejects the bug. One line inside `pytest.raises`
would close it, and given the transpose bug is the library's founding motivation, it is the
cheapest high-value fix on this list.

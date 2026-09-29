# Implementation status

The library is a working research alpha, version **0.1.0a2**.

## Current validation (2026-09-21, after WP1–WP4)

**120 tests pass, 0 failures, 0 skips.** `ruff check .` passes. `mypy` is now in the
`dev` extra and installed, so the type claim is reproducible — and it now holds:
**`mypy` reports no issues across 41 source files**, `src/`, `tests/` and `examples/`
alike. Reproducing it first surfaced 16 errors, 11 of which this work introduced:
`MeasureSpec.unit`/`representation` had been widened to `UnitSpec | str` /
`EncoderSpec | str` so string axis values could normalize in `__post_init__`, after which
`measures/__init__.py` read `.name` off them unguarded. The stored value is never a `str`
after construction, so the annotations were narrowed to the concrete spec types and the
string-accepting normalization documented at the field; `EncoderSpec.parse` no longer
rebinds its `str` parameter to `int`. The remaining five were pre-existing `var-annotated`
requests on numpy results and now carry explicit annotations. Earlier releases claimed
"mypy passes" without mypy installed anywhere — see `AUDIT-CONFORMANCE.md`.

## What the four work packages closed

- **WP1 — the horizon sweep on word and sentence units.** `sample()` trims the
  boundary-triggering lookahead token, so the artifact holds exactly H units and
  `truncate(sample(H_max), H) == sample(H)` now holds for word and sentence units, not only
  for tokens. The sentence detector no longer closes a sentence on a terminator at
  end-of-text in non-final mode. The sampler records an `unterminated` count in
  `AlternativeSet` provenance. **All stored `content_hash` values changed**; alternatives
  cached before this change are invalidated.
- **WP2 — statistics.** Energy's within-set blocks are memoised behind a bounded
  process-global LRU (1200 → 400 distance calls on a repeat). All three statistics answer
  `range_for(distance)`. `lifted(distance, reduction)` ships as a tested helper for new
  factorable statistics.
- **WP3 — API surface, registries, gates, manifest.** `register_unit` and
  `register_sampler` exist and `boundaries()` dispatches through the registry, so a fourth
  unit can be added from a test file with nothing under `src/` edited. `MeasureSpec`
  normalizes string `unit`/`representation`, parses `"layer:8"`, accepts `level=` as an
  alias and `mode=` in `from_config`. `Measure.validate()` enforces
  `warping.domain ⊇ statistic.range_for(distance)` generically instead of by name test.
  `_diagnostics` turns the `unterminated` count into a real `truncation_bound` wherever the
  distance range is finite. Run manifests carry a git SHA and a top-level `seeds` field.
- **WP4 — oracles and test gaps.** The transpose negative control now feeds the corrupted
  table back through the oracle's own assertion inside `pytest.raises`. The closed-form
  path is asserted against the Monte Carlo path at large n (R4), with the tolerance taken
  from the MC path's reported standard error. §9 oracle 1's missing half exists:
  Monte Carlo surprisal correlates with GPT-2 teacher-forced surprisal at r ≥ 0.99 across
  ten items. §9 oracle 3 is pinned on a fixed (model, context, seed) on GPT-2 in
  `tests/golden/gpt2-values.json`, per target token and in order, so a transposed table
  cannot reproduce it. The sweep dataframe's 36 columns are pinned in both pairing modes.
  `pad`, metric axioms for core's own distances, `effective_horizon` under EOS and the
  energy-versus-mean-distance contrast on two independent draws are all exercised.

## Open by decision, not by omission

Recorded with reasons in `AUDIT-CONFORMANCE.md` § Status: no `einops` and no core
torch/transformers/pandas; six distances rather than three; two persisted file classes;
flat `tests/` layout; `Result.spec_hash != MeasureSpec.spec_hash`; no general truncation
bound for unbounded distances; `lifted()` ships as a helper without rewriting the three
existing statistics.

## Scope not claimed

Full scientific v1 remains gated on full-dataset sampling and downstream replication, exact
word/sentence event likelihoods, and defensible uncertainty for directed minima and
entailment clustering. Those paths are explicit errors or unavailable statuses. GPU
throughput and a broad platform/version matrix are not validated. The historical study
re-run is **not** reproduced: `docs/validation-results.json` records the original code and
corrected fixtures as unavailable, which is why the GPT-2 golden values are pinned from the
corrected implementation and labelled as such rather than presented as study numbers.

The September 9 study extension adds exact-score comparison across all 1,487 notebook
observations, audited study-specific cosine estimators, a batched GPT-2 sampler, and a
completed pilot at N=4–512 and H=1/5/10/15. See `replication/README.md` and
`docs/replication-plan.md`.

Start with [README](README.md), see [API contracts](docs/api.md), and consult the
[validation report](docs/validation-results.json) for the recorded environments.

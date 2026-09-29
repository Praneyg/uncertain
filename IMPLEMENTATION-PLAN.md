# Implementation plan — closing the AUDIT-CONFORMANCE.md gaps

Source: [AUDIT-CONFORMANCE.md](AUDIT-CONFORMANCE.md). Ordered by the audit's own ranking.
Four work packages; WP1–WP3 own disjoint files and run concurrently, WP4 runs after them.

## Scope decisions made up front

**In scope:** every contradiction in bucket (a), the two missing gates in bucket (b) that
are mechanical, the accounting ingredient for truncation error, the registry holes, and the
§9 test gaps.

**Deliberately out of scope**, with the condition for revisiting:

- *Adding `einops` as a core dependency.* The audit found no axis-permuting code left to
  ban; adding a dependency to satisfy a prose line is backwards. Revisit if an encoder
  reintroduces axis reordering. A lint ban on `.reshape` is cheap and is in scope instead.
- *Rewriting `Mean`/`Energy`/`DirectedMin` onto `lifted()`.* They carry bootstrap and CI
  logic a generic lifter would not capture; rewriting three working estimators to satisfy a
  helper's existence is risk without benefit. `lifted()` ships as a real, tested helper for
  *new* factorable statistics, which is what §4.6's promise is actually for.
- *Removing the three surplus distances.* `discrete` is load-bearing. Removing
  `ngram_distance`/`euclidean` is a breaking change that buys only catalogue purity.
- *Inventing a general truncation bound.* Only the bounded-distance case gets a real
  number (WP3.4). Guessing a bound is worse than the current honest `None`.
- *Restructuring `tests/` into `{unit,oracles,golden,integration}`.* Cosmetic.

## WP1 — The horizon sweep on word/sentence units  *(audit gap 1, the worst one)*

Files: `src/uncertain/samplers/__init__.py`, `src/uncertain/core/values.py`,
`tests/test_sampling.py`.

`sample()` retains the boundary-triggering lookahead token (`samplers/__init__.py:83`);
`truncate()` cuts at the unit span end (`core/values.py:60`) and drops it. The two paths
must agree.

**Fix direction: trim in `sample()`, do not reconstruct in `truncate()`.** When
`len(units) >= horizon`, slice `tokens` / `model_logprobs` / `sampling_logprobs` /
`observed` / `text` to the same unit-span end `truncate()` uses, so the artifact holds
exactly H units and the two paths coincide by construction. Smaller diff than teaching
`truncate()` to reproduce the sampler's stop rule, and it matches §4.2's "truncation is
unit-aligned" framing. Leave short samples (`len(units) < horizon`, EOS or budget) alone —
they keep every token they generated.

Consequence to state in the commit: stored `content_hash` values change, so alternatives
cached before this change are invalidated. Acceptable at `0.1.0a2`.

**Also in WP1** (the sampler half of audit gap 6): record the ingredient §4.7 says the
sampler records — an `unterminated` count in `AlternativeSet` provenance, the number of
sequences with `effective_horizon < horizon`. Recording only; WP3 consumes it.

Test: a `ToyBackend` with a **whitespace-bearing vocabulary** (e.g. `("x", " y", " z")`)
and `UnitSpec("word", "whitespace:v1")`, asserting
`long.truncate(2).sequences == short.sequences`. The existing `test_prefix_pathwise_identity`
passes no `UnitSpec`, so it runs on token units where the paths coincide anyway, and the
default `("a","b")` vocabulary has no whitespace, so a word test against it degenerates to
one unit and passes vacuously. Cover both units.

## WP2 — Statistics: block reuse, range, lifted()  *(audit gaps 2, 7-declaration, 8)*

Files: `src/uncertain/statistics/__init__.py`, `tests/test_extensions_and_statistics.py`.

1. **Memoise the within-set blocks** (gap 2). `Energy.estimate` recomputes
   `weighted_mean(s, s, …)` inline at `statistics/__init__.py:145` on every call. Add a
   bounded module-level cache consulted by the within-set path only, keyed on
   `(source.content_hash, encoder descriptor, distance.descriptor, block_size, seed,
   bootstrap)` — everything the block value depends on. No signature changes anywhere, so
   plugins are unaffected. Mark it with a `ponytail:` comment naming the ceiling
   (process-global bounded LRU, not per-run; upgrade path is threading `RunMemo` through
   `estimate` if cross-run staleness ever matters). Verify by counting distance calls: a
   repeat call and a second cell sharing operand A must not re-evaluate `d(A,A)`.

2. **Declare statistic range** (gap 7, declaration half). The spec's static
   `Statistic.range` cannot work as written — a statistic's range depends on the distance
   (`Mean` inherits it, `Energy` is `(0, 2D)`, `DirectedMin` inherits it). The existing
   `range_from_distance` flag is the right hook. Expose a method
   **`range_for(distance) -> tuple[float, float]`** on all three, documented as the contract
   WP3's validation gate calls. **This is the cross-package interface: WP3 depends on it
   existing with exactly that name and signature.**

3. **`lifted(distance, reduction)`** (gap 8): a real helper returning a factorable
   `Statistic` built from a distance and a reduction over the pairwise matrix, with
   `range_for` derived from the distance. Do **not** rewrite the three existing statistics
   onto it. Test by building one new factorable statistic through `lifted()` and asserting
   it matches a hand-rolled equivalent.

## WP3 — API surface, registries, validation gate, manifest  *(audit gaps 4, 5, 7-check, 12)*

Files: `src/uncertain/registry.py`, `src/uncertain/core/specs.py`,
`src/uncertain/measures/__init__.py`, `src/uncertain/experiments/__init__.py`,
`src/uncertain/__init__.py`, `tests/test_auto_configuration.py`, `tests/test_core.py`.

1. **`register_unit` and `register_sampler`** (gap 4). Add both kinds to `_REGISTRY`
   (`registry.py:6-7`) and both decorators, exported from `uncertain/__init__.py`. Move the
   hardcoded unit tables — `core/specs.py:77-80` and `measures/__init__.py:39-42` — to
   consult the registry, registering the three built-in detectors (`token:v1`,
   `whitespace:v1`, `punctuation:v1`) at import. Acceptance test is §7's own: add a fourth
   unit from a test file **without editing anything under `src/`**.

2. **Make §5's printed examples run** (gap 5):
   - Normalize string `unit` and `representation` in `MeasureSpec.__post_init__`, mirroring
     what it already does for `statistic` at `core/specs.py:152-153`, so `replace(unit=...)`
     and `replace(representation=...)` stop raising `AttributeError`.
   - Parse the `"name:arg"` encoder form so `representation="layer:8"` resolves to
     `EncoderSpec("layer", {"layer": 8})`, making §5's `Sweep(level=["layer:0","layer:8"])`
     grid writable.
   - Accept `level=` as an alias for `representation=` in `from_config` and `replace`. The
     proposal uses both names; the alias is cheaper than a doc erratum.
   - Add `mode=` to `from_config`, so entropy and anticipatory — two of R3's six named
     cases — are reachable from the documented constructor.
   - Leave `Measure.from_config` as the named constructor; do **not** try to make
     `Measure(sampler=…, unit=…)` work. Two constructors with distinct jobs is the smaller
     surface. Note the deviation in `docs/api.md`.

3. **The domain/range gate** (gap 7, check half). Replace the hardcoded
   `warping == "neglog" and statistic.name in {"mean","energy"}` test at
   `measures/__init__.py:101-103` with a generic check: every warping declares `domain`,
   every statistic answers `range_for(distance)` (WP2's contract), and `validate()` rejects
   any pair whose range escapes the domain. A user-registered warping over a user-registered
   statistic must now be gated. Keep the failure message naming the offending pair, per §6.1.

4. **Truncation diagnostics** (gap 6, consumption half). Consume WP1's `unterminated` count
   in `_diagnostics` (`measures/__init__.py:296-302`). Where `distance.range[1]` is finite,
   report `truncation_bound = D × unterminated_fraction` with
   `truncation_status = "bounded_by_unterminated_mass"`. Where it is not finite, keep the
   current honest `None` / `"no_general_bound"`. Do not invent anything beyond this.

5. **Git SHA in the manifest** (gap 12): `git rev-parse HEAD` via subprocess in
   `experiments/__init__.py:169-183`, `None` when the directory is not a repository (it
   currently is not). Surface `seeds` as a manifest field rather than only nested in specs.

## WP4 — Oracles and test gaps  *(audit gaps 3, 9, 10, 11, 13, and the mypy claim)*

**Runs after WP1–WP3**, because golden values must be pinned from the fixed code and several
assertions target the new behaviour.

Files: `tests/test_measures.py`, `tests/test_hf.py`, `tests/test_golden.py`,
`tests/golden/`, `tests/test_store_sweeps.py`, `pyproject.toml`.

1. **Close the transpose negative control** — the audit's cheapest high-value fix.
   `tests/test_hf.py:26-29` builds the historically-buggy transposed array and asserts only
   that it differs from the correct table. Feed it back through the oracle's own assertion
   inside `pytest.raises`, so the test proves the oracle *rejects* the bug rather than that
   the fixture is discriminative.
2. **R4: assert `exact ≈ mc`** (gap 3). One test, one configuration, large n, asserting the
   closed-form and Monte Carlo paths agree within MC error. Nothing currently asserts it.
3. **Oracle 1's missing half** (gap 10): correlate real LM surprisal against the MC estimate
   across several items on `gpt2`, with the correlation threshold pinned alongside the
   existing convergence threshold. Mark slow if needed.
4. **Oracle 3** (gap 9): re-pin golden values on a fixed (model, context, seed) using `gpt2`,
   replacing the two-symbol analytic toy constants. Keep the `source` field honest — these
   are pinned from the corrected implementation, **not** recovered study re-run numbers,
   which `docs/validation-results.json` records as unavailable.
5. **Dataframe schema assertion** (gap 11): pin the expected column set in
   `tests/test_store_sweeps.py`. §1.4 and §10 both call the schema the contract; it is
   currently derived at runtime from `list(self.rows[0])` and asserted nowhere.
6. **The §9 properties the audit found missing** (gap 13): `pad` exercised end-to-end
   against `drop` and `compare_ragged` with the direction of movement asserted; metric axioms
   for core's own distances, not just the `TreeEdit` plugin; `effective_horizon` asserted
   under EOS, not only under token budget; energy-vanishes rewritten to use **two
   independent draws** from the same predictive distribution rather than `estimate(a, a)`,
   plus its companion half — mean pairwise distance staying bounded away from zero.
7. **Make the mypy claim reproducible**: add `mypy` to the dev extra (`pyproject.toml:16`),
   run it, fix what it reports. Add a lint rule banning `.reshape` (§10's "reshape banned in
   lint"), exempting the one legitimate `frombuffer(...).reshape(...)` round-trip at
   `utils/__init__.py:41`.

## Verification

After each WP: `.venv/Scripts/python -m pytest tests/ -q` and `ruff check .` — baseline is
91 passed, 0 failed, 0 skipped, and it must not regress.

End to end, after WP4:
- full suite green, plus `mypy` actually installed and passing
- the instrumented reuse check still shows one generation pass per position, not per cell
- the energy call-count check shows within-set blocks computed once, not three times
- `import uncertain` still performs no downloads and keeps `torch`/`transformers` out of
  `sys.modules`
- a fourth unit registered from a test file with nothing under `src/` edited
- `IMPLEMENTATION-STATUS.md` and `AUDIT-CONFORMANCE.md` updated to match reality: which gaps
  closed, which remain open by decision

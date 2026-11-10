# API contracts

`MeasureSpec` composes `SamplingSpec`, `UnitSpec`, `EncoderSpec`, `StatisticSpec`,
and `EstimatorSpec`. `to_dict()` / `from_dict()` round-trip without live objects.
The spec's SHA-256 captures its fields; `Measure.spec_hash` additionally captures
resolved component, sampler, backend, and engine identities. A result carries the
resolved hash. Python's process-randomized `hash()` is never a persistent key.

## Backends and samples

`Sampler(backend).sample(context, horizon, sampling_spec, unit)` returns an immutable
`AlternativeSet`. `sampler.extend(alternatives, k)` adds k draws and retains prior
sample IDs. Under the same environment it matches one request with n+k draws.

Backend probabilities describe the unmodified model. The reference sampler applies
temperature, stable top-k, then top-p, and records both log-probability streams.
Other generation controls fail validation. To measure the filtered distribution,
set `probability="sampling"`; the engine rejects a request to call it model-p entropy.

Raw samples are tuples of validated `Sequence` records. `arrays()` provides
immutable `(sample, step)` numeric views with explicit validity and observed masks.
Unavailable probabilities for observed anchors are `None` in records and `NaN`
in numeric views. Generated probabilities themselves must be finite.

For token units, `Sequence.text` is empty by design: decoded text is derived from
exact token IDs by `StringsEncoder`, including after truncation. For text units,
records retain text, unit character spans, and token ownership spans, including
lookahead. Overlapping token ownership is allowed when a tokenizer token straddles
two textual units. Raw alternatives retain lookahead for audit; encoders select
the completed units or their owning tokens.

For word/sentence prefix probabilities, equality means the normalized strings
emitted by the selected boundary detector. Monte Carlo counts all sampled
tokenizations producing that unit prefix. Exact token likelihoods must not be
substituted for those marginalized unit-event probabilities.

## Models and representations

```python
from uncertain import HFBackend, Sampler, EncoderSpec, Measure

# Use an immutable model commit revision, not a moving branch.
backend = HFBackend("MODEL_ID", revision="COMMIT_REVISION")
sampler = Sampler(backend)
# Construct SamplingSpec with backend.model_spec, n, seed.
```

`HFBackend(model=local_model, tokenizer=local_tokenizer)` fingerprints the local
weights. Empty contexts need BOS. Local token-only models can omit a tokenizer.
`score(context, tokens)` runs an independent teacher-forced forward pass.

`EncoderSpec("layer", {"layer": 2, "pooling": "mean", "context_mode": "conditioned"})`
uses the sampling backend by default. Unit vectors are token means; sequence
pooling is mean, first, last, or sum over those units. `context_mode="isolated"`
omits the original context. Padding requires a defined terminal vector and is
therefore rejected by this encoder.

For another encoding model, inject `LayerEncoder(backend=other_backend, layer=...)`
into `Measure(..., encoder=encoder)`. Its model and configuration enter the
resolved spec hash. Different tokenizer alignment across encoding models is not
implemented: the injected backend must accept the stored token ID vocabulary,
or use an external encoder that decodes and retokenizes explicitly.

## Extension protocol

Register a named factory with `register_encoder`, `register_distance`,
`register_statistic`, `register_warping`, `register_unit`, or `register_sampler`,
or inject an object into `Measure`.
Every component must have a canonical, versioned `descriptor`. Encoders declare
`output_type`, `supported_units`, and `encode(source, short_samples)` returning a
`RepresentationSet`. Pointwise scores declare `input_type`, `range`, `kind`,
`negative_type`, and `symmetric`, and implement `__call__(a, b)`.

Statistics expose `supports_uncertainty=True` and implement
`estimate(A, B, distance, estimator_spec, seeds) -> StatisticEstimate`. This flag
means the plugin returns an honest uncertainty status, not that every estimate
has a finite SE. Non-factoring statistics may read full representations and source
samples; they need not build a matrix. Warpings implement `apply(estimate)` and
return `(value, se, ci_low, ci_high, leading_bias)` with a declared `domain`.
Statistics declare `range_for(distance) -> (low, high)`; `Measure.validate` rejects
any pair whose statistic range escapes the warping domain, naming both. `neglog`
is undefined at 0, so it also rejects any statistic that can reach 0 unless it is a
probability (`estimates_probability(distance)`: a mean of a [0,1] similarity, such
as `prefix_indicator`); there, a zero estimate is the infinite surprisal. Units
declare the `boundary` detector they require; a unit whose detector is not one of
`token:v1`, `whitespace:v1`, or `punctuation:v1` also needs an encoder that lists
it in `supported_units`, and a boundary detector in `uncertain.alignment`.
Samplers are registered for name resolution only; `Measure.from_config` still
accepts an instance.

Do not mutate model weights, component state, or descriptors during a run. Version
changes to code/models/configuration must change the component descriptor. The
core cannot infer a reproducible identity for an arbitrary callback.

The POS example accepts a pinned tagger; the tree example uses optional `zss` and
a parser callback; the entailment example declares its clustering convention and
unavailable uncertainty. These examples live outside `src/uncertain`.

## Configuring a measure

`Measure.from_config(backend, ...)` is the named constructor; `Measure(spec, sampler=...)`
takes a resolved `MeasureSpec`. The proposal's keyword form also works:
`Measure(sampler=HFSampler("gpt2", n=32, horizon=5, top_k=0, seed=0), unit=..., ...)`
delegates to `from_config`, taking n/horizon/seed/generation defaults from the sampler.
`n` and `short_samples` stay required. On the constructor and on `replace`, a string
or spec for `representation`/`distance`/`statistic`/`warping` selects a registered
component; any other object is injected (`m.replace(distance=MyDistance())`).
Components written to the proposal's §7 protocol (e.g. a distance with only
`input_type`, `range`, `__call__`; a statistic with `range`, `__call__(A, B)`) are
wrapped automatically: they get a default descriptor, a statistic gets a
whole-alternative bootstrap SE, and a distance that declares no `negative_type` is
treated as not negative-type, so `energy` refuses it. Both paths
normalize string axis values identically, in `MeasureSpec.__post_init__`, so
`from_config(unit="word")` and `measure.replace(unit="word")` accept the same values.
`representation` also accepts the `"name:arg"` form, e.g. `"layer:8"` for
`EncoderSpec("layer", {"layer": 8})`, and `level=` is an accepted alias for it in
`from_config` and in both `replace` methods.

## Sweeps, artifacts, and results

`Sweep(base, **axes).plan(stimuli)` validates every grid cell before sampling and
groups compatible requests by predictive state, n, and maximum horizon. All cells
use one unit; use separate sweeps to change units or sampling settings. Invalid
Cartesian cells raise `ConfigError`; explicit compatible cell dictionaries can be
passed using `cells=[...]`.

With pandas installed, `run()` returns a tidy DataFrame that also carries `rows`,
`manifest`, `to_dataframe()` and `write()`; without pandas it returns `SweepResult`,
whose `rows` need no dataframe dependency. `to_dataframe()` uses nullable pandas floats. `write(path)` supports CSV, JSON,
and Parquet and writes a companion manifest. CSV/JSON use `Infinity` and
`-Infinity` strings for infinite scores; unavailable values are empty/null.
Parquet retains numeric infinities and nulls. Every row includes target span,
transform lineage, axes, resolved spec hash, sample counts, result and uncertainty
status, and artifact references. Optional datum values do not affect sampling.

`AlternativeStore(path=None)` is memory-only. On disk, one artifact type is stored:
`alternatives/<content_hash>.json`, with versioned numeric arrays/records and full
provenance; `requests/` holds small lookup pointers. Both content and pointer
compatibility are checked. JSON avoids pickle and remains inspectable; a compact
binary encoding can be added later without changing the logical schema.

`compute` returns a `Result`, not the samples. `measure.alternatives(result)`
loads the sets it rests on from the store: `(A,)` or `(A, continuation)`, and `()`
for exact results, which draw no samples. `AlternativeSet.summary(top=20)` returns
a text table of the most frequent continuations (count, share, mean model −log p).
`AlternativeSet.plot(top=20, ax=None)` draws the same thing as a matplotlib bar
chart (`plot` extra). Token units carry no text, so pass `decode=backend.decode`
to label them:

```python
result = measure.compute(item)
a, *_ = measure.alternatives(result)
print(a.summary(decode=measure.sampler.backend.decode))
a.plot(decode=measure.sampler.backend.decode)
```

`measure.sentence_summary(stimuli, top=3)` prints one line per stimulus (position,
target, value, top alternatives), e.g. over `positions(tokens)`. Exact results list
the most probable next tokens instead of samples. Stored samples are reused.

`contrast(left_result, right_result)` uses aligned bootstrap replicates retained
in memory. It preserves covariance for common sample streams. Different retained
subsets of a shared stream are rejected, and non-finite or unsupported joint
uncertainty is explicitly reported. These transient bootstrap draws are not part
of the alternatives artifact or dataframe. Regression and hypothesis tests are
outside the library.

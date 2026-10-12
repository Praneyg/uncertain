# uncertain

A working **research alpha** for configurable predictive uncertainty. Choose the
unit, horizon, representation, score/distance, set statistic, warping and sampler.
The default estimator evaluates supported token/H=1 measures exactly and uses
Monte Carlo for longer horizons or text units. Sampling defaults to temperature 1,
top_k 0 and top_p 1, without truncation or first-step EOS suppression. Alternatives
can be reused across compatible configurations. Results include sample accounting,
uncertainty status and provenance. Importing the package performs no downloads.

The import is `uncertain`; the provisional distribution name is `uncertain-altsets`.
This source tree has not been published to a package index.

## Install and run

From this directory, with Python 3.11 or later:

```sh
python -m pip install -e .
uncertain demo --output scores.csv --cache .cache/alternatives
```

Only NumPy is required for the reference engine. Install optional capabilities with:

```sh
python -m pip install -e ".[hf,tables,dev]"
python -m pytest -q
python -m ruff check src tests examples
```

The development extra includes Hypothesis. Hugging Face tests initialize a tiny
random GPT-2 locally; they do not download pretrained weights. The reference
backend intentionally runs one sequence at a time to make probability alignment
and indexed random streams easy to verify. Large-model GPU throughput is not yet
optimized.

## Python example

Configure the scientific choices directly:

```python
from uncertain import Measure, ToyBackend, Stimulus

measure = Measure.from_config(
    ToyBackend(),
    unit="token", horizon=1, representation="tokens",
    distance="discrete", statistic="energy", warping="identity",
    estimator="auto", n=512, seed=42, short_samples="compare_ragged",
)
result = measure.compute(Stimulus(target=(1,)))
print(result.value, result.method)  # 1.125, exact; no sampling

longer = measure.replace(horizon=3)
print(longer.compute(Stimulus(target=(1,))).method)  # mc
```

See [configuration and estimator selection](docs/configuration.md) for custom
components, sampler selection, exact weighted expectations and migration details.

The lower-level probability/surprisal presets and sweep API remain available:

```python
from uncertain import (
    ToyBackend, Sampler, SamplingSpec, EstimatorSpec, Measure,
    AlternativeStore, Stimulus, Sweep, surprisal,
)

backend = ToyBackend()  # exact next-token probabilities: a=.75, b=.25
sampling = SamplingSpec(backend.model_spec, n=512, seed=17)
measure = Measure(
    surprisal(sampling, short_samples="compare_ragged"),
    sampler=Sampler(backend),
    store=AlternativeStore(".cache/alternatives"),
)
item = Stimulus(context=(), target=(1,), stimulus_id="observed-b")
result = measure.compute(item)
print(result.value, result.se, result.status)

exact = measure.replace(estimator=EstimatorSpec(method="exact")).compute(item)
print(exact.value)  # log(4) = 1.3862943611198906 nats

plan = Sweep(measure, horizon=[1, 2, 4], warping=["identity", "neglog"]).plan([item])
print(plan.describe())  # validates the grid and lists required generation work
run = plan.run()
run.write("scores.csv")  # also writes scores.manifest.json
# frame = run.to_dataframe()  # requires the tables extra
```

To see what the model actually sampled for a Monte Carlo result, use
`a, *_ = measure.alternatives(result)` and then `print(a.summary(decode=backend.decode))`,
or `a.plot(...)` with the `plot` extra. See [docs/api.md](docs/api.md).

`n` and `short_samples` are required everywhere, `Measure.from_config` included:
`compare_ragged`, `pad` and `drop` are the available policies. No epsilon is inserted into a zero probability: sampled surprisal can
be infinite and its status/interval explain why.

## Implemented capabilities

- Immutable, serializable specs and versioned extension registries.
- Toy and Hugging Face causal-model sampling, indexed sample extension, separate
  raw-model and filtered-sampling log probabilities, and teacher-forced scoring.
- Token units and explicit `whitespace:v1` word / `punctuation:v1` sentence detectors.
- Pure sample truncation and observed-span anchoring; duplicate draws retain their
  multiplicity. A one-unit target at H=1 is a deterministic singleton operand.
- Token/unit, decoded-string, and hidden-layer encoders. A separately supplied
  encoder backend can differ from the sampling model.
- Prefix indicator, n-gram overlap/distance, discrete, cosine, and Euclidean scores;
  mean, directed minimum, and V/U energy statistics; identity and negative log.
- Exact token-event probability/surprisal and next-token entropy; MC token-sequence
  entropy; a defined, experimental nested anticipatory expectation.
- Exact one-token weighted means and energy distance with selectable encoders,
  scores and warpings; automatic exact/MC estimator selection.
- Sample-level bootstrap accounting, Wilson probability intervals, bounded-energy
  concentration intervals, optional negative-log bias correction, and paired contrasts.
- Content-addressed, checksummed, atomic JSON alternatives; generation-free replay;
  bounded encoding memoization; preplanned horizon sweeps; CSV/JSON/Parquet outputs.
- Optional data values, token-position stimulus generation, JSON/JSONL corpus
  loading, and deterministic context transforms.
- External POS, tree-edit, and entailment-clustering extension examples.

## Scientific and release limits

This is an implementation of the planned engine. GPT-2 exact probability and
surprisal have now been checked against all 1,487 observations used by the local
study notebook. Study-specific sampling estimators and a reproducible pilot are
available in `uncertain.emnlp2024`; they are distinct from the generic energy and
nested anticipatory presets. Full paper replication is still a separate milestone.
See [replication plan and definitions](docs/replication-plan.md) and
[runners and output files](replication/README.md).

Word/sentence distances and Monte Carlo prefix-event probabilities work with the
explicit simple detectors. Those probabilities count matching normalized unit
strings, thereby summing over sampled tokenizations. Exact word/sentence probability
and entropy presets remain unsupported. The sentence splitter is punctuation based,
not abbreviation aware.
Ambiguous prefix retokenization and decoders that revise previous characters are
rejected for text-unit alignment; a byte-level split of one multi-byte character
(GPT-2) is not a revision and aligns once the character completes.

Directed minima carry a whole-alternative bootstrap SE (`bootstrap_approximate`);
the entailment example returns `se=None` and an explicit unavailable status. Energy has an
approximate bootstrap SE; bounded distances additionally receive conservative
population confidence bounds that remain valid at the equal-distribution
degeneracy. For unbounded energy, the population CI is unavailable. No generic
full-sequence truncation bound is claimed.

Alternative persistence avoids regeneration; hidden-layer encoders still need the
pinned model resources and can perform GPU work. Reproducible regeneration assumes
the same model, tokenizer, execution policy, hardware, and dependencies. There is
no cross-hardware bitwise guarantee.

See [API and extension contracts](docs/api.md), [mathematical decisions](docs/design-decisions.md),
and [validation and remaining release gates](docs/validation.md).

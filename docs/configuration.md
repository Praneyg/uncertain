# Configurable measures and estimator selection

The core API defines a measure through independent scientific choices:

1. Unit: token, word or sentence (with explicit boundary rules).
2. Horizon: number of future units.
3. Representation: token/unit identities, text, hidden layer, or a registered encoder.
4. Scoring/distance function: built-in or user-defined through the registry.
5. Set statistic: mean, energy, directed minimum, or a registered statistic.
6. Warping: identity, negative log, or a registered transformation applied last.
7. Sampler: a supplied sampler instance; the reference ancestral sampler is the default.
8. Estimator: automatic by default, with explicit Monte Carlo or exact overrides.

```python
from uncertain import HFBackend, Measure, EncoderSpec

backend = HFBackend("openai-community/gpt2",
                    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
measure = Measure.from_config(
    backend,
    unit="token", horizon=1,
    representation="tokens",
    distance="discrete", statistic="energy", warping="identity",
    estimator="auto", n=512, seed=42,
)

# Hidden-layer representations use the same API with a full EncoderSpec:
# representation=EncoderSpec("layer", {"layer": 0, "context_mode": "conditioned"})
# paired with distance="cosine".
```

`MeasureSpec` and `Measure(...)` remain available for low-level component injection.
Use `register_encoder`, `register_distance`, `register_statistic`, `register_warping`,
`register_unit` and `register_sampler` to define named components with stable provenance
descriptors. All six kinds are open: a new unit supplies a boundary detector and is
resolved by `alignment.boundaries`, so adding one edits nothing under `src/`.
The reference defaults are temperature 1, top_k 0 and top_p 1 (truncation disabled),
ancestral sampling and no artificial first-step EOS suppression. Filtered sampling
requires explicit `probability="sampling"`; it is not silently treated as raw p.

## Automatic estimator policy

| Configuration | Default |
|---|---|
| Token unit, H=1, probability/surprisal, mean score, energy or next-token entropy | Exact |
| Token unit, H>1 | Monte Carlo |
| Word or sentence unit, any H | Monte Carlo |
| Token/H=1 with an unsupported exact statistic, such as directed minimum | Monte Carlo, with a diagnostic reason |

`estimator="mc"` always requests sampling. `estimator="exact"` requires an
implemented exact evaluator and raises a configuration error otherwise. Explicit
exact token-prefix probability remains supported for multiple observed tokens;
this mathematical exception does not alter the default H>1 Monte Carlo policy.
The existing low-level drop policy is outside the exact path. A model backend
must expose its full next-token distribution for exact evaluation.

For token/H=1 means, enumerate outcomes with positive model probability and compute
`sum_t p(t | context) * score(repr(t), repr(observed))`, then apply the warping.
EOS retains its probability mass. Outcomes are probability-weighted, never treated
as an equally weighted artificial sample.

This follows the finite-vocabulary expected-distance and similarity-adjusted
surprisal formulations in Meister, Giulianelli and Pimentel (2024), equations 3
and 6: [Towards a Similarity-adjusted Surprisal Theory](https://aclanthology.org/2024.emnlp-main.921/).
The paper's expected-distance mean should not be confused with energy distance.

For energy against a deterministic observed token y, compute
`2 E[d(X,y)] - E[d(X,X')] - d(y,y)`, with independent X and X' from p. The
implementation uses probability-mass identities for discrete distance and a
normalized-mean identity for cosine distance. Other negative-type distances use
weighted pairwise enumeration. Exact means and energy report zero sampling SE and
zero samples used. This is numerical evaluation under the chosen model, not a
claim that the model itself has no uncertainty.

Generic exact energy can cost O(|V|^2) distance evaluations. Context-dependent
representations may require model work for every vocabulary outcome; exact does
not mean cheap. Select explicit MC if that computation is impractical.

## Valid combinations and statistical interpretation

Components remain subject to their declared input types and domains. Energy
requires a negative-type dissimilarity, not an arbitrary similarity function.
Negative log requires a nonnegative statistic. A negative finite-sample U-energy
estimate cannot be passed to negative log. Registered warpings are applied after
the set statistic and must define how uncertainty transforms.

Unfiltered ancestral draws follow p without truncation bias. This does not imply
every statistic or nonlinear warping is unbiased. The default empirical energy
uses the V-statistic; select `StatisticSpec("energy", {"variant": "u"})` for its
U-statistic estimator. Those finite-sample variants evaluate the same population
quantity on the exact path.

Study-specific EOS suppression and study-specific off-diagonal estimators stay in
`uncertain.emnlp2024`; they are not global library defaults.

## Migration from 0.1.0a1

Version 0.1.0a2 changes `EstimatorSpec()` from MC to auto. Scripts specifically
studying sampling error should use `EstimatorSpec(method="mc")`. Existing explicit
exact reproduction runners retain their behavior. The new high-level
`Measure.from_config` defaults to discrete-distance energy; the low-level
`MeasureSpec` probability preset keeps its prefix-indicator/mean convention.
The engine version is part of result identities, separating new dispatch behavior
from previous results. Existing saved result files are not rewritten.

Run `python examples/configurable_measure.py` for an offline example that exercises
exact H=1, MC H=3, and a user-defined similarity with negative-log warping.

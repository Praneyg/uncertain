"""`Measure`: one configured uncertainty measure, computed per stimulus.

A measure compares alternatives A ~ p(.|context) with the observed target B (anchored
to its own sampled continuation up to horizon H) through encoder -> distance ->
statistic -> warping. Token/H=1 configurations with a finite statistic are evaluated
exactly; everything else is Monte Carlo over stored, content-addressed samples.
"""
from dataclasses import replace
import math
import numpy as np
from uncertain.core import (ConfigError, AlternativeSet, MeasureSpec, Result, StatisticEstimate,
                            EstimatorSpec, EncoderSpec, SamplingSpec, StatisticSpec)
from uncertain.data import Stimulus
from uncertain.store import AlternativeStore, RunMemo
from uncertain.registry import resolve, adapt, _SpecWarping
from uncertain.samplers import Sampler, sampling_probabilities
from uncertain.statistics import Energy, Mean, normal_interval, sample_mean, weights
from uncertain.measures.exact import finite_expectation
from uncertain._version import __version__
from uncertain.utils import digest, seed_for
from uncertain import distances
from uncertain import encoders as encoders, warpings as warpings  # register built-ins


def _is_axis_value(value):
    """A name or spec selects a registered component; anything else is an injected object."""
    return isinstance(value, (str, StatisticSpec, EncoderSpec))


class Measure:
    """A validated measure bound to a sampler and an alternatives store.

    Build one with `from_config` (named axes) or `Measure(spec, sampler=...)`; derive
    variants with `replace`; evaluate with `compute(stimulus) -> Result`. Construction
    resolves every component and validates the combination; it never samples.
    """

    @classmethod
    def from_config(cls, backend=None, *, unit="token", horizon=1, representation="tokens",
                    distance="discrete", distance_config=None, statistic="energy",
                    warping="identity", estimator="auto", sampler=None, n, seed=0,
                    generation=None, token_budget=128, short_samples,
                    probability="model", mode="observed", level=None, store=None):
        """Configure a measure along independent scientific axes.

        ``n`` and ``short_samples`` are required (§4.4, §4.2): how many alternatives a
        measure rests on and how short samples are treated are the experimenter's
        choices, never defaults. Custom component names use the existing registries;
        representation and statistic also accept their full specs. Supply a sampler
        instance (or a registered sampler name) to select a different sampling algorithm.
        """
        if isinstance(sampler, str):
            if backend is None:
                raise ConfigError("Naming a sampler also requires a backend")
            sampler = resolve("sampler", sampler, backend=backend)
        if backend is None:
            if sampler is None:
                raise ConfigError("Supply a backend or a sampler with a backend")
            backend = sampler.backend
        if sampler is None:
            sampler = Sampler(backend)
        if level is not None:
            representation = level
        if isinstance(estimator, str):
            estimator = EstimatorSpec(method=estimator)
        # MeasureSpec normalizes string unit/representation/statistic, so replace()
        # accepts exactly what this constructor does.
        spec = MeasureSpec(
            SamplingSpec(backend.model_spec, n=n, seed=seed,
                         generation=generation or {}, token_budget=token_budget),
            horizon, short_samples, unit=unit, representation=representation,
            distance=distance, distance_config=distance_config or {}, statistic=statistic,
            warping=warping, estimator=estimator, mode=mode, probability=probability)
        return cls(spec, sampler=sampler, store=store)

    def __init__(self, spec: MeasureSpec | None = None, *, sampler, store=None, encoder=None,
                 distance=None, statistic=None, warping=None, memo=None, **axes):
        """``Measure(spec, sampler=...)`` takes a resolved spec. The §5 keyword form,
        ``Measure(sampler=HFSampler(..., n=32, horizon=5), unit=..., distance=...)``,
        builds the spec via ``from_config``; n and short_samples stay required, from
        the sampler or here. Strings/specs on distance/statistic/warping/representation
        are axis values; any other object is an injected component.
        """
        injected = dict(encoder=encoder, distance=distance, statistic=statistic, warping=warping)
        for key in ("distance", "statistic", "warping"):
            if _is_axis_value(injected[key]):
                axes[key] = injected[key]
                injected[key] = None
        if spec is None:
            spec = Measure.from_config(sampler=sampler,
                                       **{**getattr(sampler, "defaults", {}), **axes}).spec
        elif axes:
            spec = spec.replace(**axes)
        injected = {kind: None if c is None else adapt(kind, c, type(c).__name__, injected=True)
                    for kind, c in injected.items()}
        encoder, distance, statistic, warping = injected.values()
        self.spec, self.sampler = spec, sampler
        self.store = store if store is not None else AlternativeStore()
        self.memo = memo if memo is not None else RunMemo()
        self._injected = injected
        self.encoder = encoder or resolve("encoder", spec.representation.name,
                                         spec.representation.config, version=spec.representation.version,
                                         backend=sampler.backend)
        self.distance = distance or resolve("distance", spec.distance, spec.distance_config)
        stat_spec = spec.statistic
        assert isinstance(stat_spec, StatisticSpec)  # MeasureSpec normalizes strings
        self.statistic_spec = stat_spec
        self.statistic = statistic or resolve("statistic", stat_spec.name, stat_spec.config,
                                             version=stat_spec.version)
        self.warping = warping or resolve("warping", spec.warping)
        if isinstance(self.warping, _SpecWarping):
            self.warping.n = spec.sampling.n  # §7 gap(raw, n) needs the set size
        self.validate()
        # adapt()/resolve() guarantee a descriptor; digesting it rejects unserializable ones.
        self.spec_hash = digest({"spec": spec, "components": [c.descriptor for c in
                                (self.encoder, self.distance, self.statistic, self.warping)],
                                 "backend": sampler.backend.descriptor, "sampler": sampler.descriptor,
                                 "engine": __version__})

    def validate(self):
        """Reject any component combination that has no defined meaning (ConfigError)."""
        s = self.spec
        if getattr(self.sampler, "study_specific_distribution", False):
            raise ConfigError("Use estimate_study with StudySampler; generic Measure does not "
                              "model its first-step EOS conditioning")
        if s.sampling.model != self.sampler.backend.model_spec:
            raise ConfigError("sampling model/revision does not match backend")
        if self.encoder.output_type != self.distance.input_type:
            raise ConfigError(f"encoder {self.encoder.output_type} incompatible with distance {self.distance.input_type}")
        if s.unit.name not in self.encoder.supported_units:
            raise ConfigError("unit unsupported by encoder")
        if not getattr(self.statistic, "supports_uncertainty", False):
            raise ConfigError("statistic must supply an uncertainty contract")
        if self.statistic_spec.name == "energy":
            if not self.distance.negative_type or self.distance.kind != "dissimilarity":
                raise ConfigError("energy requires a negative-type dissimilarity")
        domain = getattr(self.warping, "domain", None)
        range_for = getattr(self.statistic, "range_for", None)
        if domain is None or range_for is None:
            raise ConfigError("warpings must declare a domain and statistics a "
                              "range_for(distance)")
        low, high = range_for(self.distance)
        if not (domain[0] <= low and high <= domain[1]):
            raise ConfigError(f"warping {s.warping} domain {tuple(domain)} does not contain "
                              f"statistic {self.statistic_spec.name} range {(low, high)}")
        probability = getattr(self.statistic, "estimates_probability", None)
        if (getattr(self.warping, "domain_open_low", False) and low <= domain[0]
                and not (probability and probability(self.distance))):
            raise ConfigError(f"warping {s.warping} is undefined at {domain[0]}, which statistic "
                              f"{self.statistic_spec.name} over {s.distance} can reach; only a "
                              "probability (a mean of a [0,1] similarity) may")
        if s.representation.name == "layer" and s.short_samples == "pad":
            raise ConfigError("layer encoder does not define a terminal padding embedding")
        if s.mode in {"entropy", "anticipatory"} and s.unit.name != "token":
            raise ConfigError("entropy/anticipation require token units until text event marginalization is defined")
        if s.mode == "entropy" and (s.short_samples == "drop" or s.warping != "identity"):
            raise ConfigError("entropy requires identity warping and no dropping of samples")
        if s.probability == "model" and not s.sampling.model_distribution:
            raise ConfigError("samples follow q, not model p; select probability='sampling' or unfiltered sampling")
        if s.estimator.method == "exact" and not self.exact_eligible:
            raise ConfigError("no exact path exists for this complete configuration")

    @property
    def exact_eligible(self):
        """An explicit exact request can also score a known multi-token prefix."""
        s = self.spec
        finite_statistic = type(self.statistic) in {Mean, Energy}
        return (callable(getattr(self.sampler.backend, "probabilities", None))
                and s.unit.name == "token" and s.short_samples != "drop"
                and s.sampling.token_budget >= s.horizon
                and ((s.mode == "entropy" and s.horizon == 1)
                     or (s.mode == "observed" and s.horizon == 1 and finite_statistic)
                     or (s.mode == "observed" and self._exact_prefix)))

    @property
    def _exact_prefix(self):
        """A known token prefix scored by teacher forcing: P(target | context)."""
        return (self.spec.distance == "prefix_indicator" and type(self.statistic) is Mean
                and self.spec.representation.name == "tokens"
                and not any(self._injected.values()))

    @property
    def uses_exact(self):
        """Automatic dispatch is token/H=1 only; explicit overrides are separate."""
        return self.exact_eligible and (self.spec.estimator.method == "exact" or (
            self.spec.estimator.method == "auto" and self.spec.unit.name == "token"
            and self.spec.horizon == 1))

    def replace(self, **changes):
        """A new Measure with spec fields changed; shares this one's sampler, store and memo.

        A name or spec re-resolves a component; an object injects it (§5).
        """
        if "level" in changes:
            changes["representation"] = changes.pop("level")
        injected = dict(self._injected)
        for spec_key, injection in (("representation", "encoder"), ("distance", "distance"),
                                    ("statistic", "statistic"), ("warping", "warping")):
            if spec_key in changes:
                # §5: m.replace(distance=MyDistance()) injects; a name re-resolves.
                value = changes[spec_key]
                injected[injection] = None if _is_axis_value(value) else changes.pop(spec_key)
        return Measure(self.spec.replace(**changes), sampler=self.sampler, store=self.store,
                       memo=self.memo, **injected)

    def _probabilities(self, context):
        p = self.sampler.backend.probabilities(context)
        return p if self.spec.probability == "model" else sampling_probabilities(p, self.spec.sampling.generation)

    def _exact(self, stimulus):
        s = self.spec
        if s.mode == "observed" and not self._exact_prefix:
            raw, diagnostics = finite_expectation(self, stimulus)
            result = self._result(StatisticEstimate(raw, 0., raw, raw), 0, 0, (),
                                  diagnostics, method="exact")
            return replace(result, se=0., warping_bias=0.,
                           corrected_value=result.value if s.estimator.correct_bias else None)
        if s.mode == "entropy":
            p = self._probabilities(stimulus.context)
            raw = float(-np.sum(p[p > 0]*np.log(p[p > 0])))
        elif s.probability == "model" and callable(getattr(self.sampler.backend, "score", None)):
            # One teacher-forced forward scores every target token; requirements()
            # already rejects EOS inside the target.
            log_probability = float(np.sum(self.sampler.backend.score(stimulus.context, stimulus.target)))
            raw = math.exp(log_probability)
        else:
            raw = 1.
            log_probability = 0.
            context = stimulus.context
            for i,t in enumerate(stimulus.target):
                p = self._probabilities(context)
                if not 0 <= t < len(p):
                    raise ConfigError("target token outside vocabulary")
                raw *= float(p[t])
                log_probability += math.log(p[t]) if p[t] > 0 else -math.inf
                if t == self.sampler.backend.eos_token_id and i < len(stimulus.target)-1:
                    raw = 0.
                    log_probability = -math.inf
                    break
                context += (t,)
        estimate = StatisticEstimate(raw, 0., raw, raw)
        result = self._result(estimate, 0, 0, (), {}, method="exact")
        if s.mode == "observed" and s.warping == "neglog":
            result = replace(result, value=-log_probability, ci_low=-log_probability,
                             ci_high=-log_probability)
        return replace(result, se=0., warping_bias=0.,
                       corrected_value=result.value if s.estimator.correct_bias else None)

    def requirements(self, stimulus):
        """Validate the target and list the (context, horizon) sample requests it needs.

        Empty when the result is exact. Sweep plans merge these across cells.
        """
        if self.spec.mode == "anticipatory":
            return []  # requires hypothetical contexts; enumerated explicitly by compute
        if self.spec.mode == "observed":
            eos = self.sampler.backend.eos_token_id
            if eos is not None and eos in stimulus.target[:-1]:
                raise ConfigError("target cannot contain tokens after EOS")
            observed = stimulus.observed(self.sampler.backend, self.spec.unit)
            k = len(observed.units)
            if not 1 <= k <= self.spec.horizon:
                raise ConfigError("observed span must contain between 1 and H units")
            if ("prefix_length" in self.spec.distance_config and
                    self.spec.distance_config["prefix_length"] != k):
                raise ConfigError("prefix_length must match the observed event length")
        else:
            k = 0
        if self.uses_exact:
            return []
        requests = [(stimulus.context, self.spec.horizon)]
        if (self.spec.mode == "observed" and self.spec.horizon > k and
                stimulus.target[-1] != self.sampler.backend.eos_token_id):
            requests.append((stimulus.context+stimulus.target, self.spec.horizon-k))
        return requests

    def _get(self, context, horizon):
        return self.store.get_or_sample(self.sampler, context, horizon, self.spec.sampling, self.spec.unit)

    def alternatives(self, result):
        """The AlternativeSets a Result rests on: (A,) or (A, continuation); () if exact."""
        return tuple(self.store.load(k) for k in result.artifact_ids)

    def sentence_summary(self, stimuli, top=3):
        """One line per stimulus (e.g. ``positions(tokens)``): target, value and the top
        alternatives at that position. Exact results draw no samples, so they list the
        most probable next tokens instead. Stored samples are reused, not redrawn.
        """
        decode = self.sampler.backend.decode
        lines = [f"{'pos':>4}  {'target':16} {'value':>7}  top alternatives"]
        for s in stimuli:
            r = self.compute(s)
            sets = self.alternatives(r)
            if sets:
                alts = [(text, share) for text, _, share, _ in sets[0]._grouped(top, decode)]
            else:
                p = self._probabilities(s.context)
                alts = [(decode((int(t),)), float(p[t])) for t in np.argsort(-p, kind="stable")[:top]]
            listed = ", ".join(f"{text!r} {share:.0%}" for text, share in alts)
            lines.append(f"{s.position:>4}  {decode(s.target)!r:16} {r.value:>7.2f}  {listed}")
        return "\n".join(lines)

    def _encode(self, source):
        key = digest((source.content_hash, self.encoder.descriptor, self.spec.short_samples))
        return self.memo.get_or_compute(key, lambda: self.encoder.encode(source, self.spec.short_samples))

    def compute(self, stimulus):
        """Evaluate the measure on one Stimulus. Samples are drawn or reused via the store."""
        if not isinstance(stimulus, Stimulus):
            raise TypeError("compute expects a Stimulus")
        if self.spec.mode == "anticipatory":
            return self._anticipatory(stimulus)
        self.requirements(stimulus)  # validate target span before sampling
        if self.uses_exact:
            return self._exact(stimulus)
        a = self._get(stimulus.context, self.spec.horizon)
        artifacts = [self.store.save(a)]
        diagnostics = {**self._diagnostics(a), "truncation_bound": None,
                       "truncation_status": "no_general_bound"}
        if self.spec.mode == "entropy":
            stream = "model_logprobs" if self.spec.probability == "model" else "sampling_logprobs"
            values = [-sum(getattr(row, stream)) for row in a.sequences]
            estimate = sample_mean(values, self.spec.estimator.confidence)
            seed = seed_for("bootstrap", a.provenance.get("stream", a.content_hash))
            wa = weights(a, self.spec.estimator.bootstrap, seed)
            estimate = replace(estimate, bootstrap_values=tuple(wa @ values / len(values)))
            diagnostics.update({"resampling_seed_a": seed, "sample_ids_a": [s.sample_id for s in a.sequences]})
            return self._result(estimate, len(a), 0, tuple(artifacts), diagnostics)
        observed = stimulus.observed(self.sampler.backend, self.spec.unit)
        remaining = self.spec.horizon-len(observed.units)
        context = stimulus.context+stimulus.target
        terminal = stimulus.target[-1] == self.sampler.backend.eos_token_id
        if remaining and not terminal:
            continuation = self._get(context, remaining)
            artifacts.append(self.store.save(continuation))
        else:
            continuation = AlternativeSet.empty(context, self.spec.unit)
            if remaining:  # ended at EOS: the target is complete, nothing to continue
                continuation = replace(continuation, horizon=remaining)
        b = AlternativeSet.anchor(observed, continuation, stimulus.context)
        diagnostics["target"] = self._diagnostics(b)
        if self.spec.short_samples == "drop":
            old_a, old_b = len(a), len(b)
            a = a.select([i for i,row in enumerate(a.sequences) if len(row.units) == a.horizon])
            if not b.deterministic:
                b = b.select([i for i,row in enumerate(b.sequences) if len(row.units) == b.horizon])
            diagnostics["drop_fraction_a"] = 1-len(a)/old_a
            diagnostics["drop_fraction_b"] = 1-len(b)/old_b
        if not len(a) or not len(b):
            return self._result(StatisticEstimate(math.nan, None, None, None, "all_samples_dropped"),
                                len(a), len(b), tuple(artifacts), diagnostics)
        distance = self.distance
        if self.spec.distance == "prefix_indicator" and self._injected["distance"] is None:
            distance = distances.PrefixIndicator(len(observed.units))
        ra, rb = self._encode(a), self._encode(b)
        seeds = (seed_for("bootstrap", a.provenance.get("stream", a.content_hash)),
                 seed_for("bootstrap", continuation.provenance.get("stream", continuation.content_hash)))
        estimate = self.statistic.estimate(ra, rb, distance, self.spec.estimator, seeds)
        # The anchored target is open unless it ends at EOS: at H = k its own continuation
        # past the observed units is the missing mass.
        open_a = self._open_fraction(a, "unterminated" in a.provenance)
        open_b = 0. if terminal else self._open_fraction(
            b, b.deterministic or "unterminated" in continuation.provenance)
        bound, status = self._truncation(estimate, distance, open_a, open_b)
        diagnostics.update({"truncation_bound": bound, "truncation_status": status,
                            "unterminated_fraction_a": open_a, "unterminated_fraction_b": open_b,
                            "random_a": not a.deterministic, "random_b": not b.deterministic,
                            "resampling_seed_a": seeds[0], "resampling_seed_b": seeds[1],
                            "bootstrap_replicates": self.spec.estimator.bootstrap,
                            "sample_ids_a": [s.sample_id for s in a.sequences],
                            "sample_ids_b": [s.sample_id for s in b.sequences]})
        return self._result(estimate, len(a), len(b), tuple(artifacts), diagnostics)

    def _result(self, estimate, n_a, n_b, artifacts, diagnostics, method="mc"):
        """Warp a raw StatisticEstimate into the reported Result."""
        diagnostics = {**diagnostics, "estimator_requested": self.spec.estimator.method}
        if self.spec.estimator.method == "auto":
            diagnostics["estimator_selection"] = (
                "exact_token_horizon_one" if method == "exact" else
                "mc_exact_configuration_unsupported" if self.spec.unit.name == "token"
                and self.spec.horizon == 1 else "mc_unit_or_horizon")
        if math.isnan(estimate.value):
            value, se, low, high, bias = math.nan, None, None, None, None
        else:
            value, se, low, high, bias = self.warping.apply(estimate)
        corrected = value-bias if self.spec.estimator.correct_bias and bias is not None else None
        resamples = tuple(self.warping.apply(StatisticEstimate(float(x), None, None, None))[0]
                          for x in estimate.bootstrap_values)
        return Result(value, estimate.value, se, low, high, self.spec.sampling.n,
                      n_a, n_b, self.spec_hash, method, estimate.status,
                      self.spec.estimator.confidence, artifacts, bias, corrected, diagnostics, resamples)

    def _diagnostics(self, source):
        """Per-set stopping summary recorded on every MC result."""
        n = len(source)
        return {"effective_horizons": [len(s.units) for s in source.sequences],
                "eos_fraction": sum(s.stop_reason == "eos" for s in source.sequences)/n,
                "budget_fraction": sum(s.stop_reason == "budget" for s in source.sequences)/n,
                "estimand": "finite_horizon"}

    @staticmethod
    def _open_fraction(source, recorded):
        """§4.7: share of draws cut off at H or by the token budget; only they have mass
        past the horizon (an EOS stop is complete). Counted from the sequences, not
        provenance, which a truncated view inherits from the longer set. None unless
        the stops are the library sampler's own record (`recorded`)."""
        if not recorded or not len(source):
            return None
        return sum(s.stop_reason != "eos" for s in source.sequences)/len(source)

    def _truncation(self, estimate, distance, open_a, open_b):
        """§4.7 bound on |M_H - M_inf|, in the units of the reported value: the
        statistic's sensitivity to the open draws of both operands, pushed through f."""
        if open_a is None or open_b is None:
            return None, "no_general_bound"
        if not (open_a or open_b):
            return 0., "bounded_by_unterminated_mass"  # every draw ended at EOS
        bound_for = getattr(self.statistic, "truncation_bound", None)
        if bound_for is None or not all(map(math.isfinite, distance.range)):
            return None, "no_general_bound"
        low, high = self.statistic.range_for(distance)
        raw = min(bound_for(open_a, open_b, distance), high-low)

        def f(x):
            return self.warping.apply(StatisticEstimate(x, None, None, None))[0]
        x = estimate.value
        fx = f(x)
        ends = [f(min(max(x+d, low), high)) for d in (-raw, raw)]
        # e == fx also covers inf == inf (a zero statistic under -log stays at inf).
        return max(0. if e == fx else abs(e-fx) for e in ends), "bounded_by_unterminated_mass"

    def _anticipatory(self, stimulus):
        """Exact next-token marginal, independent inner MC per hypothetical target.

        General nested expectation, not labeled as a published IAS estimator.
        Independent streams permit a variance sum without ignoring covariance.
        """
        p = self._probabilities(stimulus.context)
        values, ses, artifacts = [], [], []
        n_a = n_b = 0
        for token in np.flatnonzero(p):
            sampling = replace(self.spec.sampling,
                               seed=seed_for(self.spec.sampling.seed, "anticipatory", int(token)))
            inner = self.replace(mode="observed", sampling=sampling)
            r = inner.compute(replace(stimulus, target=(int(token),)))
            values.append(float(p[token])*r.value)
            ses.append(None if r.se is None else (float(p[token])*r.se)**2)
            artifacts.extend(r.artifact_ids)
            n_a += r.n_a_used
            n_b += r.n_b_used
        value = float(sum(values))
        se = None if None in ses else math.sqrt(sum(x for x in ses if x is not None))
        low, high = normal_interval(value, se, self.spec.estimator.confidence)
        return Result(value, value, se, low, high, self.spec.sampling.n,
                      n_a, n_b, self.spec_hash, "anticipatory",
                      "nested_approximate" if se is not None else "inner_uncertainty_unavailable",
                      self.spec.estimator.confidence, tuple(artifacts),
                      diagnostics={"outer_marginal": "exact", "hypotheses": len(values),
                                   "inner_streams": "independent", "definition": "E_U[f(s(A,B_U))]"})


# Presets: MeasureSpec factories for the standard measures. Pass the result to Measure.

def contextual_probability(sampling, *, short_samples, **kwargs):
    """P(target | context): mean prefix indicator over sampled alternatives (H=1 default)."""
    return MeasureSpec(sampling=sampling, horizon=kwargs.pop("horizon", 1),
                       short_samples=short_samples, **kwargs)


def surprisal(sampling, **kwargs):
    """-log P(target | context), in nats."""
    return contextual_probability(sampling, warping="neglog", **kwargs)


def entropy(sampling, **kwargs):
    """Expected -log p of the sampled continuations (no target needed)."""
    return contextual_probability(sampling, mode="entropy", **kwargs)


def information_value(sampling, *, distance="discrete", statistic="mean", **kwargs):
    """Mean distance between alternatives and the observed target."""
    return contextual_probability(sampling, distance=distance, statistic=statistic, **kwargs)


__all__ = ["Measure", "contextual_probability", "surprisal", "entropy", "information_value",
           "EstimatorSpec"]

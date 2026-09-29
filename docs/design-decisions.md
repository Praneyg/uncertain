# Mathematical and implementation decisions

1. H is the total comparison horizon. For K observed units, compare the predictive
   H-unit sample with the observed K-unit prefix plus an H-K continuation from its
   updated context. K must be between 1 and H. EOS is terminal.
2. An observed event at H=K is deterministic and represented by one sample, with
   unavailable sampling probabilities for its observed tokens. Random duplicates
   are never deduplicated. Context-dependent tokens are never silently retokenized.
3. Contextual probability is the fraction matching the whole observed token prefix;
   surprisal is its negative natural logarithm. Exact token scoring accumulates
   logarithms to avoid probability-product underflow. Exact paths have zero random
   draws used and SE=0, while retaining the requested n.
4. Token entropy is a mean of negative sequence log probabilities under the same
   distribution that generated the samples. EOS terminates a sequence. A token
   budget shorter than H changes the event; diagnostics expose this censoring.
5. The experimental anticipatory definition is E_U[f(s(A,B_U))]. The next-token
   marginal is enumerated; each hypothetical target uses an independent inner
   sampling seed. It can be expensive and is not labeled a published IAS preset.
6. Energy uses a negative-type dissimilarity. V includes diagonals; U excludes
   sample-position diagonals and requires two draws on each random side. A
   deterministic singleton has zero within-side dispersion. U values may be
   negative. Cosine energy measures differences between means of normalized
   vectors; it does not identify every difference between distributions.
7. N-gram overlap is set Jaccard similarity over n-grams; its complementary score
   is a dissimilarity. No negative-type guarantee is declared for that adapter.
   The minimum statistic is mean over B queries of each query's nearest A sample.
8. Independent pairwise entries are not assumed. Bootstrap resampling operates on
   whole alternatives, and common streams share resampling indices. Deterministic
   operands remain fixed. Bootstrap SEs are approximations, not exact precision.
9. For bounded energy with 0<=d<=D and d(x,x)=0, changing one A draw alters the
   statistic by at most 4D/n; similarly for B. McDiarmid's inequality gives radius
   sqrt(8 D^2 (1/n + 1/m) log(2/alpha)). Deterministic sides contribute zero.
   The V-statistic's nonnegative bias is at most D/n + D/m; the lower endpoint is
   adjusted by that amount. These conservative population intervals survive the
   equal-distribution degeneracy. An ordinary percentile bootstrap alone does not.
10. Negative-log leading plug-in bias is Var(raw)/(2 raw^2), not the population
    Jensen gap. Correction is optional and leaves the uncorrected score available.
    At zero hits, surprisal and its upper CI are infinite; SE/bias are unavailable.
11. Finite H is usually part of the estimand. EOS, budget censoring, and sampling
    error are separate diagnostics. There is no universal tail bound for arbitrary
    distances and warpings. Dropping short samples conditions on retention.
12. One sampling pass is reused only when horizon-independent ancestral transition
    and stopping policies permit it. The backend and sampler versions, realized
    generation controls, context, n, unit, seed, budget, and numerical policy enter
    identity. Every anchor/truncate/selection and encoder configuration has its own
    derived identity. Encoder forwards are still computation.
13. NumPy is the reference engine's only required dependency. PyTorch/Transformers
    are optional backend dependencies. This deliberate departure from the proposal
    keeps model-free analysis and tests lightweight. Numeric arrays are immutable
    byte-backed NumPy views; model tensors stay inside the HF adapter.
14. Alternatives use inspectable canonical JSON in the first artifact schema.
    Metadata and values are checksummed, files publish atomically, and no pickle
    is used. Request identity is separate from the actual content identity.
15. Simple named boundary detectors ship without a universal default language
    policy. MC word/sentence probability counts matching normalized unit strings,
    rather than one canonical tokenization. Exact probability and entropy for
    these units remain unsupported until event likelihoods are established.
16. Directed minima and empirical entailment clustering expose missing uncertainty
    explicitly. They are not advertised as satisfying the full scientific v1 gate.

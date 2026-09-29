r"""Runnable examples for the uncertain research-alpha library.

Run the offline examples from the project directory:

    .\.venv\Scripts\python.exe usage_examples.py

Optionally run a Hugging Face example with a pinned model revision:

    .\.venv\Scripts\python.exe usage_examples.py \
        --hf-model MODEL_ID --hf-revision IMMUTABLE_COMMIT_HASH

The default run uses only the built-in toy model and performs no downloads.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from uncertain import (
    AlternativeStore,
    EstimatorSpec,
    HFBackend,
    Measure,
    Sampler,
    SamplingSpec,
    Stimulus,
    Sweep,
    ToyBackend,
    contextual_probability,
    entropy,
    information_value,
    positions,
    surprisal,
)


def print_result(label: str, result) -> None:
    """Print the fields most users should inspect after a computation."""
    print(f"\n{label}")
    print(f"  value: {result.value}")
    print(f"  raw value: {result.raw_value}")
    print(f"  standard error: {result.se}")
    print(f"  confidence interval: ({result.ci_low}, {result.ci_high})")
    print(f"  samples used: A={result.n_a_used}, B={result.n_b_used}")
    print(f"  method: {result.method}")
    print(f"  status: {result.status}")


def run_toy_examples(output_dir: Path) -> None:
    """Run every model-free example against a known binary distribution."""
    print("Running offline toy-model examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The default toy model has P(token 0)=0.75 and P(token 1)=0.25.
    backend = ToyBackend()
    sampler = Sampler(backend)
    sampling = SamplingSpec(
        model=backend.model_spec,
        n=512,
        seed=17,
    )
    store = AlternativeStore(output_dir / "alternatives")
    observed_b = Stimulus(context=(), target=(1,), stimulus_id="observed-b")

    # 1. Monte Carlo surprisal with an uncertainty estimate.
    surprisal_measure = Measure(
        surprisal(
            sampling,
            horizon=1,
            short_samples="compare_ragged",
            estimator=EstimatorSpec(method="mc"),
        ),
        sampler=sampler,
        store=store,
    )
    print_result("Monte Carlo surprisal", surprisal_measure.compute(observed_b))

    # 2. Exact surprisal. The answer is -log(0.25) = log(4).
    exact_surprisal = surprisal_measure.replace(
        estimator=EstimatorSpec(method="exact")
    )
    print_result("Exact surprisal", exact_surprisal.compute(observed_b))

    # 3. Contextual probability for the same observed token.
    probability_measure = Measure(
        contextual_probability(
            sampling,
            horizon=1,
            short_samples="compare_ragged",
        ),
        sampler=sampler,
        store=store,
    )
    print_result("Contextual probability", probability_measure.compute(observed_b))

    # 4. Exact next-token entropy.
    entropy_measure = Measure(
        entropy(
            sampling,
            horizon=1,
            short_samples="compare_ragged",
            estimator=EstimatorSpec(method="exact"),
        ),
        sampler=sampler,
        store=store,
    )
    print_result("Exact next-token entropy", entropy_measure.compute(Stimulus()))

    # 5. Monte Carlo entropy for three-token sequences.
    sequence_entropy = Measure(
        entropy(
            sampling,
            horizon=3,
            short_samples="compare_ragged",
        ),
        sampler=sampler,
        store=store,
    )
    print_result("Three-token sequence entropy", sequence_entropy.compute(Stimulus()))

    # 6. Energy information value under discrete sequence distance.
    iv_measure = Measure(
        information_value(
            sampling,
            horizon=3,
            short_samples="compare_ragged",
            distance="discrete",
            statistic="energy",
        ),
        sampler=sampler,
        store=store,
    )
    print_result("Energy information value", iv_measure.compute(observed_b))

    # 7. Sampling from a filtered distribution q. This explicitly says the
    # measure targets q rather than pretending the samples came from raw model p.
    filtered_sampling = replace(
        sampling,
        generation={
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 2,
            "top_p": 1.0,
            "num_beams": 1,
        },
    )
    filtered_measure = Measure(
        contextual_probability(
            filtered_sampling,
            horizon=1,
            short_samples="compare_ragged",
            probability="sampling",
        ),
        sampler=sampler,
        store=store,
    )
    print_result("Probability under filtered sampling distribution q",
                 filtered_measure.compute(observed_b))

    # 8. Sweep several compatible horizons and statistics. The planner groups
    # sampling requests so smaller horizons reuse samples drawn at H_max.
    stimuli = positions(
        tokens=(0, 1, 0),
        data=(21.0, 35.0, 23.0),
        document_id="toy-document",
    )
    sweep = Sweep(
        iv_measure,
        horizon=[1, 2, 3],
        statistic=["mean", "energy"],
    )
    plan = sweep.plan(stimuli)
    print("\nSweep plan")
    print(plan.describe())

    run = plan.run()
    csv_path = run.write(output_dir / "scores.csv")
    print(f"\nWrote sweep results to {csv_path}")
    print(f"Wrote manifest to {csv_path.with_suffix('.manifest.json')}")

    # DataFrame support is optional and comes from the `tables` extra.
    try:
        frame = run.to_dataframe()
    except ImportError:
        print("Install the tables extra to create a pandas DataFrame.")
    else:
        print("\nSelected DataFrame columns")
        print(frame[["stimulus_id", "horizon", "statistic", "value", "se", "status"]])
        run.write(output_dir / "scores.parquet")
        print(f"Wrote Parquet results to {output_dir / 'scores.parquet'}")

    print(f"\nSampler calls made in this process: {sampler.calls}")
    print("Run the script again with the same output directory to exercise disk-cache reuse.")


def run_hugging_face_example(
    output_dir: Path,
    model_id: str,
    revision: str,
    context: str,
    target: str,
) -> None:
    """Run one sampled and one exact result with a pinned causal language model."""
    print("\nRunning optional Hugging Face example")
    backend = HFBackend(model_id, revision=revision)
    sampler = Sampler(backend)
    sampling = SamplingSpec(
        model=backend.model_spec,
        n=128,
        seed=42,
        generation={
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "num_beams": 1,
        },
        token_budget=64,
    )
    stimulus = Stimulus.from_text(
        context=context,
        target=target,
        backend=backend,
        stimulus_id="hugging-face-example",
    )
    measure = Measure(
        surprisal(
            sampling,
            horizon=len(stimulus.target),
            short_samples="compare_ragged",
            estimator=EstimatorSpec(method="mc"),
        ),
        sampler=sampler,
        store=AlternativeStore(output_dir / "hf-alternatives"),
    )
    print_result("Hugging Face Monte Carlo surprisal", measure.compute(stimulus))
    exact = measure.replace(estimator=EstimatorSpec(method="exact"))
    print_result("Hugging Face exact surprisal", exact.compute(stimulus))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("usage-example-output"),
        help="Directory for caches and sweep outputs (default: usage-example-output)",
    )
    parser.add_argument("--hf-model", help="Optional Hugging Face causal model ID")
    parser.add_argument("--hf-revision", help="Immutable model commit revision")
    parser.add_argument("--hf-context", default="The capital of France is")
    parser.add_argument("--hf-target", default=" Paris")
    args = parser.parse_args()
    if bool(args.hf_model) != bool(args.hf_revision):
        parser.error("--hf-model and --hf-revision must be supplied together")
    return args


def main() -> None:
    args = parse_args()
    run_toy_examples(args.output_dir)
    if args.hf_model:
        run_hugging_face_example(
            args.output_dir,
            args.hf_model,
            args.hf_revision,
            args.hf_context,
            args.hf_target,
        )
    else:
        print("\nHugging Face example skipped. Supply --hf-model and --hf-revision to run it.")


if __name__ == "__main__":
    main()

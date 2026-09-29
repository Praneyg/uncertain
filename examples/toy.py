"""Run after installing the package; writes output only to the requested directory."""
from pathlib import Path
import argparse
from uncertain import (ToyBackend, Sampler, SamplingSpec, Measure, AlternativeStore,
                       Sweep, positions, information_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    backend = ToyBackend()
    sampling = SamplingSpec(backend.model_spec, 128, 2026)
    measure = Measure(information_value(sampling, short_samples="compare_ragged"),
                      sampler=Sampler(backend), store=AlternativeStore(args.output/"alternatives"))
    sweep = Sweep(measure, horizon=[1,2,3], statistic=["mean", "energy"])
    plan = sweep.plan(positions((0,1,0), data=[21.,35.,23.]))
    print(plan.describe())
    plan.run().write(args.output/"scores.csv")


if __name__ == "__main__":
    main()

"""Small local demo and declarative JSON/YAML sweep runner."""
import argparse
import json
from pathlib import Path
from uncertain import (ToyBackend, HFBackend, Sampler, SamplingSpec, Measure, MeasureSpec,
                       AlternativeStore, Sweep, surprisal, positions, load_dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="uncertain")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a completely offline toy-model sweep")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--cache", type=Path)
    run = sub.add_parser("run", help="run a JSON/YAML configuration")
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        backend: ToyBackend | HFBackend = ToyBackend()
        sampling = SamplingSpec(backend.model_spec, n=256, seed=17)
        measure = Measure(surprisal(sampling, short_samples="compare_ragged"),
                          sampler=Sampler(backend), store=AlternativeStore(args.cache))
        plan = Sweep(measure, horizon=[1, 2, 4]).plan(positions((0, 1, 0)))
        output = args.output
    else:
        raw = args.config.read_text(encoding="utf-8")
        if args.config.suffix in {".yaml", ".yml"}:
            import yaml
            config = yaml.safe_load(raw)
        else:
            config = json.loads(raw)
        spec = MeasureSpec.from_dict(config["measure"])
        backend = HFBackend(spec.sampling.model.model_id, revision=spec.sampling.model.revision)
        measure = Measure(spec, sampler=Sampler(backend), store=AlternativeStore(config.get("cache")))
        plan = Sweep(measure, **config.get("axes", {})).plan(load_dataset(config["stimuli"]))
        output = Path(config["output"])
    print(json.dumps(plan.describe(), indent=2))
    if getattr(args, "dry_run", False):
        return
    plan.run().write(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

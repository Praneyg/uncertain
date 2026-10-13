"""Mutation check: can the eval suite fail? Each mutation is applied to a scratch COPY of
src/uncertain (the working tree has uncommitted library changes, so a git branch from HEAD
would test the wrong code); the relevant tests run against the copy and against the
unmutated source. A mutation is CAUGHT if some test that passes on the original fails on
the mutant. Writes eval/mutations.json.

    .venv/Scripts/python.exe eval/mutate.py
"""
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
SRC = ROOT / "src"
WORK = EVAL / ".mutants"

MUTATIONS = [
    dict(name="reshape_in_logits_path",
         about="StudySampler: reorder (batch, vocab) log-probs by reshape instead of a named swap",
         file="emnlp2024.py",
         old="logp = torch.log_softmax(logits, dim=-1).cpu().numpy()",
         new="logp = torch.log_softmax(logits, dim=-1).transpose(0, 1).reshape(logits.shape).cpu().numpy()",
         tests=["eval/tests/test_d_sampling.py"], k="batched or study_sampler"),
    dict(name="reshape_in_arrays",
         about="AlternativeSet.arrays: (sample, step) view rebuilt by reshape of the transpose",
         file="core/values.py",
         old="return {key: immutable_array(value, value.dtype) for key, value in",
         new="return {key: immutable_array(value.T.reshape(value.shape), value.dtype) for key, value in",
         tests=["eval/tests/test_d_sampling.py", "eval/tests/test_a_core.py"], k="arrays or alternative_set_fields"),
    dict(name="default_top_k_50",
         about="SamplingSpec default top_k=50 (the generalized-surprisal bug)",
         file="core/specs.py", old="top_k=0, top_p=1.0", new="top_k=50, top_p=1.0",
         tests=["eval/tests/test_d_sampling.py"], k="top_k or tail"),
    dict(name="energy_without_within_terms",
         about="Energy = 2 d(A,B) (drop d(A,A) and d(B,B))",
         file="statistics/__init__.py",
         old="value = 2*cross-within[0][0]-within[1][0]\n        boots = 2*bcross-within[0][1]-within[1][1]",
         new="value = 2*cross\n        boots = 2*bcross",
         tests=["eval/tests/test_a_core.py"], k="energy"),
    dict(name="extend_regenerates_everything",
         about="Sampler.extend re-samples all n1+n2 draws (values identical by per-draw seeding)",
         file="samplers/__init__.py",
         old="replace(original, n=k),\n                            alternatives.unit, start=len(ids))\n"
             "        sequences = alternatives.sequences + extra.sequences",
         new="replace(original, n=len(ids)+k),\n                            alternatives.unit, start=0)\n"
             "        sequences = extra.sequences",
         tests=["eval/tests/test_d_sampling.py"], k="extend"),
    # --- extra mutations beyond the four required -------------------------------------------
    dict(name="truncate_by_tokens_not_units",
         about="Sequence.truncate cuts at token H instead of the end of unit H",
         file="core/values.py",
         old="end = self.unit_spans[horizon - 1][1] if horizon else 0",
         new="end = min(horizon, len(self.tokens))",
         tests=["eval/tests/test_c_units.py"], k="truncate_equals"),
    dict(name="short_samples_not_hashed",
         about="MeasureSpec.spec_hash ignores the short-sample policy",
         file="core/specs.py",
         old="    def spec_hash(self):\n        return digest(self)",
         new="    def spec_hash(self):\n        return digest(replace(self, short_samples=\"compare_ragged\"))",
         tests=["eval/tests/test_c_units.py", "eval/tests/test_f_repro.py"], k="short_sample or every_spec_field"),
    dict(name="se_divides_by_n",
         about="sample_mean SE = sd/n instead of sd/sqrt(n)",
         file="statistics/__init__.py",
         old="values.std(ddof=1)/np.sqrt(len(values))", new="values.std(ddof=1)/len(values)",
         tests=["eval/tests/test_a_core.py", "eval/tests/test_e_estimates.py"],
         k="h1_information or se_scales or presets"),
    dict(name="continuation_from_wrong_context",
         about="right operand continuation sampled from c_t instead of c_t + observed",
         file="measures/__init__.py",
         old="continuation = self._get(context, remaining)",
         new="continuation = self._get(stimulus.context, remaining)",
         tests=["eval/tests/test_a_core.py"], k="horizon_mean or operands"),
]


def run(tests, k, src, xml):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if src != SRC:
        env["EVAL_SRC"] = str(src)
    cmd = [sys.executable, "-m", "pytest", *tests, "-k", k, "-q", "-p", "no:cacheprovider",
           "-o", f"pythonpath={Path(src).as_posix()}", f"--junitxml={xml}"]  # backslashes get mangled
    done = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    if not Path(xml).exists():
        raise RuntimeError("pytest produced no report:\n" + done.stdout[-2000:] + done.stderr[-2000:])
    outcome = {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        bad = case.find("failure") is not None or case.find("error") is not None
        outcome[name] = "skipped" if case.find("skipped") is not None else ("failed" if bad else "passed")
    return outcome


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    report = []
    for m in MUTATIONS:
        dst = WORK / m["name"]
        shutil.copytree(SRC / "uncertain", dst / "uncertain", ignore=shutil.ignore_patterns("__pycache__"))
        target = dst / "uncertain" / m["file"]
        text = target.read_text(encoding="utf-8")
        assert text.count(m["old"]) == 1, f"{m['name']}: anchor not found exactly once"
        target.write_text(text.replace(m["old"], m["new"]), encoding="utf-8")
        base = run(m["tests"], m["k"], SRC, WORK / f"{m['name']}.base.xml")
        mutant = run(m["tests"], m["k"], dst, WORK / f"{m['name']}.mutant.xml")
        killers = sorted(t for t, o in mutant.items() if o == "failed" and base.get(t) == "passed")
        report.append(dict(name=m["name"], about=m["about"], file=f"src/uncertain/{m['file']}",
                           tests_run=len(mutant), caught=bool(killers), killed_by=killers))
        print(f"{m['name']:34s} {'CAUGHT' if killers else 'NOT CAUGHT'}  ({len(killers)}/{len(mutant)} tests)")
    (EVAL / "mutations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()

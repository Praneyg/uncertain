"""Run the eval suite and grade every requirement in eval/requirements.md.

    .venv/Scripts/python.exe eval/grade.py            # suite + grading  -> eval/results.json
    .venv/Scripts/python.exe eval/mutate.py           # (separately) mutation check -> eval/mutations.json

Status rules: PASS if every covering test passed; FAIL if any failed or errored; NOT RUN if all
were skipped. OVERRIDES below re-label a computed status where the spec is ambiguous or the
feature is absent -- each carries its reason, and the underlying test outcomes stay in the output.
"""
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
JUNIT = EVAL / ".junit.xml"

OVERRIDES = {
    "23": ("AMBIGUOUS", "Batch invariance is not stated in the PDF (nearest: §6.3 bit-reproducibility). "
                        "Sub-checks are reported as measured."),
    "38": ("AMBIGUOUS", "§6.3 pooling is 'tokenizer offsets -> unit spans'; the library's pooling option "
                        "pools unit vectors into a span (token->unit is always mean). Tested: hashed + effective."),
    "40": ("AMBIGUOUS", "§4.5's example (-log over a range containing 0 must fail) contradicts the §5 "
                        "surprisal preset (-log over a [0,1] mean). Tested: domain violations are rejected."),
    "39": ("NOT IMPLEMENTED", "Golden values exist and pass, but are self-pinned from 0.1.0a2; §9 asks for the "
                              "corrected re-run numbers, which the repo records as unavailable."),
}


def requirement_ids():
    rows = re.findall(r"^\| (\d+b?) \| ([^|]+) \| ([^|]+) \|", (EVAL / "requirements.md").read_text(encoding="utf-8"),
                      flags=re.M)
    return {i: {"section": s.strip(), "requirement": r.strip()} for i, s, r in rows}


def run_suite():
    start = time.time()
    cmd = [sys.executable, "-m", "pytest", "eval/tests", "-q", "-p", "no:cacheprovider", f"--junitxml={JUNIT}"]
    done = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, text=True,
                          capture_output=True)
    print(done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-2000:])
    return time.time() - start


def parse_junit():
    tests = []
    for case in ET.parse(JUNIT).getroot().iter("testcase"):
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        if case.find("skipped") is not None:
            outcome = "skipped"
        elif case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        else:
            outcome = "passed"
        node = case.find("failure") if case.find("failure") is not None else case.find("error")
        tests.append(dict(name=f"{case.get('classname').split('.')[-1]}::{case.get('name')}", outcome=outcome,
                          reqs=[r for r in props.get("req", "").split(",") if r],
                          measured={k[2:]: v for k, v in props.items() if k.startswith("m:")},
                          failure=(node.get("message") or "")[:300] if node is not None else None))
    return tests


def usability():
    text = (EVAL / "usability_grading.md").read_text(encoding="utf-8")
    scores = {a: int(n) for a, n in re.findall(r'<score axis="([^"]+)">(\d)</score>', text)}
    return scores


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    duration = run_suite()
    reqs, tests = requirement_ids(), parse_junit()
    scores = usability()
    out = {}
    for rid, meta in reqs.items():
        covering = [t for t in tests if rid in t["reqs"]]
        outcomes = {t["outcome"] for t in covering}
        if rid == "25":
            status = "PASS" if scores and min(scores.values()) >= 3 else "FAIL"
        elif not covering:
            status = "NOT RUN"
        elif "failed" in outcomes:
            status = "FAIL"
        elif outcomes == {"skipped"}:
            status = "NOT RUN"
        else:
            status = "PASS"
        entry = {**meta, "status": status,
                 "tests": [{"name": t["name"], "outcome": t["outcome"], **({"failure": t["failure"]} if t["failure"] else {})}
                           for t in covering],
                 "measured": {t["name"]: t["measured"] for t in covering if t["measured"]}}
        if rid in OVERRIDES:
            entry["computed_status"], (entry["status"], entry["note"]) = status, OVERRIDES[rid]
        if rid == "25":
            entry["llm_scores"] = scores
            entry["llm_mean"] = round(sum(scores.values()) / len(scores), 2) if scores else None
        out[rid] = entry
    mutations = json.loads((EVAL / "mutations.json").read_text(encoding="utf-8")) if (EVAL / "mutations.json").exists() else None
    summary = {s: sum(e["status"] == s for e in out.values())
               for s in ("PASS", "FAIL", "AMBIGUOUS", "NOT IMPLEMENTED", "NOT RUN")}
    totals = {o: sum(t["outcome"] == o for t in tests) for o in ("passed", "failed", "skipped")}
    result = {"command": "python eval/grade.py", "suite_seconds": round(duration, 1), "test_totals": totals,
              "status_summary": summary, "requirements": out, "mutations": mutations}
    (EVAL / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"suite {duration:.0f}s  tests {totals}  requirements {summary}")
    for rid, e in out.items():
        print(f"  {rid:>4} {e['status']:16s} {e['requirement'][:80]}")


if __name__ == "__main__":
    main()

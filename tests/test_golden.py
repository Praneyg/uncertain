import json
from pathlib import Path
import pytest
from uncertain import (EstimatorSpec, Stimulus, Sampler, SamplingSpec, MeasureSpec, Measure)

GOLDEN = Path(__file__).parent/"golden"


def test_independent_analytic_golden_values(make_measure):
    gold = json.loads((GOLDEN/"toy-values.json").read_text())
    m = make_measure(estimator=EstimatorSpec(method="exact"))
    assert m.compute(Stimulus((), (1,))).value == gold["probability_b"]
    assert m.replace(warping="neglog").compute(Stimulus((), (1,))).value == gold["surprisal_b"]
    assert m.replace(horizon=2).compute(Stimulus((), (0,1))).value == gold["probability_ab"]
    assert m.replace(mode="entropy").compute(Stimulus()).value == pytest.approx(gold["next_token_entropy"])


@pytest.mark.gpt2
def test_gpt2_golden_values(gpt2):
    """§9 oracle 3 on the path the founding bug lived in: a fixed (model, context, seed).

    `token_surprisals` is pinned per target token and in order, so a table transposed
    between the step and sample axes cannot reproduce it.
    """
    gold = json.loads((GOLDEN/"gpt2-values.json").read_text())
    assert (gpt2.model_spec.model_id, gpt2.model_spec.revision) == (gold["model_id"], gold["revision"])
    sampling = SamplingSpec(gpt2.model_spec, n=gold["n"], seed=gold["seed"])
    for row in gold["items"]:
        item = Stimulus.from_text(row["context"], row["target"], gpt2)
        assert (list(item.context), list(item.target)) == (row["context_tokens"], row["target_tokens"])
        surprisals = [-float(x) for x in gpt2.score(item.context, item.target)]
        assert surprisals == pytest.approx(row["token_surprisals"], abs=1e-4)
        exact = Measure(MeasureSpec(sampling, len(item.target), "compare_ragged", warping="neglog",
                                    estimator=EstimatorSpec(method="exact")),
                        sampler=Sampler(gpt2)).compute(item)
        assert exact.value == pytest.approx(row["exact_surprisal"], abs=1e-4)
        if "mc_surprisal" not in row:
            continue
        mc = Measure(MeasureSpec(sampling, 1, "compare_ragged", warping="neglog",
                                 estimator=EstimatorSpec(method="mc", bootstrap=99)),
                     sampler=Sampler(gpt2)).compute(item)
        # The pinned draw is reproducible; the tolerance is its own reported error, so a
        # last-bit difference in the model's softmax cannot flip one sample into a failure.
        assert mc.se == pytest.approx(row["mc_se"], rel=.2)
        assert abs(mc.value-row["mc_surprisal"]) < 3*row["mc_se"]

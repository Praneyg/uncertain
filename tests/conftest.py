import pytest
import os
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from uncertain import ToyBackend, Sampler, SamplingSpec, MeasureSpec, Measure, EstimatorSpec


@pytest.fixture
def tmp_path():
    # Windows sandbox workers cannot traverse Python's owner-only mode=0o700 dirs.
    # These fixtures contain synthetic, non-sensitive data only.
    root = Path(os.environ.get("UNCERTAIN_TEST_TMP", ".test-artifacts"))
    path = root / uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def backend():
    return ToyBackend()


@pytest.fixture
def sampling(backend):
    return SamplingSpec(backend.model_spec, n=384, seed=31)


@pytest.fixture
def make_measure(backend, sampling):
    def make(**kwargs):
        defaults = dict(sampling=sampling, horizon=1, short_samples="compare_ragged",
                        estimator=EstimatorSpec(method="mc", bootstrap=39))
        defaults.update(kwargs)
        return Measure(MeasureSpec(**defaults), sampler=Sampler(backend))
    return make


GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


@pytest.fixture(scope="session")
def gpt2():
    """Pretrained GPT-2 from the local Hugging Face cache, pinned to one revision."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from uncertain import HFBackend
    backend = HFBackend("openai-community/gpt2", revision=GPT2_REVISION,
                        bos_token_id=50256, eos_token_id=50256)
    inner = backend.probabilities

    @lru_cache(maxsize=None)
    def cached(context):
        # p(.|context) is a deterministic function of the context, and an H=1 Monte Carlo
        # draw asks for the same one n times. Memoising changes no value; it is what makes
        # n=512 against a real model affordable on CPU.
        p = inner(context)
        p.flags.writeable = False
        return p
    backend.probabilities = cached  # type: ignore[method-assign]
    return backend

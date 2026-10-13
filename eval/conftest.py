"""Shared fixtures. Every test is tagged @pytest.mark.req("ID", ...) so grade.py can map
results back to eval/requirements.md; measured numbers go through `measure(...)` into the
JUnit XML as properties."""
import os
import sys
from pathlib import Path
from uuid import uuid4
import pytest

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))

import uncertain  # noqa: E402
from uncertain import Measure, MeasureSpec, SamplingSpec, Sampler, EstimatorSpec  # noqa: E402

# mutate.py points the suite at a mutated copy of src/; refuse to run against anything else.
if os.environ.get("EVAL_SRC"):
    assert Path(uncertain.__file__).resolve().is_relative_to(Path(os.environ["EVAL_SRC"]).resolve()),         f"expected uncertain from {os.environ['EVAL_SRC']}, got {uncertain.__file__}"

GPT2_ID = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def pytest_configure(config):
    config.addinivalue_line("markers", "req(*ids): requirement IDs from eval/requirements.md")
    config.addinivalue_line("markers", "gpt2: needs the cached pretrained GPT-2")


@pytest.fixture(autouse=True)
def _tag_requirements(request, record_property):
    marker = request.node.get_closest_marker("req")
    record_property("req", ",".join(marker.args) if marker else "")


@pytest.fixture
def measure(record_property):
    """measure(name, value): attach a measured number to this test's JUnit record."""
    def _record(name, value):
        record_property(f"m:{name}", repr(value))
    return _record


@pytest.fixture
def scratch():
    # Python's mkdtemp uses 0o700 dirs, which Windows sandbox workers cannot traverse
    # (same workaround as tests/conftest.py).
    root = Path(os.environ.get("UNCERTAIN_EVAL_TMP", EVAL / ".tmp"))
    path = root / uuid4().hex
    path.mkdir(parents=True)
    return path


def make_measure(lm, *, n=256, seed=0, horizon=1, short_samples="compare_ragged",
                 method="mc", bootstrap=99, sampler=None, store=None, generation=None,
                 token_budget=32, **spec):
    sampling = SamplingSpec(lm.model_spec, n=n, seed=seed, generation=generation or {},
                            token_budget=token_budget)
    ms = MeasureSpec(sampling, horizon, short_samples,
                     estimator=EstimatorSpec(method=method, bootstrap=bootstrap), **spec)
    return Measure(ms, sampler=sampler or Sampler(lm), store=store)


@pytest.fixture(scope="session")
def tiny_gpt2():
    """Randomly initialised 3-layer GPT-2 built locally: no download, real HF code path."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from uncertain import HFBackend
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=40, n_positions=64, n_embd=16, n_layer=3,
                                     n_head=2, bos_token_id=0, eos_token_id=39)
    model = transformers.GPT2LMHeadModel(config).eval()
    return HFBackend(model=model, bos_token_id=0, eos_token_id=39)


@pytest.fixture(scope="session")
def gpt2():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pytest.importorskip("transformers")
    from uncertain import HFBackend
    backend = HFBackend(GPT2_ID, revision=GPT2_REVISION, bos_token_id=50256, eos_token_id=50256)
    try:
        backend._load()
        backend._load_tokenizer()
    except OSError as exc:  # not in the local cache
        pytest.skip(f"GPT-2 not cached: {exc}")
    return backend

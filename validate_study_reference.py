"""Compare study estimators to original helper functions on identical alternatives.

Uses a tiny seeded GPT-2 model with no downloads, isolating estimator semantics
from random generation, hardware, or changed model checkpoints.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel

from uncertain import AlternativeSet, HFBackend, Sequence, UnitSpec
from uncertain.emnlp2024 import GPT2StudyEncoder, estimate_study
from uncertain.store import atomic_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=Path(
        r'C:\Users\praney\OneDrive\Desktop\generalized-surprisal'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent /
                        'replication/reference_validation.json')
    args = parser.parse_args()
    source = args.reference_root/'src/measures.py'
    sys.path.insert(0, str(source.parent))
    module_spec = importlib.util.spec_from_file_location('original_study_measures', source)
    original = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(original)
    torch.set_num_threads(1)
    torch.manual_seed(778)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=32, n_embd=16,
                                     n_layer=2, n_head=2, bos_token_id=0, eos_token_id=6,
                                     attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.)).eval()
    backend = HFBackend(model=model, bos_token_id=0, eos_token_id=6)
    encoder = GPT2StudyEncoder(backend)
    scorer = original.SamplingScorer.__new__(original.SamplingScorer)
    scorer.model = model
    scorer.tokenizer = SimpleNamespace(eos_token_id=6)
    scorer.device = 'cpu'
    scorer.embed_size = 16
    scorer.bsize_dist = 2
    padded = torch.tensor([[1,2,3], [2,3,4], [1,2,6], [4,6,6], [1,2,3]])
    rows = []
    for i, raw in enumerate(padded.tolist()):
        ids = raw[:raw.index(6)+1] if 6 in raw else raw
        rows.append(Sequence(tuple(ids), (-1.,)*len(ids), (-1.,)*len(ids),
                             tuple(str(t) for t in ids), tuple((j,j+1) for j in range(len(ids))),
                             stop_reason='eos' if 6 in ids else 'horizon', sample_id=i))
    a = AlternativeSet(tuple(rows), (0,1), UnitSpec(), 3)
    ours = estimate_study(a, 2, encoder, bootstrap=30)
    iv, expected_token = scorer.compute_token_distances(
        list(padded[:, 0]), torch.tensor([0,1]), torch.tensor(2), -1, -1)
    expected_seq = scorer.compute_expected_sequence_distances(
        list(padded), torch.tensor([0,1]), -1, -1)
    observed = {'decontextualised': float(iv[0]),
                'expected_decontextualised': float(expected_token[0]),
                'expected_seq_decontextualised': float(expected_seq[0])}
    report = {'scope': 'Original point-estimator helpers; same model and same draws',
              'includes': ['duplicate draws', 'later EOS', 'masked padding', 'multi-batch encoding'],
              'excludes': ['historical RNG replay', 'legacy bootstrap variance',
                           'legacy post-EOS padding log probabilities'],
              'reference_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'transformers_version': transformers.__version__,
              'model_seed': 778, 'comparisons': {}}
    for name, expected in observed.items():
        difference = abs(ours[name].value-expected)
        report['comparisons'][name] = {'library': ours[name].value, 'original': expected,
                                        'absolute_difference': difference, 'passed': difference < 2e-6}
        np.testing.assert_allclose(ours[name].value, expected, atol=2e-6, rtol=0)
    report['passed'] = True
    atomic_write(args.output, json.dumps(report, indent=2).encode('utf-8'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

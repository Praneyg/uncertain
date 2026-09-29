# GPT-2 results on the aligned dataset

`scores.csv` retains every source column and appends columns starting with
`uncertain_`. `manifest.json` records the source checksum, pinned model revision,
configuration, row counts, timing, and comparison with the existing `s_GPT2` column.

- Input: `C:\Users\praney\OneDrive\Desktop\generalized-surprisal\data\psyling\aligned.csv`
- Model: `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Method: exact model probability of the observed target token sequence given its context.
- Surprisal is supplied in both nats and bits. Probability is stored separately.
- All rows are retained, including sentence-boundary and multi-token targets.
- Multi-token scores describe the joint token-sequence event. They do not marginalize
  over all possible tokenizations or continuations defining a complete word.
- The comparison in the manifest uses single-token, non-boundary rows only.
- Literal text such as `None` is preserved. The input data is not modified.

This run computes exact surprisal and contextual probability. It does not compute
the sampling-based information-value measures or reproduce the original paper's grid.
The other CSVs in the input directory contain sentence text or existing estimates;
they are not separately scored in this run.

Run again in PowerShell (uses the already cached model, without network access):

```powershell
$env:PYTHONPATH = 'C:\Users\praney\OneDrive\Desktop\uncertain\src'
& 'C:\Users\praney\OneDrive\Desktop\uncertain\.venv\Scripts\python.exe' `
  'C:\Users\praney\OneDrive\Desktop\uncertain\run_aligned_dataset.py'
```

Use `--limit 10 --output-dir PATH` for a short run in another directory. The default
command overwrites the generated scores and manifest in this output directory.

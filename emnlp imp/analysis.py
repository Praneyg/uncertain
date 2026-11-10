"""Join new estimates to the Aligned dataset, rerun the paper's cross-validation on both the
original and the new estimates, and compare. Ports apply_estimates_to_dataset.py and
crossvalidation.ipynb (cells 9, 11, 13); significance as in analysis.ipynb cell 36.

    python "emnlp imp/analysis.py" [--seeds 100]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
ORIGINAL = Path(r"C:\Users\praney\OneDrive\Desktop\generalized-surprisal")
RESULTS = HERE / "results"
LMS = {"gpt2-small": "gpt2_small", "neo-125m": "neo_125m"}
SUFFIX = "512_5_1_0"  # N=512, H=5, T=1.0, as in the original column names

EXACT = ["prob", "surprisal", "expected_prob", "entropy"]
SAMPLING = ["decontextualised", "expected_decontextualised", "expected_seq_decontextualised",
            "sequence_entropy"]
ANTICIPATORY = ["expected_prob", "expected_decontextualised", "expected_seq_decontextualised",
                "sequence_entropy"]
RESPONSIVE = ["prob", "decontextualised"]
RT = ["RTfirstfix", "RTfirstpass", "RTrightbound", "self_paced_reading_time"]
YS = ["rating_mean", "cloze_p_smoothed"] + RT + ["N400", "P600", "ELAN", "LAN", "EPNP", "PNP", "entropy"]
BASE = ["Subtlex_log10", "context_length", "length"]
BASE_RT = BASE + ["Subtlex_log10_prev", "length_prev", "Subtlex_log10_prev_prev", "length_prev_prev"]
TS = ["", "_prev", "_prev_prev"]


def name(metric, lm):
    return f"{metric}_{lm}" if metric in EXACT else f"{metric}_{lm}_{SUFFIX}"


def join():
    """apply_estimates_to_dataset.py: filter, baseline lags, estimates at t, t-1, t-2."""
    aligned = pd.read_csv(ORIGINAL / "data/psyling/aligned.csv", keep_default_na=False, na_values=[""])
    aligned = aligned[~aligned.word.astype(str).str.contains(r"[.,']")].reset_index(drop=True)
    key = aligned.set_index(["sent_id", "context_length"])
    for lag, suffix in ((1, "_prev"), (2, "_prev_prev")):
        idx = list(zip(aligned.sent_id, aligned.context_length - lag))
        for col in ("Subtlex_log10", "length"):
            aligned[col + suffix] = [key[col].get(i, 0.) for i in idx]
    for label, lm in LMS.items():
        words = pd.read_csv(RESULTS / label / "word_estimates.csv")
        for metric in EXACT + SAMPLING:
            scores = {s: g.sort_values("word_index")[metric].tolist() for s, g in words.groupby("sent_id")}
            for lag, suffix in zip((0, 1, 2), TS):
                vals = []
                for s, pos in zip(aligned.sent_id, aligned.context_length):
                    try:  # the original indexes a Python list: pos-lag == -1 wraps to the last word
                        vals.append(scores[s][pos - lag])
                    except IndexError:
                        vals.append(0.)
                aligned[name(metric, lm) + suffix] = vals
    aligned.to_csv(RESULTS / "aligned_with_estimates_uncertain.csv", index=False)
    return aligned


def ols_r2(train, test, y, xs):
    """Test R^2 exactly as smf.ols + compute_rsquared: training drops rows with any NaN;
    pandas sums skip NaN, so RSS skips rows with NaN predictions while TSS keeps every
    non-NaN y (Subtlex_log10 has 21 NaNs)."""
    X = np.column_stack([np.ones(len(train[y]))] + [train[x] for x in xs])
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(train[y])
    beta = np.linalg.lstsq(X[ok], train[y][ok], rcond=None)[0]
    pred = np.column_stack([np.ones(len(test[y]))] + [test[x] for x in xs]) @ beta
    yt = test[y]
    return 1 - np.nansum((yt - pred) ** 2) / np.nansum((yt - np.nanmean(yt)) ** 2)


def designs(y, lm):
    """(analysis, metric, baseline columns, target columns), per crossvalidation.ipynb."""
    ts = TS if y in RT else [""]
    base = BASE_RT if y in RT else BASE
    col = lambda m: [name(m, lm) + t for t in ts]
    out = [("control", m, base, base + col(m)) for m in EXACT + SAMPLING]
    out += [("surprisal", m, base + col("surprisal"), base + col("surprisal") + col(m))
            for m in ["entropy"] + ANTICIPATORY]
    se = base + col("entropy") + col("surprisal")
    out += [("surprisal_entropy", m, se, base + col(m) + col("surprisal")) for m in ANTICIPATORY]
    out += [("surprisal_entropy", m, se, base + col("entropy") + col(m)) for m in RESPONSIVE]
    return out


def cross_validate(data, lm, seeds):
    metric_cols = [name(m, lm) + t for m in EXACT + SAMPLING for t in TS]
    d = data.copy()
    d[metric_cols] = MinMaxScaler().fit_transform(d[metric_cols])
    arrays = {c: d[c].to_numpy(float) for c in set(metric_cols + BASE_RT + YS)}
    rows = []
    for y in YS:
        ds = designs(y, lm)
        for seed in range(seeds):
            for fold, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=seed).split(d)):
                train = {c: v[tr] for c, v in arrays.items()}
                test = {c: v[te] for c, v in arrays.items()}
                cache = {}
                for analysis, m, b, t in ds:
                    if tuple(b) not in cache:
                        cache[tuple(b)] = ols_r2(train, test, y, b)
                    r2 = ols_r2(train, test, y, t)
                    rows.append((analysis, y, m, f"{seed}_{fold}", r2, r2 - cache[tuple(b)]))
    return pd.DataFrame(rows, columns=["analysis", "y", "metric", "fold", "rsquared_test",
                                       "delta_rsquared_test"])


def summarize(cv):
    rng = np.random.default_rng(0)
    out = []
    for (analysis, y, m), g in cv.groupby(["analysis", "y", "metric"], sort=False):
        delta = g.delta_rsquared_test.to_numpy()
        target = g.rsquared_test.to_numpy()
        boots = rng.choice(delta, (1000, len(delta))).mean(axis=1)  # seaborn errorbar=("ci", 95)
        p = stats.permutation_test((target, target - delta), lambda a, b, axis: a.mean(axis) - b.mean(axis),
                                   vectorized=True, n_resamples=10000, alternative="greater",
                                   permutation_type="samples", random_state=0).pvalue
        out.append((analysis, y, m, delta.mean(), *np.quantile(boots, [.025, .975]), p))
    return pd.DataFrame(out, columns=["analysis", "y", "metric", "delta_r2", "ci_low", "ci_high", "p"])


def check_ols(data, lm):
    import statsmodels.formula.api as smf
    tr, te = next(KFold(10, shuffle=True, random_state=0).split(data))
    xs = BASE_RT + [name("surprisal", lm) + t for t in TS]
    train, test = data.iloc[tr], data.iloc[te]
    fit = smf.ols(f"RTfirstpass ~ {' + '.join(xs)}", data=train).fit()
    resid = test.RTfirstpass - fit.predict(test)
    ref = 1 - np.sum(resid ** 2) / np.sum((test.RTfirstpass - np.mean(test.RTfirstpass)) ** 2)
    ours = ols_r2({c: train[c].to_numpy(float) for c in xs + ["RTfirstpass"]},
                  {c: test[c].to_numpy(float) for c in xs + ["RTfirstpass"]}, "RTfirstpass", xs)
    assert np.isclose(ref, ours, atol=1e-10), (ref, ours)


def agreement(orig, new):
    rows = []
    for label, lm in LMS.items():
        for m in EXACT + SAMPLING:
            a, b = orig[name(m, lm)].to_numpy(float), new[name(m, lm)].to_numpy(float)
            rows.append((label, m, stats.pearsonr(a, b)[0], stats.spearmanr(a, b)[0],
                         np.mean(np.abs(a - b)), a.mean(), b.mean()))
    return pd.DataFrame(rows, columns=["model", "metric", "pearson", "spearman", "mae",
                                       "mean_original", "mean_uncertain"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    new = join()
    orig = pd.read_csv(ORIGINAL / "data/psyling/aligned_with_estimates_v2.csv")
    assert (orig[["sent_id", "context_length"]].to_numpy() == new[["sent_id", "context_length"]].to_numpy()).all()
    agree = agreement(orig, new)
    agree.to_csv(RESULTS / "estimate_agreement.csv", index=False)
    print(agree.round(4).to_string(index=False))
    summaries = []
    for source, data in (("original", orig), ("uncertain", new)):
        for label, lm in LMS.items():
            check_ols(data, lm)
            cv = cross_validate(data, lm, args.seeds)
            cv.to_csv(RESULTS / f"cv_{source}_{label}.csv", index=False)
            summaries.append(summarize(cv).assign(source=source, model=label))
            print(f"cv done: {source} {label}", flush=True)
    summary = pd.concat(summaries)
    summary.to_csv(RESULTS / "cv_summary.csv", index=False)
    wide = summary.pivot_table(index=["model", "analysis", "y", "metric"], columns="source",
                               values=["delta_r2", "p"], sort=False)
    wide.columns = [f"{v}_{s}" for v, s in wide.columns]
    wide = wide.reset_index()
    wide["significant_agree"] = (wide.p_original < .05) == (wide.p_uncertain < .05)
    wide.to_csv(RESULTS / "cv_comparison.csv", index=False)
    for (model, analysis), g in wide.groupby(["model", "analysis"], sort=False):
        r = stats.pearsonr(g.delta_r2_original, g.delta_r2_uncertain)[0]
        print(f"{model} {analysis}: cells={len(g)} significance agreement={g.significant_agree.mean():.2%} "
              f"r(deltaR2)={r:.4f}")


if __name__ == "__main__":
    main()

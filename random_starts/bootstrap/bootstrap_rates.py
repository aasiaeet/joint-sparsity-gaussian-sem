"""Sample-level uncertainty of single-start success rates in the random-start study, and
best of K versus any success among the first K starts (Supplementary Section S18).

Success: the final support thresholded at delta is a minimum-edge support (column is_emin).

Usage:
  python3 bootstrap_rates.py original    # outcomes at the 20,000-cycle cap (../results/runs.csv.gz)
  python3 bootstrap_rates.py continued   # outcomes after continuing the capped runs to 400,000
                                         # cycles (run ../continuation/analyze_continuation.py first)

Cluster bootstrap: for each cell, 10,000 resamples of the 100 samples with replacement; the
rate of a resample is (successes in the resampled samples) / (runs in the resampled samples).
Seed: numpy default_rng(SeedSequence([20261022, 2, cell_index])) with
cell_index = 3 * design_index + n_index (G1, G2, G3; 250, 1000, 4000).  The same seeds are used
for both outcome sets, so the two bootstraps resample the same samples.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb

HERE = Path(__file__).resolve().parent
STUDY_RESULTS = HERE.parent / "results"
CONTINUED = HERE.parent / "continuation" / "results" / "runs_with_continuation.csv.gz"
OUT = HERE / "results"
DESIGNS, NS, KS = ["G1", "G2", "G3"], [250, 1000, 4000], [1, 2, 5, 10, 20]
B_RESAMPLES = 10000
SEED = 20261022


def wilson(k, n, z=1.959963984540054):
    # as global_selection.wilson
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load(which):
    if which == "original":
        r = pd.read_csv(STUDY_RESULTS / "runs.csv.gz",
                        usecols=["design", "n", "dataset", "start_type", "start", "objective", "is_emin"])
    elif which == "continued":
        r = pd.read_csv(CONTINUED,
                        usecols=["design", "n", "dataset", "start_type", "start", "new_objective", "new_is_emin"])
        r = r.rename(columns={"new_objective": "objective", "new_is_emin": "is_emin"})
    else:
        raise SystemExit("argument must be 'original' or 'continued'")
    r = r[r.start_type == "random"].copy()
    r["is_emin"] = r["is_emin"].astype(bool)
    return r


def main():
    which = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    r = load(which)
    rows, krows = [], []
    for di, d in enumerate(DESIGNS):
        for ni, n in enumerate(NS):
            cell = r[(r.design == d) & (r.n == n)].sort_values(["dataset", "start"])
            ds = np.sort(cell.dataset.unique())
            assert len(ds) == 100 and len(cell) == 2000, (d, n, len(ds), len(cell))
            y = cell.groupby("dataset").is_emin.sum().reindex(ds).to_numpy().astype(float)
            m = cell.groupby("dataset").size().reindex(ds).to_numpy().astype(float)
            assert np.all(m == 20)
            k, N = int(y.sum()), int(m.sum())
            a = k / N
            lo, hi = wilson(k, N)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, 3 * di + ni]))
            idx = rng.integers(0, len(ds), size=(B_RESAMPLES, len(ds)))
            boot = y[idx].sum(axis=1) / m[idx].sum(axis=1)
            p_i = y / m
            rows.append(dict(
                outcomes=which, design=d, n=n, datasets=len(ds), random_runs=N, successes=k, rate=a,
                wilson_lo=lo, wilson_hi=hi, binomial_se=math.sqrt(a * (1 - a) / N),
                dataset_level_se_analytic=float(p_i.std(ddof=1) / math.sqrt(len(ds))),
                cluster_boot_se=float(boot.std(ddof=1)),
                cluster_boot_lo=float(np.quantile(boot, 0.025)),
                cluster_boot_hi=float(np.quantile(boot, 0.975)),
                design_effect=float((boot.std(ddof=1) / math.sqrt(a * (1 - a) / N)) ** 2) if 0 < a < 1 else np.nan,
                datasets_with_any_success=int((y > 0).sum()),
                max_successes_in_one_dataset=int(y.max()),
                bootstrap_resamples=B_RESAMPLES, bootstrap_seed=f"SeedSequence([{SEED}, 2, {3 * di + ni}])"))
            by = {g: sub.sort_values("start") for g, sub in cell.groupby("dataset")}
            for K in KS:
                any_first, best_obj, order_avg = 0, 0, 0.0
                for g in ds:
                    first = by[g][by[g].start <= K]
                    assert len(first) == K
                    any_first += bool(first.is_emin.any())
                    best = first.sort_values(["objective", "start"], kind="stable").iloc[0]
                    best_obj += bool(best.is_emin)
                    yi = int(by[g].is_emin.sum())
                    # probability that a uniformly random K-subset of the 20 starts contains a success
                    order_avg += 1 - comb(20 - yi, K, exact=True) / comb(20, K, exact=True)
                krows.append(dict(outcomes=which, design=d, n=n, K=K,
                                  any_success_first_K=any_first / len(ds),
                                  any_success_avg_over_start_subsets=order_avg / len(ds),
                                  best_of_K_by_objective_emin=best_obj / len(ds),
                                  pooled_benchmark=1 - (1 - a) ** K,
                                  single_start_rate=a))
    out = pd.DataFrame(rows)
    kout = pd.DataFrame(krows)
    out.to_csv(out_dir / f"single_start_intervals_{which}.csv", index=False)
    kout.to_csv(out_dir / f"any_vs_bestofK_{which}.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["design", "n", "successes", "rate", "wilson_lo", "wilson_hi", "binomial_se",
               "dataset_level_se_analytic", "cluster_boot_se", "cluster_boot_lo", "cluster_boot_hi",
               "design_effect"]].round(4).to_string())
    piv = kout.pivot_table(index=["design", "n"], columns="K",
                           values=["any_success_first_K", "best_of_K_by_objective_emin", "pooled_benchmark"])
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()

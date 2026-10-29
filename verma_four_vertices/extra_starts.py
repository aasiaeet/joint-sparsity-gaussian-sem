"""Extra-start refits of the non-exact supports with at most 5 edges and the smallest gaps.

For the 40 non-exact supports with at most e* + 1 = 5 edges and the smallest
high-precision gaps (support_table.csv), rerun
 - the residual fit from 600 starts (200 each in [-2, 2], [-10, 10], [-100, 100]),
 - the KL fit with 100 random starts in [-3, 3] (plus the usual zero and regression starts),
both with |B| <= 1000, seed SeedSequence([20261022, 6, support_index, 6]).
Reads results/support_table.csv from analyze.py; writes results/extra_starts.csv.
Usage: python3 extra_starts.py [--workers W]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mpc
import time

import numpy as np
import pandas as pd

import fit_supports as fs
from common import MASTER_SEED, PAIRS, RESULTS, SUPPORTS, STUDY_ID
from high_precision_recheck import gap_hp

N_SUPPORTS = 40


def refit(idx):
    t0 = time.time()
    s = SUPPORTS[idx]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, idx, 6]))
    D, U = s.D, s.U
    req = [pr for pr in PAIRS if pr not in set(U)]
    nd = len(D)
    starts = [rng.uniform(-w, w, size=nd) for w in (2.0, 10.0, 100.0) for _ in range(200)]
    best, _, n_run = fs._ls_multistart(D, req, starts, fs.CONFIG["coefficient_bound"], True, fs.CONFIG["resid"])
    old = fs.CONFIG["kl"]["random_starts"], fs.CONFIG["kl"]["start_box"]
    fs.CONFIG["kl"]["random_starts"], fs.CONFIG["kl"]["start_box"] = 100, [-3.0, 3.0]
    try:
        kl = fs.fit_kl(s, rng=rng)
    finally:
        fs.CONFIG["kl"]["random_starts"], fs.CONFIG["kl"]["start_box"] = old
    g = gap_hp(kl["B"], kl["O"])
    return dict(index=idx, name=s.name, edges=s.count, resid_600=best[0], resid_600_maxabsB=float(np.max(np.abs(best[1]))),
                gap_hp_102=float(g), kl_maxabsB=float(np.max(np.abs(kl["B"]))), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    t = pd.read_csv(RESULTS / "support_table.csv", low_memory=False)
    t = t[(t.edges <= 5) & (t.gap_hp >= 1e-10)].sort_values("gap_hp").head(N_SUPPORTS)
    with mpc.Pool(args.workers) as pool:
        rows = pool.map(refit, t["index"].tolist(), chunksize=1)
    out = pd.DataFrame(rows).merge(t[["index", "gap_hp", "resid", "separated"]], on="index")
    out["gap_change"] = out.gap_hp_102 - out.gap_hp
    out.to_csv(RESULTS / "extra_starts.csv", index=False)
    print(out.to_string())
    print(json.dumps(dict(min_resid_600=float(out.resid_600.min()), min_gap_hp_102=float(out.gap_hp_102.min()),
                          max_gap_decrease=float(-out.gap_change.min()), wall_cpu_seconds=float(out.seconds.sum())), indent=1))


if __name__ == "__main__":
    main()

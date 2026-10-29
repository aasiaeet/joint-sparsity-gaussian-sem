"""Coefficient-bound scan for the near-limit supports (not exact, 50-digit gap below 1e-6).

Same procedure as high_precision_recheck.bound_scan (KL refits with |B| <= 10, 100, 1000, 10000,
warm-started from the previous bound, 50-digit gap at each optimum), applied to the
non-exact supports whose 50-digit gap is below 1e-6.  A gap that keeps falling as
the bound grows means Sigma* is a limit of the support's model at infinity (in its
closure, not attained in the box).  Reads results/support_table.csv from a first run
of analyze.py; writes results/bound_scan_near_limit.csv.
Usage: python3 near_limit_scan.py [--workers W]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mpc

import pandas as pd

from common import RESULTS
from high_precision_recheck import _init, bound_scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    t = pd.read_csv(RESULTS / "support_table.csv", low_memory=False)
    sel = t[(~t.exact) & (t.gap_hp < 1e-6)]["index"].tolist()
    with mpc.Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(bound_scan, sel, chunksize=1)
    out = pd.DataFrame(rows)
    out = out.merge(t[["index", "gap_hp", "separated", "resid"]], on="index")
    out["ratio_100_10"] = out.gap_hp_M100 / out.gap_hp_M10
    out["ratio_1000_100"] = out.gap_hp_M1000 / out.gap_hp_M100
    out["ratio_10000_1000"] = out.gap_hp_M10000 / out.gap_hp_M1000
    out.to_csv(RESULTS / "bound_scan_near_limit.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["name", "edges", "gap_hp_M10", "gap_hp_M100", "gap_hp_M1000", "gap_hp_M10000", "ratio_10000_1000"]].to_string())
    print(out[["ratio_100_10", "ratio_1000_100", "ratio_10000_1000"]].describe())


if __name__ == "__main__":
    main()

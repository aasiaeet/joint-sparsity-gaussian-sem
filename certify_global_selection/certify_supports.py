"""Certified two-sided bounds on l_min(H; S) for all 200 supports of each subset sample.

Lower bounds from the closed-form profile over Omega without the spectral constraint
(support_bounds.py), upper bounds from the stored fit points evaluated in interval
arithmetic.  A fit is certified when both bounds are within 1e-8 of the reported value.
Supports that fail here are passed to certify_supports_spectral.py.

Usage: python3 certify_supports.py [--workers W] [--only k1,k2,...] [--out DIR]
Reads support_refits.npz; writes support_bounds.csv.gz (one row per sample and support)
and support_bounds_samples.csv (one row per sample).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "global_selection"))
import global_selection as gs  # noqa: E402
from support_bounds import TOL_CERT, SampleIA, certified_upper, profile_value, support_lower  # noqa: E402

RESULTS = HERE / "results"
Z = None
REFITS = RESULTS / "support_refits.npz"


def load():
    global Z
    if Z is None:
        Z = dict(np.load(REFITS))
    return Z


def certify_sample(k):
    z = load()
    t0 = time.monotonic()
    S = z["S"][k]
    sia = SampleIA(S)
    fsat_lo, fsat_hi = float(sia.fsat.lo), float(sia.fsat.hi)
    rows = []
    for s in gs.SUPPORTS:
        ts = time.monotonic()
        found = float(z["npz_value"][k][s.index])
        B = z["B"][k][s.index]
        O = z["O"][k][s.index]
        ub, shrink = certified_upper(sia, s, B, O)
        saturated = found <= fsat_hi + 1e-12
        lb_struct, info = support_lower(sia, s, skip_rho2=saturated)
        lb = max(fsat_lo, lb_struct)
        gap = found - lb
        ubgap = ub - found
        cert = (gap <= TOL_CERT) and (ubgap <= TOL_CERT)
        rows.append(dict(sample=k, design=str(z["design"][k]), n=int(z["n"][k]), rep=int(z["rep"][k]),
                         support_index=s.index, support=s.name, count=s.count, bowed=s.bowed,
                         nD=len(s.D), nU=len(s.U), fit_method=str(z["method"][k][s.index]),
                         found=found, lb=lb, ub=ub, lb_saturated=fsat_lo, lb_structured=lb_struct,
                         lb_source="saturated" if fsat_lo >= lb_struct else "structured",
                         found_minus_lb=gap, ub_minus_found=ubgap, ub_shrink=shrink,
                         certified=cert, form=info.get("form"),
                         boxes=int(info.get("gram_boxes", 0) + info.get("rho2_boxes", 0)),
                         bnb_converged=bool(info.get("gram_converged", True) and info.get("rho2_converged", True)),
                         runtime_s=time.monotonic() - ts))
    return rows, time.monotonic() - t0


def main():
    global REFITS
    ap = argparse.ArgumentParser(description="Certify the support fits of the subset samples.")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only", type=str, default=None, help="comma-separated sample indices")
    ap.add_argument("--refits", type=str, default=str(REFITS))
    ap.add_argument("--out", type=str, default=str(RESULTS))
    a = ap.parse_args()
    REFITS = Path(a.refits)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = load()
    ks = list(range(len(z["S"]))) if not a.only else [int(x) for x in a.only.split(",")]
    import multiprocessing as mp
    t0 = time.monotonic()
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(certify_sample, ks, chunksize=1)
    else:
        res = [certify_sample(k) for k in ks]
    rows = [r for rr, _ in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(out / "support_bounds.csv.gz", index=False)
    srows = []
    for k, (rr, rt) in zip(ks, res):
        d = pd.DataFrame(rr)
        srows.append(dict(sample=k, design=d.design[0], n=d.n[0], rep=d.rep[0],
                          supports_certified=int(d.certified.sum()), supports=len(d),
                          max_found_minus_lb=float(d.found_minus_lb.max()),
                          max_ub_minus_found=float(d.ub_minus_found.max()),
                          runtime_s=rt))
    sd = pd.DataFrame(srows)
    sd.to_csv(out / "support_bounds_samples.csv", index=False)
    print(sd.to_string())
    print("wall", time.monotonic() - t0)
    bad = df[~df.certified]
    print("uncertified:", len(bad))
    if len(bad):
        print(bad[["sample", "design", "n", "rep", "support", "fit_method", "found", "lb", "ub",
                   "found_minus_lb", "ub_minus_found", "form", "boxes", "bnb_converged"]].to_string())


if __name__ == "__main__":
    main()

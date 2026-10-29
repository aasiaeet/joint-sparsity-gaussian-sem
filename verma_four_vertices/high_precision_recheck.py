"""50-digit re-evaluation of the population fits and the coefficient-bound scan (Supplement S18).

1. gap_hp: the KL gap at every stored KL optimum (results/fits_params.npz),
   recomputed with mpmath (50 digits) from the float parameters (exactly
   representable) and Sigma* built from the decimal generating parameters.
   Float gaps at |B| near the bound 1000 lose precision and can be negative.
2. Residual points: for every support, the KL gap at the residual fit's best
   point B with Omega = R(B) restricted to the pattern (when positive definite),
   also in 50 digits.  The residual fit is rerun with its recorded seed (it is
   deterministic), so this is the same point as in fits.csv.
3. Limit-at-infinity scan for every non-exact support with at most 5 edges whose
   KL optimum sits at the coefficient bound: refit with bounds 10, 100, 1000, 10000
   (warm start from the previous bound's optimum, plus the regular starts) and
   record the high-precision gap at each bound.  If gap(10^4) is close to
   gap(10^3), the infimum over the unbounded domain is approached at infinity but
   stays positive.

Reads results/fits.csv and results/fits_params.npz; writes results/hp_recheck.csv,
results/bound_scan.csv and results/hp_recheck_meta.json.
Usage: python3 high_precision_recheck.py [--workers W]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import multiprocessing as mpc
import time

import mpmath as mp
import numpy as np
import pandas as pd

import fit_supports as fs
from common import P, RESULTS, SUPPORTS

mp.mp.dps = 50


def _sigma_hp():
    rec = fs.param_record()
    B = mp.zeros(P)
    B[0, 1] = mp.mpf(repr(rec["B"]["1->2"]))
    B[1, 2] = mp.mpf(repr(rec["B"]["2->3"]))
    B[2, 3] = mp.mpf(repr(rec["B"]["3->4"]))
    O = mp.diag([mp.mpf(repr(x)) for x in rec["Omega_diag"]])
    O[1, 3] = O[3, 1] = mp.mpf(repr(rec["Omega_offdiag"]["2<->4"]))
    C = mp.inverse(mp.eye(P) - B)
    return C.T * O * C


SIG_HP = _sigma_hp()
LSAT_HP = mp.log(mp.det(SIG_HP)) / 2 + mp.mpf(P) / 2


def gap_hp(B, O):
    """KL gap at (B, O) in 50 digits; None if O is not positive definite."""
    Bm = mp.matrix([[mp.mpf(float(B[i, j])) for j in range(P)] for i in range(P)])
    Om = mp.matrix([[mp.mpf(float(O[i, j])) for j in range(P)] for i in range(P)])
    Om = (Om + Om.T) / 2  # the float A^T S A is symmetric only up to rounding
    A = mp.eye(P) - Bm
    R = A.T * SIG_HP * A
    try:
        L = mp.cholesky(Om)
    except (ValueError, ZeroDivisionError):
        return None
    ld = 2 * sum(mp.log(L[i, i]) for i in range(P))
    val = (ld + sum((mp.inverse(Om) * R)[i, i] for i in range(P))) / 2
    return val - LSAT_HP


def masked_R(B, s):
    A = np.eye(P) - B
    R = A.T @ fs.SIGMA @ A
    O = np.diag(np.diag(R))
    for (i, j) in s.U:
        O[i, j] = O[j, i] = R[i, j]
    return O


def recheck(idx):
    s = SUPPORTS[idx]
    Bk, Ok = PARAMS["B"][idx], PARAMS["O"][idx]
    g1 = gap_hp(Bk, Ok)
    rs = fs.fit_resid(s, diagnostic=False)
    Br = rs["B"]
    Or = masked_R(Br, s)
    g2 = gap_hp(Br, Or) if np.linalg.eigvalsh(Or)[0] > 0 else None
    return dict(index=idx, gap_hp_klpoint=float(g1) if g1 is not None else None,
                gap_hp_residpoint=float(g2) if g2 is not None else None,
                resid_rerun=rs["resid"])


def _kl_with_bound(s, M, warm):
    old = fs.CONFIG["coefficient_bound"]
    fs.CONFIG["coefficient_bound"] = M
    try:
        out = fs.fit_kl(s, warm=warm)
    finally:
        fs.CONFIG["coefficient_bound"] = old
    return out


def bound_scan(idx):
    s = SUPPORTS[idx]
    row = dict(index=idx, name=s.name, edges=s.count)
    warm = []
    for M in [10.0, 100.0, 1000.0, 10000.0]:
        if warm:
            Bw, Ow = warm[0]
            # stretch the previous optimum toward the new bound along its direction
            warm = warm + [(np.clip(Bw * 3.0, -M, M), None)]
        kl = _kl_with_bound(s, M, warm)
        g = gap_hp(kl["B"], kl["O"])
        row[f"gap_hp_M{int(M)}"] = float(g) if g is not None else None
        row[f"maxabsB_M{int(M)}"] = float(np.max(np.abs(kl["B"])))
        warm = [(kl["B"], kl["O"])]
    return row


def _init():
    global PARAMS
    z = np.load(RESULTS / "fits_params.npz")
    PARAMS = dict(B=z["B"], O=z["O"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    t0 = time.time()
    fits = pd.read_csv(RESULTS / "fits.csv")
    with mpc.Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(recheck, range(len(SUPPORTS)), chunksize=32)
    t1 = time.time()
    with open(RESULTS / "hp_recheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    exact = (fits.exact_kl == 1) & (fits.exact_resid == 1)
    sel = fits[(fits.edges <= 5) & (~exact) & (fits.kl_maxabsB > 999)]["index"].tolist()
    with mpc.Pool(args.workers, initializer=_init) as pool:
        scan = pool.map(bound_scan, sel, chunksize=2)
    t2 = time.time()
    if scan:
        with open(RESULTS / "bound_scan.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(scan[0].keys()))
            w.writeheader()
            w.writerows(scan)
    meta = dict(recheck_seconds=t1 - t0, scan_seconds=t2 - t1, n_scanned=len(sel), workers=args.workers,
                dps=mp.mp.dps)
    (RESULTS / "hp_recheck_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

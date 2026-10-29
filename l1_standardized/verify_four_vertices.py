"""Verification of the four-vertex candidates with margin at least 1e-3 (Supplement S13).

For every candidate:
  (1) exact recomputation of the representation outside the class found by S1: with Sigma
      taken as the exact binary value of the stored floats (symmetrized) and B as stored,
      Omega = (I-B)^T Sigma (I-B) and Phi(B, Omega) are recomputed in 60-digit arithmetic
      (mpmath); we report max |Phi(B, Omega) - Sigma| in double and in 60 digits, the
      smallest eigenvalue of Omega, the actual support (entries above 1e-9), the largest
      entry below that threshold, the smallest entry above it, R_1, and the invariant of the
      actual support;
  (2) solver S2 on the supports of the generating class (all of them for classes of at most
      40 supports, otherwise the maximal ones), with R = S1 class minimum + 0.05 (an in-class
      representation below the S1 class minimum has all |B_ij| below R), to confirm the class
      minimum;
  (3) S2 on the 24 complete supports (global minimum) for the largest-margin candidate of
      each generating graph.

Reads results/p4_search.csv (or its gzipped copy) and writes results/p4_verify.csv; finished
rows are appended to results/p4_verify.csv.partial.jsonl so that a run can be resumed.  The
stored results/p4_verify.csv.gz covers 967 of the 1,614 candidates, including at least one for
each of the 89 generating graphs; running this script without options verifies all of them.
Usage: python3 verify_four_vertices.py [--workers W] [--pilot N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1core import (all_supports, complete_supports, support_name, parse_support, phi, omega_of,
                    actual_support, r1_value)
from stage_solver import s2_support
from search_common import RESULTS, invariant_table, maximal_supports, open_table

P = 4
INV = None
mp.mp.dps = 60


def init():
    global INV
    INV = invariant_table(P)


def exact_check(S, B):
    p = len(S)
    # the stored Sigma is symmetric only up to rounding; its exact symmetric part is the
    # target, and the asymmetry is reported separately
    Sm = mp.matrix([[(mp.mpf(float(S[i, j])) + mp.mpf(float(S[j, i]))) / 2 for j in range(p)] for i in range(p)])
    Bm = mp.matrix([[mp.mpf(float(B[i, j])) for j in range(p)] for i in range(p)])
    W = mp.eye(p) - Bm
    Om = W.T * Sm * W
    Om = (Om + Om.T) / 2
    C = mp.inverse(W)
    Ph = C.T * Om * C
    err_mp = max(abs(Ph[i, j] - Sm[i, j]) for i in range(p) for j in range(p))
    eig = mp.eigsy(Om)[0]
    min_eig = min(eig[i] for i in range(p))
    O = np.array([[float(Om[i, j]) for j in range(p)] for i in range(p)])
    err_double = float(np.max(np.abs(phi(B, O) - S)))
    r1 = mp.fsum(abs(Bm[i, j]) for i in range(p) for j in range(p)) + \
        mp.fsum(abs(Om[i, j]) for i in range(p) for j in range(i + 1, p))
    vals = [abs(B[i, j]) for i in range(p) for j in range(p) if i != j] + \
           [abs(O[i, j]) for i in range(p) for j in range(i + 1, p)]
    above = [v for v in vals if v > 1e-9]
    below = [v for v in vals if v <= 1e-9]
    return dict(sigma_asymmetry=float(np.max(np.abs(S - S.T))), phi_err_double=err_double, phi_err_mp=float(err_mp),
                omega_min_eig=float(min_eig),
                r1_exact=float(r1), min_entry_in_support=min(above) if above else np.nan,
                max_entry_off_support=max(below) if below else 0.0, O=O)


def work(args):
    row, do_global = args
    t0 = time.time()
    S = np.array(json.loads(row["sigma"]))
    B = np.array(json.loads(row["global_B"]))
    G = parse_support(row["gstar"])
    key = INV[G]
    ex = exact_check(S, B)
    O = ex.pop("O")
    sup = actual_support(B, O)
    skey = INV.get(sup)
    out = dict(sigma_id=int(row["sigma_id"]), gstar=row["gstar"], family=row["family"],
               s1_global=float(row["global_min"]), s1_class=float(row["class_min"]),
               s1_margin=float(row["margin"]), outside_support=support_name(*sup),
               outside_rank=skey[0] if skey else -1, outside_in_class=int(skey == key))
    out.update(ex)
    # S2 on the class supports
    R = float(row["class_min"]) + 0.05
    cls_all = [s for s in all_supports(P) if INV[s] == key]
    maxi = maximal_supports(cls_all)
    # all class supports for classes of at most 40 supports, otherwise the maximal ones
    # (for generic Sigma the class minimum is attained on the closure of a maximal class support)
    cls = cls_all if len(cls_all) <= 40 else maxi
    best2, arg2, nfeas2, param_level = np.inf, None, 0, 0
    for D, U in cls:
        v, B2, order, diags = s2_support(S, D, U, R=R, n_grid={1: 801, 2: 121, 3: 41}, n_refine=6)
        param_level += int(any(d["param_level"] for d in diags))
        if np.isfinite(v):
            nfeas2 += 1
            if v < best2:
                best2, arg2 = v, (D, U)
    out.update(s2_class=best2, s2_class_support=support_name(*arg2) if arg2 else "",
               s2_class_feasible=nfeas2, n_class=len(cls_all), n_s2_supports=len(cls),
               s2_scope="all" if len(cls) == len(cls_all) else "maximal", s2_param_level_supports=param_level,
               class_min_both=min(best2, float(row["class_min"])))
    out["margin_verified"] = out["class_min_both"] - out["r1_exact"]
    if do_global:
        bestg = np.inf
        for D, U, perm in complete_supports(P):
            v, Bg, order, _ = s2_support(S, D, U, R=float(row["global_min"]) + 0.05, n_grid=25, n_refine=4)
            bestg = min(bestg, v)
        out["s2_global"] = bestg
        out["s1_minus_s2_global"] = float(row["global_min"]) - bestg
    else:
        out["s2_global"] = np.nan
        out["s1_minus_s2_global"] = np.nan
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--infile", default="p4_search.csv")
    ap.add_argument("--out", default="p4_verify.csv")
    ap.add_argument("--global-per-graph", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--pilot", type=int, default=0, help="only N randomly chosen candidates")
    args = ap.parse_args()
    t0 = time.time()
    init()
    with open_table(args.infile) as f:
        rows = [r for r in csv.DictReader(f) if float(r["margin"]) >= args.threshold]
    rng = np.random.default_rng(20261022)
    if args.pilot:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.pilot, replace=False))]
    # S2 global check for the largest-margin candidate (and optionally one random other) per graph
    by_g = {}
    for r in rows:
        by_g.setdefault(r["gstar"], []).append(r)
    glob = set()
    for g, rs in by_g.items():
        rs = sorted(rs, key=lambda r: -float(r["margin"]))
        glob.add(rs[0]["sigma_id"])
        if len(rs) > 1 and args.global_per_graph > 1:
            glob.add(rs[1 + rng.integers(len(rs) - 1)]["sigma_id"])
    jobs = [(r, r["sigma_id"] in glob) for r in rows]
    print(f"{len(jobs)} candidates, {len(glob)} with the S2 global check", flush=True)
    part = RESULTS / (args.out + ".partial.jsonl")
    done = {}
    if part.exists():
        for line in open(part):
            o = json.loads(line)
            done[o["sigma_id"]] = o
    jobs = [j for j in jobs if int(j[0]["sigma_id"]) not in done]
    print(f"{len(done)} already verified, {len(jobs)} to run", flush=True)
    out = list(done.values())
    with Pool(args.workers, initializer=init) as pool, open(part, "a") as fpart:
        for k, o in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            out.append(o)
            fpart.write(json.dumps(o, default=float) + "\n"); fpart.flush()
            print(f"{k+1}/{len(jobs)} {time.time()-t0:.0f}s Sigma {o['sigma_id']} {o['seconds']:.0f}s "
                  f"margin {o['margin_verified']:.4f} s2-s1 class {o['s2_class'] - o['s1_class']:.2e}", flush=True)
    out.sort(key=lambda r: r["sigma_id"])
    with open(RESULTS / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

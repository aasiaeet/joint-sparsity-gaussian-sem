"""Standardized l1 search on three vertices (Supplement S13).

Part A covers, in the standardized parametrization, every class on three vertices that is
not saturated.  Up to relabeling and sign flips of the variables (which permute supports and
leave R_1 unchanged), such a class is one of
  collider type  rho_12 = 0:            Sigma = [[1,0,a],[0,1,b],[a,b,1]]     (dim 5)
  chain type     rho_13.2 = 0:          Sigma = [[1,a,ab],[a,1,b],[ab,b,1]]   (dim 5)
  single edge    rho_13 = rho_23 = 0:   Sigma = [[1,a,0],[a,1,0],[0,0,1]]     (dim 4)
  empty          Sigma = I (R_1 = 0 at the empty graph).
We scan a, b in {0.01, 0.03, ..., 0.99} with a <= b (exchanging the outer vertices exchanges
a and b).  Part B takes each of the 62 bow-free acyclic mixed graphs with random parameters
in two regimes and equal-magnitude grid parameters, and standardizes the covariance.
For every Sigma, solver S1 runs on the 6 complete supports and on the class supports, and
solver S2 on all 200 supports.  In total 4,243 covariances.

Writes results/p3_search.csv (one row per Sigma) and results/p3_s2_values.npz.
Usage: python3 search_three_vertices.py [--workers W] [--only a:b]
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1core import all_supports, support_name, parse_support, is_bowfree, phi, standardize, constraint_names, constraint_label
from search_common import (RESULTS, MASTER_SEED, invariant_table, draw_parameter, grid_parameters,
                           evaluate_sigma_p3, extra_vanishing)

P = 3
INV = None


def init():
    global INV
    INV = invariant_table(P)


def build_sigmas():
    sigmas = []
    vals = np.round(np.arange(0.01, 1.0, 0.02), 2)
    types = {
        "collider": ((((0, 2), (1, 2)), ()), lambda a, b: np.array([[1, 0, a], [0, 1, b], [a, b, 1.0]])),
        "chain": ((((0, 1), (1, 2)), ()), lambda a, b: np.array([[1, a, a * b], [a, 1, b], [a * b, b, 1.0]])),
    }
    for t, (G, f) in types.items():
        for a in vals:
            for b in vals:
                if a > b:
                    continue
                S = f(a, b)
                if np.linalg.eigvalsh(S)[0] <= 1e-6:
                    continue
                sigmas.append(dict(part="A", family=t, gstar=support_name(*G), params=json.dumps([float(a), float(b)]),
                                   S=S.tolist()))
    for a in vals:
        S = np.array([[1, a, 0], [a, 1, 0], [0, 0, 1.0]])
        sigmas.append(dict(part="A", family="single", gstar="1->2", params=json.dumps([float(a)]), S=S.tolist()))
    # Part B
    bf = [s for s in all_supports(P) if is_bowfree(*s)]
    for gi, (D, U) in enumerate(bf):
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, P, 2, gi]))
        for regime in ("moderate", "wide"):
            for r in range(10):
                B, O = draw_parameter(D, U, P, rng, regime)
                sigmas.append(dict(part="B", family=regime, gstar=support_name(D, U),
                                   params=json.dumps(dict(B=B.tolist(), O=O.tolist())),
                                   S=standardize(phi(B, O)).tolist()))
        for c, sg, B, O in grid_parameters(D, U, P, (0.5, 1.0, 2.0), 4, rng):
            sigmas.append(dict(part="B", family="grid", gstar=support_name(D, U),
                               params=json.dumps(dict(c=c, signs=list(sg))), S=standardize(phi(B, O)).tolist()))
    for k, t in enumerate(sigmas):
        t["sigma_id"] = k
    return sigmas


def work(t):
    S = np.array(t["S"])
    G = parse_support(t["gstar"])
    key = INV[G]
    seed = np.random.SeedSequence([MASTER_SEED, P, 1, t["sigma_id"]])
    out, v2 = evaluate_sigma_p3(S, key, INV, seed)
    extra = extra_vanishing(S, key[1], P)
    names = constraint_names(P)
    row = dict(sigma_id=t["sigma_id"], part=t["part"], family=t["family"], gstar=t["gstar"], dim_star=key[0],
               params=t["params"], sigma=json.dumps(np.round(S, 15).tolist()),
               extra_vanishing=" ".join(constraint_label(names[i]) for i in extra))
    row.update(out)
    return row, v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only", default="", help="range a:b of Sigma indices (default: all)")
    args = ap.parse_args()
    t0 = time.time()
    init()
    sigmas = build_sigmas()
    if args.only:
        a, b = (int(x) for x in args.only.split(":"))
        sigmas = sigmas[a:b]
    print(f"{len(sigmas)} Sigma", flush=True)
    rows, V = [None] * len(sigmas), [None] * len(sigmas)
    with Pool(args.workers, initializer=init) as pool:
        for k, (row, v2) in enumerate(pool.imap(work, sigmas, chunksize=2)):
            rows[k], V[k] = row, v2
            if (k + 1) % 200 == 0:
                print(f"{k+1}/{len(sigmas)}  {time.time()-t0:.0f}s  max margin so far "
                      f"{max(r['margin'] for r in rows[:k+1]):.3e}", flush=True)
    with open(RESULTS / "p3_search.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(RESULTS / "p3_s2_values.npz", values=np.array(V),
                        supports=np.array([support_name(*s) for s in all_supports(P)]))
    m = np.array([r["margin"] for r in rows])
    print(f"done {time.time()-t0:.0f}s; max margin {m.max():.3e}; n margin>=1e-3: {(m >= 1e-3).sum()}", flush=True)


if __name__ == "__main__":
    main()

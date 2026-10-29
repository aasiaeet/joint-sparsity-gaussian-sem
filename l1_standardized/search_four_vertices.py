"""Standardized l1 search on four vertices (Supplement S13).

Generating graphs: every bow-free acyclic mixed graph on four vertices up to relabeling
(192 graphs; R_1 is invariant under relabeling), restricted to the 161 whose model is not
saturated (generic Jacobian rank < 10).  For a saturated G* every exact representation of a
generic Sigma has a support of rank 10, so the class contains all of them.

Parameters per graph: 12 random draws in a moderate regime, 12 in a wide regime (large
coefficients, small error variances, strong error correlations), and equal-magnitude grid
parameters (|B| in {0.5, 1, 2}, up to 4 sign patterns, unit error variances, bidirected
error correlation +-0.5); Sigma is then standardized.  In total 5,721 covariances.

Per Sigma: the minimum of R_1 over all exact representations equals the minimum over the 24
complete supports (one per vertex order), so solver S1 runs on those, keeping every local
solution and classifying its actual support.  If no local solution in the generating class
comes within 1e-3 of the global value, S1 also runs on the supports of the class (same
Jacobian rank and the same vanishing correlations, partial correlations and tetrads as G*)
and, with more starts, on the maximal supports of the class; the margin is the class minimum
minus the global minimum.  Otherwise the reported margin is an upper bound below 1e-3
(column margin_is_upper_bound).  For Sigma indices below FULL_CLASS_BEFORE every support of
the class is solved; from that index on, classes with more than 40 supports are represented
by their maximal supports only (for generic Sigma the class minimum is attained on the
closure of a maximal class support).

Writes results/p4_search.csv; finished rows are also appended to
results/p4_search.csv.partial.jsonl, so an interrupted run resumes where it stopped.
Usage: python3 search_four_vertices.py [--workers W] [--pilot N]
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
from l1core import all_supports, support_name, parse_support, phi, standardize, constraint_names, constraint_label
from search_common import (RESULTS, MASTER_SEED, invariant_table, draw_parameter, grid_parameters,
                           evaluate_sigma_p4, extra_vanishing, bowfree_graphs_up_to_iso,
                           maximal_supports)

P = 4
INV = None
CLASS = {}
MAXIMAL = {}
FULL_CLASS_BEFORE = 3691
LARGE_CLASS = 40


def init():
    global INV
    INV = invariant_table(P)


def class_supports(key):
    if key not in CLASS:
        CLASS[key] = [s for s in all_supports(P) if INV[s] == key]
        MAXIMAL[key] = maximal_supports(CLASS[key])
    return CLASS[key]


def build_sigmas(n_random=12, graphs=None):
    inv = invariant_table(P)
    reps = bowfree_graphs_up_to_iso(P)
    sigmas = []
    for gi, (D, U) in enumerate(reps):
        if inv[(D, U)][0] >= 10:
            continue
        if graphs is not None and gi not in graphs:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, P, 2, gi]))
        for regime in ("moderate", "wide"):
            for r in range(n_random):
                B, O = draw_parameter(D, U, P, rng, regime)
                sigmas.append(dict(graph_index=gi, family=regime, gstar=support_name(D, U),
                                   params=json.dumps(dict(B=np.round(B, 15).tolist(), O=np.round(O, 15).tolist())),
                                   S=standardize(phi(B, O)).tolist()))
        for c, sg, B, O in grid_parameters(D, U, P, (0.5, 1.0, 2.0), 4, rng):
            sigmas.append(dict(graph_index=gi, family="grid", gstar=support_name(D, U),
                               params=json.dumps(dict(c=c, signs=list(sg))), S=standardize(phi(B, O)).tolist()))
    for k, t in enumerate(sigmas):
        t["sigma_id"] = k
    return sigmas


def work(t):
    S = np.array(t["S"])
    G = parse_support(t["gstar"])
    key = INV[G]
    seed = np.random.SeedSequence([MASTER_SEED, P, 1, t["sigma_id"]])
    cls = class_supports(key)
    all_class_max = 10**9 if t["sigma_id"] < FULL_CLASS_BEFORE else LARGE_CLASS
    out, Bbest, class_vals = evaluate_sigma_p4(S, key, INV, cls, seed, maximal=MAXIMAL[key],
                                               all_class_max=all_class_max)
    extra = extra_vanishing(S, key[1], P)
    names = constraint_names(P)
    row = dict(sigma_id=t["sigma_id"], graph_index=t["graph_index"], family=t["family"], gstar=t["gstar"],
               n_edges_star=len(G[0]) + len(G[1]), dim_star=key[0],
               vanishing_star=" ".join(constraint_label(names[i]) for i in key[1]),
               params=t["params"], sigma=json.dumps(np.round(S, 15).tolist()),
               extra_vanishing=" ".join(constraint_label(names[i]) for i in extra))
    row.update(out)
    row["global_B"] = json.dumps(np.round(Bbest, 15).tolist())
    row["class_values"] = json.dumps({k: (v if np.isfinite(v) else None) for k, v in class_vals.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--pilot", type=int, default=0, help="only N randomly chosen Sigma")
    ap.add_argument("--out", default="p4_search.csv")
    args = ap.parse_args()
    t0 = time.time()
    init()
    sigmas = build_sigmas()
    if args.pilot:
        rng = np.random.default_rng(1)
        sigmas = [sigmas[i] for i in sorted(rng.choice(len(sigmas), args.pilot, replace=False))]
    print(f"{len(sigmas)} Sigma", flush=True)
    part = RESULTS / (args.out + ".partial.jsonl")
    done = {}
    if part.exists():
        for line in open(part):
            row = json.loads(line)
            done[row["sigma_id"]] = row
    pending = [t for t in sigmas if t["sigma_id"] not in done]
    print(f"{len(done)} already done, {len(pending)} to run", flush=True)
    rows = list(done.values())
    with Pool(args.workers, initializer=init) as pool, open(part, "a") as fpart:
        for k, row in enumerate(pool.imap_unordered(work, pending, chunksize=1)):
            rows.append(row)
            fpart.write(json.dumps(row, default=float) + "\n"); fpart.flush()
            if (k + 1) % 100 == 0:
                m = np.array([r["margin"] for r in rows])
                print(f"{k+1}/{len(pending)}  {time.time()-t0:.0f}s  max margin {m.max():.3e}  "
                      f"n>=1e-3 {(m >= 1e-3).sum()}", flush=True)
    rows.sort(key=lambda r: r["sigma_id"])
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(RESULTS / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)
    m = np.array([r["margin"] for r in rows])
    print(f"done {time.time()-t0:.0f}s; max margin {m.max():.3e}; n margin>=1e-3: {(m >= 1e-3).sum()}", flush=True)


if __name__ == "__main__":
    main()

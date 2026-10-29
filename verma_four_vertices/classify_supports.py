"""Algebraic and Markov invariants of all 34,752 acyclic mixed supports on four vertices.

For each support H:
 - generic Jacobian rank (= dim V_H) over GF(2^61 - 1) at two random points,
   cross-checked by the float SVD rank (relative tolerance 1e-8, 3 random real draws);
 - which of the 24 CI minors and the 24 relabeled Verma cubics vanish identically
   on M_H (zero at both random GF(p) points; Schwartz-Zippel);
 - the m-separation statements of H (graphical), and whether they coincide with
   the vanishing CI minors (they must, by Spirtes et al. 1998);
 - membership in the class of G*: dim V_H = 8, f_CI and f_V vanish identically on
   M_H, and s11 s22 s13 does not (Supplement S18);
 - Markov equivalence to G* (same m-separation statements).
Writes results/classes.csv and results/classes_meta.json.
Usage: python3 classify_supports.py [--workers W]
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

import numpy as np

from common import (CI_STATEMENTS, GSTAR, MASTER_SEED, P, RESULTS, SUPPORTS, STUDY_ID, VERMA_PERMS,
                    algebraic_invariants, covariance, in_gstar_class, msep_vector)



def float_rank(s, draws=3, tol=1e-8):
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, s.index, 5]))
    iu = np.triu_indices(P)
    rank = 0
    for _ in range(draws):
        B = np.zeros((P, P))
        for e in s.D:
            B[e] = rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)
        while True:
            O = np.diag(rng.uniform(0.6, 1.6, P))
            for i, j in s.U:
                O[i, j] = O[j, i] = rng.choice([-1, 1]) * rng.uniform(0.1, 0.4)
            if np.linalg.eigvalsh(O)[0] > 0.1:
                break
        C = np.linalg.inv(np.eye(P) - B)
        Sig = C.T @ O @ C
        cols = []
        for (i, j) in s.D:
            E = np.zeros((P, P)); E[i, j] = 1
            cols.append((C.T @ E.T @ Sig + Sig @ E @ C)[iu])
        for i in range(P):
            E = np.zeros((P, P)); E[i, i] = 1
            cols.append((C.T @ E @ C)[iu])
        for (i, j) in s.U:
            E = np.zeros((P, P)); E[i, j] = E[j, i] = 1
            cols.append((C.T @ E @ C)[iu])
        sv = np.linalg.svd(np.column_stack(cols), compute_uv=False)
        rank = max(rank, int(np.sum(sv > tol * sv[0])))
    return rank


def bits(v):
    return "".join("1" if x else "0" for x in v)


def classify(idx):
    s = SUPPORTS[idx]
    inv = algebraic_invariants(s)
    ms = msep_vector(s)
    return dict(index=s.index, name=s.name, edges=s.count, n_directed=len(s.D), n_bidirected=len(s.U),
                bowed=int(s.bowed), rank=inv["rank"], rank_float=float_rank(s),
                ci_vanish=bits(inv["ci_vanish"]), verma_vanish=bits(inv["verma_vanish"]), msep=bits(ms),
                msep_matches_ci=int(tuple(ms) == tuple(inv["ci_vanish"])),
                aux_nonvanishing=int(inv["aux_nonvanishing"]), in_gstar_class=int(in_gstar_class(inv)),
                class_key=f"dim{inv['rank']}|ci:{bits(inv['ci_vanish'])}|verma:{bits(inv['verma_vanish'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    t0 = time.time()
    with mpc.Pool(args.workers) as pool:
        rows = pool.map(classify, range(len(SUPPORTS)), chunksize=64)
    gmsep = rows[GSTAR.index]["msep"]
    for r in rows:
        r["markov_equiv_gstar"] = int(r["msep"] == gmsep)
    with open(RESULTS / "classes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = dict(seconds=time.time() - t0, workers=args.workers,
                ci_statement_order=[f"{i+1}_||_{j+1}|" + "".join(str(v + 1) for v in S) for i, j, S in CI_STATEMENTS],
                verma_perm_order=["".join(str(v + 1) for v in pm) for pm in VERMA_PERMS],
                n_rank_mismatch_float=sum(r["rank"] != r["rank_float"] for r in rows),
                n_msep_ci_mismatch=sum(1 - r["msep_matches_ci"] for r in rows),
                n_in_gstar_class=sum(r["in_gstar_class"] for r in rows),
                n_markov_equiv_gstar=sum(r["markov_equiv_gstar"] for r in rows),
                n_distinct_class_keys=len({r["class_key"] for r in rows}))
    (RESULTS / "classes_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if not k.endswith("order")}, indent=1))


if __name__ == "__main__":
    main()

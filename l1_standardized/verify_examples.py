"""Both solvers on the four rational diamond examples of analytic_family.py (Supplement S13).

For each example: S1 (40 random starts per order) and S2 (stage solver, 41-point grid per outer
coordinate) on the 24 complete supports (global minimum of R_1), and S1 and S2 on every support of
the generating class (class minimum), plus the exact 60-digit recomputation of the S1 optimum.  For the first example (the diamond of
Supplement S13) both solvers give 2.1 for the global minimum and 2.4 for the class minimum.
Writes results/verify_examples.json.
Usage: python3 verify_examples.py [--only name[,name...]]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1core import all_supports, complete_supports, solve_support, support_name, actual_support, omega_of
from stage_solver import s2_support
from search_common import RESULTS, invariant_table
from analytic_family import diamond_sigma
from verify_four_vertices import exact_check

Rat = sp.Rational
EXAMPLES = [
    dict(name="simplest", r=Rat(1, 2), s=Rat(1, 2), b2=Rat(4, 5), b3=Rat(-3, 5)),
    dict(name="moderate", r=Rat(4, 5), s=Rat(4, 5), b2=Rat(1), b3=Rat(-9, 10)),
    dict(name="strong", r=Rat(9, 10), s=Rat(9, 10), b2=Rat(3, 2), b3=Rat(-7, 5)),
    dict(name="asymmetric", r=Rat(3, 5), s=Rat(4, 5), b2=Rat(1), b3=Rat(-4, 5)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated example names (default: all four)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    inv = invariant_table(4)
    gstar = (((0, 1), (0, 2), (1, 3), (2, 3)), ())
    key = inv[gstar]
    cls = [s for s in all_supports(4) if inv[s] == key]
    out = []
    for k_ex, ex in enumerate(EXAMPLES):
        if only is not None and ex["name"] not in only:
            continue
        t0 = time.time()
        Sig, *_ = diamond_sigma(ex["r"], ex["s"], ex["b2"], ex["b3"])
        S = np.array(Sig.evalf(40).tolist(), float)
        rng = np.random.default_rng(np.random.SeedSequence([20261022, 4, 9, k_ex]))
        g1 = []
        for D, U, perm in complete_supports(4):
            sol = solve_support(S, D, U, rng, n_random=40)
            g1.append((sol.value, perm, sol))
        g1.sort(key=lambda x: x[0])
        v1, perm1, sol1 = g1[0]
        R = v1 + 0.05
        g2 = []
        for D, U, perm in complete_supports(4):
            v, B, order, _ = s2_support(S, D, U, R=R, n_grid=41, n_refine=6)
            g2.append((v, perm, B))
        g2.sort(key=lambda x: x[0])
        c1, c2 = {}, {}
        for D, U in cls:
            sol = solve_support(S, D, U, rng, n_random=30)
            c1[support_name(D, U)] = sol.value if sol is not None else None
            v, B, order, _ = s2_support(S, D, U, R=float(sum(abs(float(x)) for x in (ex["r"], ex["s"], ex["b2"], ex["b3"]))) + 0.05,
                                        n_grid=201, n_refine=6)
            c2[support_name(D, U)] = v if np.isfinite(v) else None
        ex_chk = exact_check(S, sol1.B)
        ex_chk.pop("O")
        sup1 = sol1.support
        rec = dict(name=ex["name"], s1_global=v1, s1_global_order="".join(str(x + 1) for x in perm1),
                   s1_global_support=support_name(*sup1), s1_global_support_rank=inv[sup1][0],
                   s1_global_in_class=int(inv[sup1] == key),
                   s2_global=g2[0][0], s2_global_order="".join(str(x + 1) for x in g2[0][1]),
                   s2_global_support=support_name(*actual_support(g2[0][2], omega_of(g2[0][2], S))),
                   class_s1=min(v for v in c1.values() if v is not None),
                   class_s2=min(v for v in c2.values() if v is not None),
                   class_s1_values=c1, class_s2_values=c2,
                   class_lower_bound=float(abs(ex["r"]) + abs(ex["s"]) + abs(ex["b2"]) + abs(ex["b3"])),
                   exact_check_s1_optimum=ex_chk, s1_B=np.round(sol1.B, 15).tolist(), s1_Omega=np.round(sol1.O, 15).tolist(),
                   seconds=time.time() - t0)
        rec["margin"] = min(rec["class_s1"], rec["class_s2"]) - min(rec["s1_global"], rec["s2_global"])
        out.append(rec)
        print(ex["name"], {k: rec[k] for k in ("s1_global", "s2_global", "class_s1", "class_s2", "class_lower_bound",
                                                "margin", "s1_global_support", "s1_global_support_rank", "seconds")},
              ex_chk, flush=True)
    with open(RESULTS / "verify_examples.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

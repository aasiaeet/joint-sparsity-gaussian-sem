"""The diamond family of four-vertex counterexamples, in exact rational arithmetic (Supplement S13).

Generating graph G*: the diamond 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4, all variables standardized:
    rho_12 = r, rho_13 = s, rho_23 = r s  (2 _||_ 3 | 1),
    X4 = beta2 X2 + beta3 X3 + e4,  omega_4 = 1 - beta2^2 - beta3^2 - 2 beta2 beta3 r s > 0.
Then sigma_14 = r beta2 + s beta3, sigma_24 = beta2 + beta3 r s, sigma_34 = beta2 r s + beta3.

Inside the class:  every exact representation whose graph is Markov (hence also every one
whose graph is algebraically) equivalent to G* has R_1 >= |r| + |s| + |beta2| + |beta3|,
with equality at the generating representation (proof in Supplement S13).

Exact representations outside the class (error covariance determined by B):
  theta_A, graph 4 -> 2, 4 -> 3, 2 <-> 3, 2 -> 1, 3 -> 1 (bow-free, 5 edges):
      B_42 = sigma_24, B_43 = sigma_34, Omega_23 = r s - sigma_24 sigma_34,
      B_21 = r (1 - s^2) / (1 - r^2 s^2), B_31 = s (1 - r^2) / (1 - r^2 s^2);
  theta_B, the DAG 4 -> 3, 4 -> 2, 3 -> 2, 2 -> 1, 3 -> 1 (regressions along the order 4, 3, 2, 1);
  theta_C, graph 2 -> 1, 3 -> 1, 2 <-> 3, 2 <-> 4, 3 <-> 4 (bow-free, 5 edges): vertices 2, 3, 4
      have no parents, so Omega_{234} = Sigma_{234}, and X1 is regressed on (X2, X3); the
      regression of X1 on (X2, X3, X4) has a zero X4 coefficient because 1 _||_ 4 | {2, 3}, so
      Omega_14 = 0 and theta_C is exact:
      R_1(theta_C) = |r s| + |sigma_24| + |sigma_34| + (|r|(1 - s^2) + |s|(1 - r^2)) / (1 - r^2 s^2).
All three graphs make 2 and 3 adjacent, so none is Markov equivalent to G*.
When r, s > 0, beta2 > 0 > beta3, sigma_24 > 0 > sigma_34 (opposite-sign paths through 2 and 3):
      R_in - R_1(theta_C) = (|beta2| + |beta3|) r s - kappa(r, s),
      kappa(r, s) = r s - r - s + (r(1 - s^2) + s(1 - r^2)) / (1 - r^2 s^2)  (= rho^2 (1 - rho)^2 / (1 + rho^2) at r = s = rho).
Writes results/analytic_family.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1core import complete_supports, solve_support, support_name, invariant, omega_of, actual_support
from search_common import RESULTS, invariant_table

Rat = sp.Rational
I4 = sp.eye(4)


def diamond_sigma(r, s, b2, b3):
    B = sp.zeros(4, 4)
    B[0, 1], B[0, 2], B[1, 3], B[2, 3] = r, s, b2, b3
    w4 = 1 - (b2 ** 2 + b3 ** 2 + 2 * b2 * b3 * r * s)
    O = sp.diag(1, 1 - r ** 2, 1 - s ** 2, w4)
    Sig = ((I4 - B).inv()).T * O * (I4 - B).inv()
    return sp.simplify(Sig), B, O, w4


def rep_from_B(Sig, B):
    O = sp.simplify((I4 - B).T * Sig * (I4 - B))
    ok = sp.simplify(((I4 - B).inv()).T * O * (I4 - B).inv() - Sig) == sp.zeros(4, 4)
    R1 = sum(abs(x) for x in B) + sum(abs(O[i, j]) for i in range(4) for j in range(i + 1, 4))
    return O, ok, sp.nsimplify(R1)


def witness_A(Sig):
    r, s = Sig[0, 1], Sig[0, 2]
    s24, s34 = Sig[1, 3], Sig[2, 3]
    B = sp.zeros(4, 4)
    B[3, 1], B[3, 2] = s24, s34
    B[1, 0] = (r - s * Sig[1, 2]) / (1 - Sig[1, 2] ** 2)
    B[2, 0] = (s - r * Sig[1, 2]) / (1 - Sig[1, 2] ** 2)
    return B


def witness_B(Sig):
    B = sp.zeros(4, 4)
    B[3, 2] = Sig[2, 3]
    M = sp.Matrix([[1, Sig[2, 3]], [Sig[2, 3], 1]])
    b = M.inv() * sp.Matrix([Sig[1, 2], Sig[1, 3]])
    B[2, 1], B[3, 1] = b[0], b[1]
    M2 = sp.Matrix([[1, Sig[1, 2]], [Sig[1, 2], 1]])
    c = M2.inv() * sp.Matrix([Sig[0, 1], Sig[0, 2]])
    B[1, 0], B[2, 0] = c[0], c[1]
    return B


def witness_C(Sig):
    B = sp.zeros(4, 4)
    r, s, t = Sig[0, 1], Sig[0, 2], Sig[1, 2]
    B[1, 0] = (r - s * t) / (1 - t ** 2)
    B[2, 0] = (s - r * t) / (1 - t ** 2)
    return B


def support_of(B, O):
    D = tuple((i, j) for i in range(4) for j in range(4) if i != j and B[i, j] != 0)
    U = tuple((i, j) for i in range(4) for j in range(i + 1, 4) if O[i, j] != 0)
    return D, U


def numeric_l1(Sig_f, seed=0, n_random=40):
    rng = np.random.default_rng(seed)
    best = None
    for D, U, perm in complete_supports(4):
        sol = solve_support(Sig_f, D, U, rng, n_random=n_random)
        if best is None or sol.value < best[0]:
            best = (sol.value, sol, perm)
    return best


def main():
    inv = invariant_table(4)
    gstar = (((0, 1), (0, 2), (1, 3), (2, 3)), ())
    key = inv[gstar]
    examples = [
        dict(name="simplest", r=Rat(1, 2), s=Rat(1, 2), b2=Rat(4, 5), b3=Rat(-3, 5)),
        dict(name="moderate", r=Rat(4, 5), s=Rat(4, 5), b2=Rat(1), b3=Rat(-9, 10)),
        dict(name="strong", r=Rat(9, 10), s=Rat(9, 10), b2=Rat(3, 2), b3=Rat(-7, 5)),
        dict(name="asymmetric", r=Rat(3, 5), s=Rat(4, 5), b2=Rat(1), b3=Rat(-4, 5)),
    ]
    out = []
    for ex in examples:
        Sig, B0, O0, w4 = diamond_sigma(ex["r"], ex["s"], ex["b2"], ex["b3"])
        assert w4 > 0
        Rin = abs(ex["r"]) + abs(ex["s"]) + abs(ex["b2"]) + abs(ex["b3"])
        O_gen, ok_gen, R_gen = rep_from_B(Sig, B0)
        rec = dict(name=ex["name"], r=str(ex["r"]), s=str(ex["s"]), beta2=str(ex["b2"]), beta3=str(ex["b3"]),
                   omega4=str(w4), sigma=[[str(Sig[i, j]) for j in range(4)] for i in range(4)],
                   R1_generating=str(R_gen), R1_class_lower_bound=str(Rin))
        for wname, wf in (("A", witness_A), ("B", witness_B), ("C", witness_C)):
            Bw = wf(Sig)
            Ow, ok, R1w = rep_from_B(Sig, Bw)
            D, U = support_of(Bw, Ow)
            eig = [float(x) for x in np.linalg.eigvalsh(np.array(Ow.evalf(30).tolist(), float))]
            rec[f"witness_{wname}"] = dict(support=support_name(D, U), exact=bool(ok), R1=str(R1w),
                                           R1_float=float(R1w), margin=str(sp.nsimplify(Rin - R1w)),
                                           margin_float=float(Rin - R1w), omega_eigs=eig,
                                           rank=inv[(tuple(sorted(D)), tuple(sorted(U)))][0],
                                           B=[[str(Bw[i, j]) for j in range(4)] for i in range(4)],
                                           Omega=[[str(Ow[i, j]) for j in range(4)] for i in range(4)])
        Sf = np.array(Sig.evalf(30).tolist(), float)
        best = numeric_l1(Sf)
        sol = best[1]
        rec["numeric_l1_min"] = best[0]
        rec["numeric_l1_support"] = support_name(*sol.support)
        rec["numeric_l1_support_rank"] = inv[sol.support][0]
        rec["numeric_l1_B"] = np.round(sol.B, 12).tolist()
        rec["numeric_l1_Omega"] = np.round(sol.O, 12).tolist()
        rec["numeric_margin"] = float(Rin) - best[0]
        out.append(rec)
        print(ex["name"], "R_in", float(Rin), "A", rec["witness_A"]["R1_float"], "B", rec["witness_B"]["R1_float"],
              "C", rec["witness_C"]["R1"], rec["witness_C"]["R1_float"], rec["witness_C"]["support"], "rank", rec["witness_C"]["rank"],
              "num", best[0], rec["numeric_l1_support"], flush=True)
    # opposite-sign family: r = s = rho, beta2 = b > 0 > beta3 = -c with sigma_24 > 0 > sigma_34
    rho, bsum = sp.symbols("rho bsum", positive=True)
    kappa = sp.simplify(rho ** 2 - 2 * rho + 2 * rho / (1 + rho ** 2))
    fam = dict(R_in="2 rho + |beta2| + |beta3|",
               R_C="rho^2 + |sigma_24| + |sigma_34| + 2 rho/(1 + rho^2)",
               margin_when_opposite_signs="(|beta2| + |beta3|) rho^2 - kappa(rho)",
               kappa=str(sp.factor(kappa)),
               threshold="|beta2| + |beta3| > (1 - rho)^2 / (1 + rho^2)",
               admissible="omega_4 = 1 - beta2^2 - beta3^2 - 2 beta2 beta3 rho^2 > 0, sigma_24 > 0 > sigma_34")
    scan = []
    for rv in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        thr = (1 - rv) ** 2 / (1 + rv ** 2)
        # largest admissible |beta2| + |beta3| with beta2 = -beta3 = b: 2 b^2 (1 - rho^2) < 1
        bmax = 2 / np.sqrt(2 * (1 - rv ** 2))
        scan.append(dict(rho=rv, threshold_sum_abs_beta=thr, sup_sum_abs_beta_symmetric=bmax,
                         sup_margin_symmetric=bmax * rv ** 2 - float(kappa.subs(rho, rv))))
    fam["scan"] = scan
    with open(RESULTS / "analytic_family.json", "w") as f:
        json.dump(dict(examples=out, symmetric_family=fam), f, indent=1)
    print(json.dumps(fam, indent=1))


if __name__ == "__main__":
    main()

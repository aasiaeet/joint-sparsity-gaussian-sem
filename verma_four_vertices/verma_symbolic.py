"""The Verma constraint of G* = 1->2->3->4, 2<->4 (main Section 7.2, Supplement S18).

Writes results/verma_constraint.json and prints a log.  Checks:
 (a) symbolic: f_CI = s13 s22 - s12 s23 and the Verma cubic
     f_V = s13 s22 s34 - s14 s22 s33 - s13 s23 s24 + s14 s23^2
     vanish identically on Sigma(theta) for the 8 free parameters of G*;
 (b) high precision: |f_V(Sigma(theta))| at random real parameters, mpmath 60 digits;
 (c) of the 24 CI minors only f_CI vanishes identically (each other minor is
     nonzero at an exact rational point);
 (d) generic Jacobian rank 8 (exact rank at a rational point; 8 columns);
 (e) f_V is irreducible, is not a CI minor, and is not in the prime ideal <f_CI>:
     a positive definite rational Sigma0 on the DAG 1->2->3, 1->4, 2->4, 3->4
     (which satisfies 1 _||_ 3 | 2) has f_CI = 0 and f_V != 0;
 (f) the rational inverse lambda12 = s12/s11, lambda23 = s23/s22, lambda34 = s14/s13
     maps every Sigma in V(f_CI, f_V) with s11 s22 s13 != 0 to G*-parameters
     reproducing Sigma: every off-pattern entry of Omega = A^T Sigma A reduces to 0
     modulo a Groebner basis of <f_CI, f_V> (after clearing the denominators);
 (g) the vanishing ideal itself is studied in verma_ideal.py.
"""
from __future__ import annotations

import itertools
import json
import time
from fractions import Fraction

import mpmath as mp
import numpy as np
import sympy as sp

from common import (CI_STATEMENTS, P, RESULTS, ci_name, MASTER_SEED, STUDY_ID,
                    GSTAR, random_point_mod, covariance_mod, PRIME)

t0 = time.time()
out = {}
sym = sp.symbols("s11 s12 s13 s14 s22 s23 s24 s33 s34 s44")
s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = sym
SIG = sp.Matrix([[s11, s12, s13, s14], [s12, s22, s23, s24], [s13, s23, s33, s34], [s14, s24, s34, s44]])
F_CI = s13 * s22 - s12 * s23
F_V = s13 * s22 * s34 - s14 * s22 * s33 - s13 * s23 * s24 + s14 * s23 ** 2
out["f_CI"] = str(F_CI)
out["f_V"] = str(F_V)

# (a) symbolic parametrization
a, b, c, w1, w2, w3, w4, w24 = sp.symbols("a b c w1 w2 w3 w4 w24")
theta = [a, b, c, w1, w2, w3, w4, w24]
Bs = sp.zeros(4)
Bs[0, 1], Bs[1, 2], Bs[2, 3] = a, b, c
Os = sp.diag(w1, w2, w3, w4)
Os[1, 3] = Os[3, 1] = w24
Cs = (sp.eye(4) - Bs).inv()
Ss = sp.expand(Cs.T * Os * Cs)
subs = {SIG[i, j]: Ss[i, j] for i in range(4) for j in range(i, 4)}
fci_sub = sp.expand(F_CI.subs(subs, simultaneous=True))
fv_sub = sp.expand(F_V.subs(subs, simultaneous=True))
out["a_symbolic"] = dict(f_CI_on_model=str(fci_sub), f_V_on_model=str(fv_sub),
                         passed=bool(fci_sub == 0 and fv_sub == 0))
print("(a) f_CI(Sigma(theta)) =", fci_sub, "; f_V(Sigma(theta)) =", fv_sub)

# (b) high precision random real parameters
mp.mp.dps = 60
rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, 0, 10]))
hp = []
for _ in range(5):
    vals = [mp.mpf(str(x)) for x in rng.uniform(-1.5, 1.5, 3)] + \
           [mp.mpf(str(x)) for x in rng.uniform(0.5, 2.0, 4)] + [mp.mpf(str(rng.uniform(-0.4, 0.4)))]
    Bm = mp.zeros(4)
    Bm[0, 1], Bm[1, 2], Bm[2, 3] = vals[0], vals[1], vals[2]
    Om = mp.diag(vals[3:7])
    Om[1, 3] = Om[3, 1] = vals[7]
    Cm = mp.inverse(mp.eye(4) - Bm)
    Sm = Cm.T * Om * Cm
    s = lambda i, j: Sm[i - 1, j - 1]
    fv = s(1, 3) * s(2, 2) * s(3, 4) - s(1, 4) * s(2, 2) * s(3, 3) - s(1, 3) * s(2, 3) * s(2, 4) + s(1, 4) * s(2, 3) ** 2
    fci = s(1, 3) * s(2, 2) - s(1, 2) * s(2, 3)
    scale = max(abs(s(i, j)) for i in range(1, 5) for j in range(1, 5)) ** 3
    hp.append(dict(abs_f_V=float(abs(fv)), abs_f_CI=float(abs(fci)), f_V_scale=float(scale)))
out["b_high_precision"] = dict(dps=60, draws=hp, max_abs_f_V=max(h["abs_f_V"] for h in hp))
print("(b) mpmath 60 digits: max |f_V| =", out["b_high_precision"]["max_abs_f_V"])

# exact rational point on G*
qpt = {a: sp.Rational(7, 9), b: sp.Rational(-5, 7), c: sp.Rational(11, 13), w1: sp.Rational(10, 11),
       w2: sp.Rational(13, 12), w3: sp.Rational(9, 10), w4: sp.Rational(17, 16), w24: sp.Rational(3, 7)}
Sq = Ss.subs(qpt)

# (c) CI minors
ci_tab = []
for st in CI_STATEMENTS:
    i, j, S = st
    rows, cols = [i] + list(S), [j] + list(S)
    minor = sp.expand(sp.Matrix([[Ss[r, cc] for cc in cols] for r in rows]).det())
    ci_tab.append(dict(statement=ci_name(st), identically_zero=bool(minor == 0),
                       value_at_rational_point=str(sp.Matrix([[Sq[r, cc] for cc in cols] for r in rows]).det())))
van = [d["statement"] for d in ci_tab if d["identically_zero"]]
out["c_ci_minors"] = dict(vanishing=van, table=ci_tab)
print("(c) CI minors vanishing identically on M_G*:", van)

# (d) Jacobian rank
J = sp.Matrix([[sp.diff(Ss[i, j], t) for t in theta] for i in range(4) for j in range(i, 4)])
Jq = J.subs(qpt)
rk = Jq.rank()
out["d_jacobian"] = dict(rank_at_rational_point=int(rk), n_params=8, ambient_dim=10, codim=10 - int(rk),
                         rational_point={str(k): str(v) for k, v in qpt.items()})
print("(d) Jacobian rank at rational point:", rk, "(8 columns) -> dim V_G* = 8, codim 2")

# (e) f_V not CI and not in <f_CI>
fac = sp.factor_list(F_V)
fac_ci = sp.factor_list(F_CI)
cubic_minors = []
for st in CI_STATEMENTS:
    i, j, S = st
    if len(S) == 2:
        rows, cols = [i] + list(S), [j] + list(S)
        cubic_minors.append(sp.expand(SIG.extract(rows, cols).det()))
prop = any(sp.simplify(F_V / m).is_constant() for m in cubic_minors if m != 0)
# rational PD point on DAG 1->2->3, 1->4, 2->4, 3->4 (1 _||_ 3 | 2 holds)
B0 = sp.zeros(4)
B0[0, 1], B0[1, 2], B0[0, 3], B0[1, 3], B0[2, 3] = sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(-1, 4), sp.Rational(3, 5)
C0 = (sp.eye(4) - B0).inv()
S0 = C0.T * sp.eye(4) * C0
sub0 = {SIG[i, j]: S0[i, j] for i in range(4) for j in range(i, 4)}
fci0 = F_CI.subs(sub0, simultaneous=True)
fv0 = F_V.subs(sub0, simultaneous=True)
pd0 = all(S0[:k, :k].det() > 0 for k in range(1, 5))  # Sylvester
out["e_not_ci"] = dict(f_V_factorization=str(fac), f_CI_factorization=str(fac_ci),
                       f_V_proportional_to_a_cubic_CI_minor=bool(prop),
                       witness_graph="1->2, 2->3, 1->4, 2->4, 3->4 with B12=1/2, B23=2/3, B14=1/3, B24=-1/4, B34=3/5, Omega=I",
                       witness_Sigma=[[str(S0[i, j]) for j in range(4)] for i in range(4)],
                       witness_positive_definite=bool(pd0), f_CI_at_witness=str(fci0), f_V_at_witness=str(fv0))
print("(e) f_V factors:", fac, "; proportional to a CI minor:", prop,
      "; witness f_CI =", fci0, " f_V =", fv0, " PD:", pd0)

# (f) rational inverse
l12, l23, l34 = s12 / s11, s23 / s22, s14 / s13
Ainv = sp.eye(4)
Ainv[0, 1], Ainv[1, 2], Ainv[2, 3] = -l12, -l23, -l34
Om_hat = sp.simplify(Ainv.T * SIG * Ainv)
G = sp.groebner([F_CI, F_V], *sym, order="grevlex")
red = {}
for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]:
    num, den = sp.fraction(sp.together(Om_hat[i, j]))
    num = sp.expand(num)
    r = G.reduce(num)[1]
    red[f"Omega_{i+1}{j+1}"] = {"numerator": str(sp.factor(num)), "denominator": str(sp.factor(den)),
                                 "remainder": str(r)}
out["f_rational_inverse"] = dict(
    lambdas=dict(B12="s12/s11", B23="s23/s22", B34="s14/s13"),
    off_pattern_entries=red,
    passed=all(v["remainder"] == "0" for v in red.values()),
    groebner_basis=[str(g) for g in G.exprs])
print("(f) rational inverse: remainders", {k: v["remainder"] for k, v in red.items()})


out["runtime_seconds"] = time.time() - t0
RESULTS.mkdir(exist_ok=True)
(RESULTS / "verma_constraint.json").write_text(json.dumps(out, indent=1))
print("done in", round(time.time() - t0, 1), "s")

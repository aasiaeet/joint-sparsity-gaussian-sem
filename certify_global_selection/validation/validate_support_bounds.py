"""Validation of the certified support bounds (support_bounds.py).

1. Closed-form profile identities: for random S and random B on every support, the
   factor formulas of support_bounds.py (Gram identity n^T adj(S) n, rho^2 form for |U| = 2)
   reproduce 1/2 log det(inner_omega(R, U)) + 3/2 and _profile of global_selection.py.
2. Known minima: (i) for every saturated-class support and random S whose fit attains
   1/2 log det S + 3/2, the certified lower bound equals it; (ii) for DAG supports with the
   regression MLE inside the box the lower bound equals the regression value; (iii) a
   hand-built S with a known Gram-factor minimum on a bowed support.
3. Brute-force grids: on 3 subset samples, for every support with |D| >= 1, the profile is
   evaluated on a grid of [-2, 2]^|D| (201 / 121^2 / 41^3 points) followed by a local
   polish from the best grid point; the certified lower bound must not exceed the grid
   minimum (up to 1e-12), and the grid minimum must not be below the stored value by more
   than 1e-8 unless the support is spectrally constrained (then the grid point's Omega
   must leave the spectral box).
4. Random S: 20 random sample covariances, all 200 supports, LB <= fitted value and, where
   the fit is profile-feasible, LB within 1e-8.
Writes results/validation/support_bounds.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "global_selection"))
import global_selection as gs  # noqa: E402
from support_bounds import SampleIA, certified_upper, profile_value, support_lower  # noqa: E402

rng = np.random.default_rng(31415)
out = {}
t0 = time.monotonic()


def Bmat(sup, x):
    B = np.zeros((3, 3))
    for v, e in zip(x, sup.D):
        B[e] = v
    return B


# ---------------------------------------------------------------- 1. identities
maxrel = 0.0
for _ in range(30):
    S = gs.random_S(rng)
    for sup in gs.SUPPORTS:
        x = rng.uniform(-2, 2, len(sup.D))
        B = Bmat(sup, x)
        A = np.eye(3) - B
        R = A.T @ S @ A
        O = gs.inner_omega(R, sup.U)
        v1 = 0.5 * np.linalg.slogdet(O)[1] + 1.5
        v2 = profile_value(S, sup, B)
        v3 = gs._profile(x, S, sup.D, sup.U)[0] if sup.D else gs.loss(B, O, S)
        if len(sup.U) == 1:
            (i, j), = sup.U
            k = 3 - i - j
            xx = B[k, i] + B[j, i] * B[k, j]
            yy = B[k, j] + B[i, j] * B[k, i]
            n = np.array([xx, yy, 1.0])
            perm = [i, j, k]
            adjS = np.linalg.det(S) * np.linalg.inv(S)
            gram = n @ adjS[np.ix_(perm, perm)] @ n
            v4 = 0.5 * (np.log(gram) + np.log(R[k, k])) + 1.5
            maxrel = max(maxrel, abs(v4 - v1))
        maxrel = max(maxrel, abs(v1 - v2), abs(v1 - v3))
out["identity_max_abs_diff"] = maxrel

# ---------------------------------------------------------------- 2. known minima
classes = gs.all_classes()
sat_idx = [s.index for s in gs.SUPPORTS if classes[s.index]["dim"] == 6]
sat_checked = sat_max = 0
dag_checked = 0
dag_max = 0.0
for r in range(8):
    S = gs.random_S(rng)
    sia = SampleIA(S)
    fsat = 0.5 * np.linalg.slogdet(S)[1] + 1.5
    srng = np.random.default_rng(r)
    for s in gs.SUPPORTS:
        if s.index in sat_idx:
            fit = gs.solve_support(S, s, srng)
            if abs(fit.value - fsat) < 1e-10:
                lb, _ = support_lower(sia, s)
                lb = max(lb, float(sia.fsat.lo))
                sat_checked += 1
                sat_max = max(sat_max, abs(lb - fsat))
        if s.D and not s.U:
            val, B, O = gs.regression_mle(S, s.D)
            if np.max(np.abs(B)) <= 2:
                lb, _ = support_lower(sia, s)
                dag_checked += 1
                dag_max = max(dag_max, abs(lb - val))
out["saturated_supports_checked"] = sat_checked
out["saturated_max_abs_lb_minus_known"] = sat_max
out["dag_supports_checked"] = dag_checked
out["dag_max_abs_lb_minus_regression"] = dag_max

# hand-built: S = I, support 1->2, 1<->2, 3->1 (bowed): Gram(a_1, a_2) * R_33.
# With S = I, R_33 = 1 and Gram = |a_1 x a_2|^2 = x^2 + y^2 + 1 >= 1, minimum 1 at x = y = 0
# (B_31 = 0); so l_min = 1/2 log 1 + 3/2 = 1.5 exactly, attained on the flat line B_31 = 0.
sup = gs.SUPPORTS[gs.NAME_TO_INDEX["1->2,3->1,1<->2"]]
sia = SampleIA(np.eye(3))
lb, info = support_lower(sia, sup)
out["handbuilt_identity_S_bowed_lb"] = lb
out["handbuilt_identity_S_bowed_known"] = 1.5
# second hand-built case with a box-constrained minimum: S with strong 1-3 correlation,
# support 3->1, 1<->2 (Gram factor with x = B_31 only, y = 0): min over |x| <= 2 of the
# quadratic n^T adj(S) n has closed form; compare
S2 = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.1], [0.9, 0.1, 1.0]])
sup2 = gs.SUPPORTS[gs.NAME_TO_INDEX["3->1,1<->2"]]
sia2 = SampleIA(S2)
lb2, _ = support_lower(sia2, sup2)
adj2 = np.linalg.det(S2) * np.linalg.inv(S2)
perm = [0, 1, 2]
M = adj2
# q(x) = [x, 0, 1] M [x, 0, 1]^T, minimizer x* = -M[0,2]/M[0,0], clipped to [-2, 2]
xs = float(np.clip(-M[0, 2] / M[0, 0], -2, 2))
q = M[0, 0] * xs ** 2 + 2 * M[0, 2] * xs + M[2, 2]
known2 = 0.5 * (np.log(q) + np.log(S2[2, 2])) + 1.5
out["handbuilt_box_qp_lb"] = lb2
out["handbuilt_box_qp_known"] = known2
out["handbuilt_box_qp_abs_diff"] = abs(lb2 - known2)

# ---------------------------------------------------------------- 3. brute-force grids
z = np.load(ROOT / "results" / "support_refits.npz")
grid_rows = []
NG = {1: 201, 2: 121, 3: 41}
for kk in [0, 36, 60]:
    S = z["S"][kk]
    sia = SampleIA(S)
    for s in gs.SUPPORTS:
        if not s.D:
            continue
        d = len(s.D)
        g1 = np.linspace(-2, 2, NG[d])
        mesh = np.stack(np.meshgrid(*([g1] * d), indexing="ij"), -1).reshape(-1, d)
        vals = np.array([profile_value(S, s, Bmat(s, x)) for x in mesh])
        j = int(np.argmin(vals))
        r = minimize(lambda x: profile_value(S, s, Bmat(s, np.clip(x, -2, 2))), mesh[j], method="L-BFGS-B",
                     bounds=[(-2, 2)] * d, options=dict(ftol=1e-15, gtol=1e-12))
        gmin = min(vals[j], float(r.fun))
        xbest = mesh[j] if vals[j] <= r.fun else np.clip(r.x, -2, 2)
        lb, _ = support_lower(sia, s)
        lb = max(lb, float(sia.fsat.lo))
        found = float(z["npz_value"][kk][s.index])
        Bb = Bmat(s, xbest)
        A = np.eye(3) - Bb
        Ob = gs.inner_omega(A.T @ S @ A, s.U)
        ev = np.linalg.eigvalsh(Ob)
        grid_rows.append(dict(sample=kk, support=s.name, lb=lb, grid_min=gmin, found=found,
                              lb_minus_grid=lb - gmin, found_minus_grid=found - gmin,
                              grid_point_spectral_ok=bool(ev[0] >= 0.05 and ev[-1] <= 10)))
lbviol = [r for r in grid_rows if r["lb_minus_grid"] > 1e-12]
fviol = [r for r in grid_rows if r["found_minus_grid"] > 1e-8 and r["grid_point_spectral_ok"]]
out["grid_supports_checked"] = len(grid_rows)
out["grid_lb_above_grid_min"] = len(lbviol)
out["grid_max_lb_minus_grid"] = max(r["lb_minus_grid"] for r in grid_rows)
out["grid_fit_above_feasible_grid_min_by_1e-8"] = len(fviol)
out["grid_max_grid_minus_lb"] = max(-r["lb_minus_grid"] for r in grid_rows)
out["grid_spectral_cases"] = [r for r in grid_rows if r["found_minus_grid"] > 1e-8]

# ---------------------------------------------------------------- 4. random S
rs_rows = 0
rs_bad = 0
rs_uncert = []
for r in range(20):
    S = gs.random_S(rng)
    sia = SampleIA(S)
    srng = np.random.default_rng(1000 + r)
    fits, _ = gs.lmin_all(S, srng)
    for s, f in zip(gs.SUPPORTS, fits):
        lb, _ = support_lower(sia, s, skip_rho2=f.value <= float(sia.fsat.hi) + 1e-12)
        lb = max(lb, float(sia.fsat.lo))
        ub, _ = certified_upper(sia, s, f.B, f.O)
        rs_rows += 1
        if lb > f.value + 1e-12 or ub < lb:
            rs_bad += 1
        if f.value - lb > 1e-8 or ub - f.value > 1e-8:
            rs_uncert.append(dict(S=r, support=s.name, method=f.method, gap=f.value - lb, ubgap=ub - f.value))
out["randomS_fits"] = rs_rows
out["randomS_lb_above_value_or_ub"] = rs_bad
out["randomS_uncertified"] = len(rs_uncert)
out["randomS_uncertified_methods"] = sorted({u["method"] for u in rs_uncert})
out["randomS_uncertified_examples"] = rs_uncert[:10]
out["runtime_s"] = time.monotonic() - t0
print(json.dumps(out, indent=1, default=float))
(ROOT / "results" / "validation" / "support_bounds.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

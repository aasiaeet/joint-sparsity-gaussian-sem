"""Validation of ia.py against mpmath (60 significant digits).

Checks that every computed interval contains the exact result, for random inputs:
basic operations, sqr, sqrt, the rigorous log (including extreme magnitudes and values
near powers of two and 1), det3/adj3 of random 3x3 point matrices, and the interval
Cholesky PD test (agreement with eigenvalues away from the boundary).
Writes results/validation/test_ia.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from pathlib import Path

import mpmath
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "global_selection"))
from ia import IA, adj3, cholesky_pd, det3  # noqa: E402

mpmath.mp.dps = 60
rng = np.random.default_rng(20261022)
out = {}


def contains(I, exact):
    return mpmath.mpf(float(I[0])) <= exact <= mpmath.mpf(float(I[1]))


# basic ops on point intervals (worst case for rounding) and on wide intervals
N = 20000
a = rng.standard_normal(N) * 10.0 ** rng.integers(-8, 8, N)
b = rng.standard_normal(N) * 10.0 ** rng.integers(-8, 8, N)
A, B = IA.point(a), IA.point(b)
fails = 0
for name, I, f in [("add", A + B, lambda x, y: x + y), ("sub", A - B, lambda x, y: x - y),
                   ("mul", A * B, lambda x, y: x * y), ("div", A / B, lambda x, y: x / y)]:
    bad = 0
    for k in range(0, N, 7):
        ex = f(mpmath.mpf(float(a[k])), mpmath.mpf(float(b[k])))
        bad += not contains((I.lo[k], I.hi[k]), ex)
    out[f"point_{name}_failures"] = bad
    fails += bad
# sqr and sqrt
Q = A.sqr()
out["sqr_failures"] = sum(not contains((Q.lo[k], Q.hi[k]), mpmath.mpf(float(a[k])) ** 2) for k in range(0, N, 7))
pa = np.abs(a)
R = IA.point(pa).sqrt()
out["sqrt_failures"] = sum(not contains((R.lo[k], R.hi[k]), mpmath.sqrt(mpmath.mpf(float(pa[k])))) for k in range(0, N, 7))
# wide-interval multiplication: sample points inside must map inside
lo = rng.standard_normal(N); hi = lo + np.abs(rng.standard_normal(N))
lo2 = rng.standard_normal(N); hi2 = lo2 + np.abs(rng.standard_normal(N))
W = IA(lo, hi) * IA(lo2, hi2)
u = lo + rng.uniform(size=N) * (hi - lo); v = lo2 + rng.uniform(size=N) * (hi2 - lo2)
out["wide_mul_sample_failures"] = int(np.sum((u * v < W.lo) | (u * v > W.hi)))

# rigorous log
xs = np.concatenate([
    np.exp(rng.uniform(-700, 700, 4000)),
    1.0 + rng.uniform(-1e-6, 1e-6, 1000),
    np.array([1.0, 2.0, 0.5, 4.0, 0.7071067811865476, 0.7071067811865475, 1.4142135623730951,
              1.4142135623730950, np.nextafter(1.0, 0), np.nextafter(1.0, 2), 1e-300, 1e300,
              5e-324, 1.7976931348623157e308]),
    rng.uniform(0.01, 100, 4000),
])
L = IA.point(xs).log()
logfail = 0
maxrel = 0.0
for k in range(len(xs)):
    ex = mpmath.log(mpmath.mpf(float(xs[k])))
    logfail += not contains((L.lo[k], L.hi[k]), ex)
    if ex != 0:
        maxrel = max(maxrel, float((mpmath.mpf(float(L.hi[k])) - mpmath.mpf(float(L.lo[k]))) / abs(ex)))
out["log_failures"] = logfail
out["log_n"] = int(len(xs))
out["log_max_relative_width"] = maxrel

# det3 / adj3 on random point matrices
detfail = adjfail = 0
for _ in range(300):
    M = rng.standard_normal((3, 3)) * 10.0 ** rng.integers(-2, 3)
    I = IA.point(M)
    d = det3(I)
    Mm = mpmath.matrix(M.tolist())
    detfail += not contains((d.lo, d.hi), mpmath.det(Mm))
    Ad = adj3(I)
    exA = mpmath.det(Mm) * mpmath.inverse(Mm)
    for i in range(3):
        for j in range(3):
            adjfail += not contains((Ad.lo[i, j], Ad.hi[i, j]), exA[i, j])
out["det3_failures"] = detfail
out["adj3_entry_failures"] = adjfail

# Cholesky PD certification versus eigenvalues
agree = disagree_unsafe = 0
for _ in range(2000):
    G = rng.standard_normal((3, 3))
    M = G @ G.T + rng.uniform(-1.5, 1.0) * np.eye(3)
    M = (M + M.T) / 2
    r = 10.0 ** rng.uniform(-6, -1)
    I = IA(M - r, M + r)
    ok = bool(cholesky_pd(I))
    # every symmetric matrix in the box PD  <=  lambda_min(M) > 3 r (Weyl bound on 3x3 with |E_ij| <= r)
    ev = np.linalg.eigvalsh(M)[0]
    if ok and ev - 3 * r < 0:
        # check a random symmetric perturbation is still PD (certification must be safe)
        for _ in range(200):
            E = rng.uniform(-r, r, (3, 3)); E = np.triu(E) + np.triu(E, 1).T
            if np.linalg.eigvalsh(M + E)[0] <= 0:
                disagree_unsafe += 1
                break
    agree += ok == (ev > 3 * r)
out["cholesky_cases"] = 2000
out["cholesky_agree_with_weyl_sufficient"] = agree
out["cholesky_unsafe_certifications"] = disagree_unsafe

out["all_enclosures_ok"] = all(v == 0 for k, v in out.items() if k.endswith("failures")) and disagree_unsafe == 0
out = {k: (v.item() if hasattr(v, "item") else v) for k, v in out.items()}
print(json.dumps(out, indent=1))
(ROOT / "results" / "validation" / "test_ia.json").write_text(json.dumps(out, indent=1) + "\n")

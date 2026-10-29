"""The vanishing ideal of the model of G* in low degree (supplements verma_symbolic.py).

 - g = s12 s13 s34 - s12 s14 s33 - s13^2 s24 + s13 s14 s23 (the Verma cubic with the
   instrument formula lambda23 = s13/s12 in place of s23/s22) vanishes on M_G* (sympy);
 - graded pieces: dim I(V_G*)_d for d = 2..5 by exact interpolation over GF(q),
   q = 67108859 < 2^26 (all products exact in float64), evaluated at random
   G*-parameters over GF(q); compared with <f_CI, f_V>_d and <f_CI, f_V, g>_d;
 - saturations <f_CI, f_V> : s^infinity for s = s22, s13, s11 (sympy Groebner, lex).
Writes results/verma_ideal.json.  Degree 5 takes about nine minutes on one core;
--max-degree 4 stops earlier with the same values for degrees 2 to 4.
Usage: python3 verma_ideal.py [--max-degree D]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import sympy as sp

from common import MASTER_SEED, RESULTS, STUDY_ID

ap = argparse.ArgumentParser()
ap.add_argument("--max-degree", type=int, default=5)
ARGS = ap.parse_args()
t0 = time.time()
Q = 67108859
sym = sp.symbols("s11 s12 s13 s14 s22 s23 s24 s33 s34 s44")
s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = sym
IU = [(k, l) for k in range(4) for l in range(k, 4)]
F_CI = s13 * s22 - s12 * s23
F_V = s13 * s22 * s34 - s14 * s22 * s33 - s13 * s23 * s24 + s14 * s23 ** 2
G_V = s12 * s13 * s34 - s12 * s14 * s33 - s13 ** 2 * s24 + s13 * s14 * s23
out = dict(f_CI=str(F_CI), f_V=str(F_V), g=str(G_V))

# g on the model
a, b, c, w1, w2, w3, w4, w24 = sp.symbols("a b c w1 w2 w3 w4 w24")
Bs = sp.zeros(4)
Bs[0, 1], Bs[1, 2], Bs[2, 3] = a, b, c
Os = sp.diag(w1, w2, w3, w4)
Os[1, 3] = Os[3, 1] = w24
Cs = (sp.eye(4) - Bs).inv()
Ss = sp.expand(Cs.T * Os * Cs)
subs = {sym[k]: Ss[i, j] for k, (i, j) in enumerate(IU)}
out["g_on_model"] = str(sp.expand(G_V.subs(subs, simultaneous=True)))
out["g_identity"] = "s12 * f_V - s22 * g = " + str(sp.factor(sp.expand(s12 * F_V - s22 * G_V)))
print("g on model:", out["g_on_model"], "|", out["g_identity"])


def monomials(d):
    return list(itertools.combinations_with_replacement(range(10), d))


def rank_modq(M):
    M = np.array(M, dtype=np.float64) % Q
    r = 0
    nrow, ncol = M.shape
    for cidx in range(ncol):
        nz = np.nonzero(M[r:, cidx])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, cidx]), Q - 2, Q)
        M[r] = np.fmod(M[r] * inv, Q)
        f = M[:, cidx].copy()
        f[r] = 0
        nzr = np.nonzero(f)[0]
        if len(nzr):
            M[nzr] = np.mod(M[nzr] - np.fmod(np.outer(f[nzr], M[r]), Q), Q)
        r += 1
        if r == nrow:
            break
    return r


def exps_to_index(d):
    idx = {}
    for k, mono in enumerate(monomials(d)):
        e = [0] * 10
        for v in mono:
            e[v] += 1
        idx[tuple(e)] = k
    return idx


def ideal_piece_rank(gens, d):
    idx = exps_to_index(d)
    rows = []
    for g in gens:
        gd = {m: int(cf) for m, cf in sp.Poly(g, *sym).terms()}
        dg = sum(next(iter(gd)))
        if dg > d:
            continue
        for mono in monomials(d - dg):
            e = [0] * 10
            for v in mono:
                e[v] += 1
            row = np.zeros(len(idx))
            for ge, cf in gd.items():
                row[idx[tuple(x + y for x, y in zip(ge, e))]] = cf % Q
            rows.append(row)
    return rank_modq(rows) if rows else 0


def random_sigma_modq(rng):
    B = np.zeros((4, 4), dtype=object)
    B[0, 1], B[1, 2], B[2, 3] = (int(x) for x in rng.integers(1, Q, 3))
    O = np.zeros((4, 4), dtype=object)
    for i in range(4):
        O[i, i] = int(rng.integers(1, Q))
    O[1, 3] = O[3, 1] = int(rng.integers(1, Q))
    I = np.eye(4, dtype=int).astype(object)
    B2 = (B.dot(B)) % Q
    B3 = (B2.dot(B)) % Q
    C = (I + B + B2 + B3) % Q
    S = (C.T.dot(O).dot(C)) % Q
    return [int(S[i, j]) for (i, j) in IU]


def check_point(v):
    s = dict(zip(sym, v))
    return [int(sp.Integer(f.subs(s)) % Q) for f in (F_CI, F_V, G_V)]


rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, 0, 11]))
print("sanity (f_CI, f_V, g mod q at a model point):", check_point(random_sigma_modq(rng)))
graded = []
for d in range(2, ARGS.max_degree + 1):
    t1 = time.time()
    mons = monomials(d)
    npts = len(mons) + 40
    pts = np.array([random_sigma_modq(rng) for _ in range(npts)], dtype=object)
    M = np.empty((npts, len(mons)), dtype=np.float64)
    for k, mono in enumerate(mons):
        col = np.ones(npts, dtype=object)
        for v in mono:
            col = (col * pts[:, v]) % Q
        M[:, k] = col.astype(np.float64)
    dim_I = len(mons) - rank_modq(M)
    d2 = ideal_piece_rank([F_CI, F_V], d)
    d3 = ideal_piece_rank([F_CI, F_V, G_V], d)
    graded.append(dict(degree=d, n_monomials=len(mons), n_points=npts, dim_I_d=dim_I,
                       dim_fCI_fV=d2, dim_fCI_fV_g=d3, seconds=time.time() - t1))
    print(graded[-1], flush=True)
out["graded"] = dict(prime_q=Q, table=graded)

sats = {}
for name, var in [("s22", s22), ("s13", s13), ("s11", s11)]:
    t1 = time.time()
    tt = sp.symbols("t")
    Gs = sp.groebner([F_CI, F_V, 1 - tt * var], tt, *sym, order="lex")
    gens = [g for g in Gs.exprs if tt not in g.free_symbols]
    sats[name] = dict(generators=[str(sp.factor(g)) for g in gens], seconds=time.time() - t1)
    print("saturation by", name, ":", sats[name], flush=True)
out["saturations"] = sats
out["runtime_seconds"] = time.time() - t0
(RESULTS / "verma_ideal.json").write_text(json.dumps(out, indent=1))
print("done in", round(time.time() - t0, 1), "s")

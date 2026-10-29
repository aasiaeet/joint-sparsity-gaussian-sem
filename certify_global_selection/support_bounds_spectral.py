"""Spectrally constrained profile over Omega for supports with |U| <= 1.

For |U| <= 1 the admissible Omega is block diagonal (blocks {i, j} and {k}, or three 1x1
blocks), the spectral constraint 0.05 I <= Omega <= 10 I acts on each block separately and
the loss separates over blocks.  For one block with data block R_b, in K = Omega_b^{-1},
-1/2 log det K + 1/2 tr(K R_b) >= sum_i [-1/2 log kappa_i + 1/2 kappa_i mu_i] (von Neumann:
tr(K R_b) >= sum kappa_i^desc mu_i^asc), minimized per eigenvalue over kappa in [0.1, 20]; the
bound is attained by K sharing eigenvectors with R_b.  Hence the exact constrained profile is
    f_c(B) = 1/2 sum_{mu in eig(R_blocks(B))} psi(mu),   psi(mu) = log c(mu) + mu / c(mu),
c(mu) = min(max(mu, 0.05), 10).  psi is increasing, concave and C^1 with psi'(mu) = 1/c(mu).
The minimum of f_c over |B_e| <= 2 is bounded below by branch and bound with the
monotonicity-aware mean-value form; eigenvalues of the 2x2 block are
mu = m +- r, m = (a + d)/2, r = sqrt(((a - d)/2)^2 + b^2), with interval derivatives.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ia import IA, dn, up  # noqa: E402
from support_bounds import MB, bnb  # noqa: E402
from criterion_bounds import imatmul, intersect, itranspose  # noqa: E402

LOG10 = IA.point(np.array(10.0)).log()
LOG005 = IA.point(np.array(0.05)).log()


def psi_bounds(mu):
    """Interval enclosure of psi over the interval mu (psi increasing)."""
    def at(x, side):
        X = IA.point(x)
        v_hi = LOG10 + X / 10.0
        v_mid = X.log() if np.all(x > 0) else IA.point(np.full_like(x, -np.inf))
        v_mid = v_mid + 1.0
        v_lo = LOG005 + X / 0.05
        lo = np.where(x >= 10, v_hi.lo, np.where(x >= 0.05, v_mid.lo, v_lo.lo))
        hi = np.where(x >= 10, v_hi.hi, np.where(x >= 0.05, v_mid.hi, v_lo.hi))
        return lo if side == "lo" else hi
    lo = at(np.maximum(mu.lo, 1e-300), "lo")
    hi = at(np.maximum(mu.hi, 1e-300), "hi")
    return IA(lo, hi)


def dpsi_bounds(mu):
    c_lo = np.clip(mu.lo, 0.05, 10.0)
    c_hi = np.clip(mu.hi, 0.05, 10.0)
    return IA(dn(1.0 / c_hi), up(1.0 / c_lo))


def make_fc(S, sup):
    D = list(sup.D)
    U = list(sup.U)
    d = len(D)

    def build(lo, hi):
        K = lo.shape[0]
        Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
        for n_, (i, j) in enumerate(D):
            Blo[:, i, j] = lo[:, n_]; Bhi[:, i, j] = hi[:, n_]
        I3 = np.broadcast_to(np.eye(3), (K, 3, 3))
        A = IA(I3 - Bhi, I3 - Blo)
        Sb = IA.point(np.broadcast_to(S, (K, 3, 3)).copy())
        SA = imatmul(Sb, A)
        R = imatmul(itranspose(A), SA)
        R = intersect(R, itranspose(R))
        return SA, R

    def dR(SA, a, b, e):
        p, q = e
        out = None
        if a == q:
            out = -SA[:, p, b]
        if b == q:
            t = -SA[:, p, a]
            out = t if out is None else out + t
        return out

    def blocks():
        if len(U) == 1:
            (i, j), = U
            k = 3 - i - j
            return [(i, j)], [k]
        return [], [0, 1, 2]

    pairs, singles = blocks()

    def value_and_grad(lo, hi, want_grad=True):
        SA, R = build(lo, hi)
        K = lo.shape[0]
        val = IA.point(np.zeros(K))
        grads = [IA.point(np.zeros(K)) for _ in range(d)]
        for v in singles:
            mu = R[:, v, v]
            val = val + psi_bounds(mu) * 0.5
            if want_grad:
                dp = dpsi_bounds(mu)
                for n_, e in enumerate(D):
                    g = dR(SA, v, v, e)
                    if g is not None:
                        grads[n_] = grads[n_] + dp * g * 0.5
        for (i, j) in pairs:
            a, dd, b = R[:, i, i], R[:, j, j], R[:, i, j]
            m = (a + dd) * 0.5
            hdiff = (a - dd) * 0.5
            r2 = hdiff.sqr() + b.sqr()
            r2 = IA(np.maximum(r2.lo, 0.0), r2.hi)
            r = r2.sqrt()
            mup, mum = m + r, m - r
            val = val + (psi_bounds(mup) + psi_bounds(mum)) * 0.5
            if want_grad:
                okr = r.lo > 0
                rs = IA(np.where(okr, r.lo, 1.0), np.where(okr, r.hi, 1.0))
                dpp, dpm = dpsi_bounds(mup), dpsi_bounds(mum)
                for n_, e in enumerate(D):
                    ga, gd, gb = dR(SA, i, i, e), dR(SA, j, j, e), dR(SA, i, j, e)
                    z = IA.point(np.zeros(K))
                    ga = z if ga is None else ga
                    gd = z if gd is None else gd
                    gb = z if gb is None else gb
                    dm = (ga + gd) * 0.5
                    dr = (hdiff * (ga - gd) * 0.5 + b * gb) / rs
                    g = (dpp * (dm + dr) + dpm * (dm - dr)) * 0.5
                    g = IA(np.where(okr, g.lo, -np.inf), np.where(okr, g.hi, np.inf))
                    grads[n_] = grads[n_] + g
        return val, grads

    def lb_fn(lo, hi):
        _, G = value_and_grad(lo, hi)
        c = 0.5 * (lo + hi)
        for n_ in range(d):
            c[:, n_] = np.where(G[n_].lo >= 0, lo[:, n_], np.where(G[n_].hi <= 0, hi[:, n_], c[:, n_]))
        v, _ = value_and_grad(c, c, want_grad=False)
        tot = v
        for n_ in range(d):
            tot = tot + G[n_] * (IA(lo[:, n_], hi[:, n_]) - IA.point(c[:, n_]))
        nat, _ = value_and_grad(lo, hi, want_grad=False)
        out = np.maximum(tot.lo, nat.lo)
        return np.where(np.isfinite(out), out, -np.inf)

    def ub_fn(lo, hi):
        c = 0.5 * (lo + hi)
        v, _ = value_and_grad(c, c, want_grad=False)
        return v.hi

    return lb_fn, ub_fn, d


def spectral_lower(S, sup, ub0=math.inf, max_boxes=2_000_000):
    assert len(sup.U) <= 1 and len(sup.D) >= 1
    lb_fn, ub_fn, d = make_fc(np.asarray(S, float), sup)
    res = bnb(lb_fn, ub_fn, np.full(d, -MB), np.full(d, MB), rtol=1e-13, atol=1e-13,
              max_boxes=max_boxes, max_iter=400, ub0=ub0)
    return res

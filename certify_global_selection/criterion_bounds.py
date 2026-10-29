"""Certified lower bound of the quasi-MCP criterion over K, region by region (Section S18).

F(theta) = l_S(B, Omega) + lambda * [sum_{i != j} q_delta(B_ij) + sum_{i<j} q_delta(Omega_ij)],
lambda = delta = log(n)/n, over K = {B acyclic, |B_ij| <= 2, 0.05 I <= Omega <= 10 I}.

Decomposition.  Every acyclic B is supported on the forward pairs of some vertex order pi
(6 orders).  For pi, the 6 penalized coordinates are c0..c2 = B on the forward pairs
(pi0,pi1), (pi0,pi2), (pi1,pi2) and c3..c5 = Omega_12, Omega_13, Omega_23.  For a set L of
coordinates, region R(pi, L) = {theta: |theta_k| > delta' for k in L, |theta_k| <= delta' for
k not in L}, delta' = upper end of the interval enclosure of log(n)/n (>= delta).  The union
over (pi, L) covers K.  On R(pi, L), P_delta = |L| + sum_{k not in L} q_delta(theta_k).

Screen.  F >= l_S + lambda |L| >= 1/2 log det S + 3/2 + lambda |L|  (Fischer).

Branch and bound (remaining regions).  Variables x = (c0..c5, Omega_11, Omega_22, Omega_33);
u = coordinates in L plus the diagonal (ranges [-2,2], [-10,10], [0.05,10]); t = coordinates
not in L (range [-delta', delta']).  On a box X = X_u x X_t, with t0 the point of X_t closest to 0:
  F(u, t) >= lambda |L| + LB[l(u, t0)] + sum_{k in t} min_{t_k in X_k} [lambda q(t_k) + g_k (t_k - t0_k)],
g_k ranging over an interval enclosure of dl/dx_k over X (mean value theorem; the
per-coordinate minimum is attained at an endpoint of X_k or at 0 because lambda q is
concave on each side of 0).  LB[l(u, t0)] is the larger of
  (i) the mean-value (monotonicity-aware) form of l over X_u at t = t0, and
  (ii) if t0 = 0: the certified profile lower bound of support H(pi, L) from support_bounds.py
       (valid because (u, 0) has support in H(pi, L) and Omega positive definite).
Omega^{-1} is enclosed over X by a Neumann-series bound around the inverse of the center
(which also certifies that every Omega in X is nonsingular, hence positive definite).
Boxes are discarded when they are certified outside K (spectral box) or certified to have
l_S >= s* - lambda |L| through the generalized Rayleigh quotients v^T Omega v / v^T R v.
A region is certified when every box has bound >= s* - 1e-8.
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ia import IA, adj3, cholesky_pd, det3, dn, up  # noqa: E402

P = 3
MB = 2.0
OLO, OHI = 0.05, 10.0
PAIRS = [(0, 1), (0, 2), (1, 2)]
TOL = 1e-8


# ================================================================ IA helpers on (K, 3, 3)
def imatmul(A, B):
    """IA (K,3,3) @ IA (K,3,3) with explicit outward-rounded sums."""
    terms = []
    for k in range(3):
        terms.append(IA(A.lo[:, :, k:k + 1], A.hi[:, :, k:k + 1]) * IA(B.lo[:, k:k + 1, :], B.hi[:, k:k + 1, :]))
    return terms[0] + terms[1] + terms[2]


def itranspose(A):
    return IA(np.swapaxes(A.lo, 1, 2), np.swapaxes(A.hi, 1, 2))


def intersect(a, b):
    return IA(np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def qlb(tabs_lo, delta):
    """Lower bound of q_delta at |t| >= tabs_lo (q nondecreasing in |t|).  delta: IA scalar."""
    s = (IA.point(tabs_lo) / delta).lo
    s = np.minimum(np.maximum(s, 0.0), 1.0)
    S_ = IA.point(s)
    val = (S_ * 2.0 - S_.sqr()).lo
    return np.where(s >= 1.0, 1.0, np.maximum(val, 0.0))


class Region:
    def __init__(self, ctx, order, L):
        self.ctx = ctx
        self.order = order
        self.L = tuple(L)
        self.fwd = [(order[0], order[1]), (order[0], order[2]), (order[1], order[2])]
        self.tdims = [c for c in range(6) if c not in L]
        self.udims = [c for c in range(6) if c in L] + [6, 7, 8]
        # support H(pi, L)
        self.D = tuple(sorted(self.fwd[c] for c in L if c < 3))
        self.U = tuple(sorted(PAIRS[c - 3] for c in L if c >= 3))

    def initial_box(self):
        dp = self.ctx.delta_hi
        lo = np.empty(9); hi = np.empty(9)
        for c in range(6):
            if c in self.L:
                b = MB if c < 3 else OHI
                lo[c], hi[c] = -b, b
            else:
                lo[c], hi[c] = -dp, dp
        lo[6:] = OLO
        hi[6:] = OHI
        return lo, hi


class Ctx:
    """Per-sample constants."""

    def __init__(self, S, n, s_star, lb_prof_by_support):
        self.S = np.asarray(S, float)
        self.n = n
        lam = IA.point(np.array(float(n))).log() / IA.point(np.array(float(n)))
        self.lam = lam
        self.delta = lam
        self.delta_hi = float(lam.hi)
        self.s_star = float(s_star)
        self.target = self.s_star - TOL
        Si = IA.point(self.S)
        self.Si = Si
        self.logdetS = det3(Si).log()
        self.fsat = (self.logdetS * 0.5 + 1.5)
        self.lb_prof = lb_prof_by_support   # dict (D, U) -> float

    # ---------------------------------------------------------- Rayleigh-ratio bounds
    def mu_bounds(self, tau):
        """Certified [mu_lo, mu_hi] such that any Omega with some generalized eigenvalue
        (relative to R) outside it has l_S - fsat > tau."""
        if tau <= 0:
            return 1.0, 1.0
        from scipy.optimize import brentq
        h = lambda m: math.log(m) + 1.0 / m - 1.0 - 2 * tau
        mlo = brentq(h, 1e-12, 1.0, xtol=1e-15)
        mhi = brentq(h, 1.0, 1e8, xtol=1e-12)

        def hcert(m):
            M = IA.point(np.array(m))
            return float((M.log() + IA.point(np.array(1.0)) / M - 1.0).lo)
        while hcert(mlo) < 2 * tau:
            mlo *= 0.999
        while hcert(mhi) < 2 * tau:
            mhi *= 1.001
        return mlo, mhi


def pair_prune(R, Opq, p, q, mu_lo, mu_hi):
    """True where no Omega with Omega_pq in the interval Opq can satisfy
    mu_lo R <= Omega <= mu_hi R (for the R interval matrices of the boxes).
    With v+- = e_p +- s e_q (any float s > 0): mu_lo u+- <= w+- <= mu_hi u+-, where
    u+- = v+-^T R v+-, w+- = v+-^T Omega v+-, and w+ - w- = 4 s Omega_pq."""
    Rm = 0.5 * (R.lo + R.hi)
    s = np.sqrt(np.maximum(Rm[:, p, p], 1e-12) / np.maximum(Rm[:, q, q], 1e-12))
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
    Si = IA.point(s)
    up_ = R[:, p, p] + Si * R[:, p, q] * 2.0 + Si.sqr() * R[:, q, q]
    um_ = R[:, p, p] - Si * R[:, p, q] * 2.0 + Si.sqr() * R[:, q, q]
    lo_b = up_ * mu_lo - um_ * mu_hi
    hi_b = up_ * mu_hi - um_ * mu_lo
    d = Opq * Si * 4.0
    return (d.hi < lo_b.lo) | (d.lo > hi_b.hi)


def neumann_inverse(Olo, Ohi):
    """Enclose Omega^{-1} for all symmetric Omega in the interval matrix [Olo, Ohi] (K,3,3).
    Returns (ok (K,), W IA (K,3,3)).  ok certifies nonsingularity throughout the box."""
    K = Olo.shape[0]
    Oc = 0.5 * Olo + 0.5 * Ohi
    try:
        C = np.linalg.inv(Oc)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(Oc)
    bad = ~np.all(np.isfinite(C), axis=(1, 2))
    C[bad] = 0.0
    Ci = IA.point(C)
    E = IA.point(np.broadcast_to(np.eye(3), (K, 3, 3)).copy()) - imatmul(Ci, IA(Olo, Ohi))
    rows = E.mag()
    nrm = up(up(rows[:, :, 0] + rows[:, :, 1]) + rows[:, :, 2])
    nrm = np.max(nrm, axis=1)
    ok = (nrm < 0.9) & ~bad
    nrm_s = np.where(ok, nrm, 0.5)
    eta = up(up(nrm_s * nrm_s) / dn(1.0 - nrm_s))
    eta = up(eta)
    IpE = E + IA.point(np.broadcast_to(np.eye(3), (K, 3, 3)).copy())
    IpE = IA(dn(IpE.lo - eta[:, None, None]), up(IpE.hi + eta[:, None, None]))
    W = imatmul(IpE, Ci)
    # symmetric: intersect W with its transpose
    W = intersect(W, itranspose(W))
    return ok, W


def build_BO(region, lo, hi):
    K = lo.shape[0]
    Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(region.fwd):
        Blo[:, i, j] = lo[:, c]; Bhi[:, i, j] = hi[:, c]
    Olo = np.zeros((K, 3, 3)); Ohi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(PAIRS):
        Olo[:, i, j] = Olo[:, j, i] = lo[:, 3 + c]
        Ohi[:, i, j] = Ohi[:, j, i] = hi[:, 3 + c]
    for v in range(3):
        Olo[:, v, v] = lo[:, 6 + v]; Ohi[:, v, v] = hi[:, 6 + v]
    return Blo, Bhi, Olo, Ohi


def loss_points(ctx, region, x):
    """Interval l_S at points x (K, 9).  Requires Omega(x) positive definite."""
    K = x.shape[0]
    Blo, Bhi, Olo, Ohi = build_BO(region, x, x)
    A = IA.point(np.broadcast_to(np.eye(3), (K, 3, 3)).copy() - Blo)  # exact
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    R = imatmul(itranspose(A), imatmul(Sb, A))
    O = IA.point(Olo)
    dO = det3(O)
    adjO = adj3(O)
    tr = None
    for i in range(3):
        for j in range(3):
            t = adjO[:, i, j] * R[:, j, i]
            tr = t if tr is None else tr + t
    okdet = dO.lo > 0
    dOs = IA(np.where(okdet, dO.lo, 1.0), np.where(okdet, dO.hi, 1.0))
    val = dOs.log() * 0.5 + (tr / dOs) * 0.5
    lo_ = np.where(okdet, val.lo, -np.inf)
    return lo_, okdet


def bound_boxes(ctx, region, lo, hi, mu):
    """Rigorous lower bounds of F over boxes (K, 9).  Returns (lb, status, G mag (K,9))."""
    K = lo.shape[0]
    lam = ctx.lam
    nL = len(region.L)
    lb = np.full(K, -np.inf)
    status = np.zeros(K, dtype=np.int8)   # 0 bounded, 1 infeasible-pruned, 2 KL-pruned, 3 no inverse
    gmag = np.zeros((K, 9))
    Blo, Bhi, Olo, Ohi = build_BO(region, lo, hi)
    Oc = 0.5 * Olo + 0.5 * Ohi
    # ---------------- spectral feasibility (outside K -> discard)
    ev, V = np.linalg.eigh(Oc)
    O = IA(Olo, Ohi)
    infeas = np.zeros(K, bool)
    for col, test in [(0, "lo"), (2, "hi")]:
        v = IA.point(V[:, :, col])
        Ov = None
        for j in range(3):
            t = O[:, :, j] * v[:, j:j + 1]
            Ov = t if Ov is None else Ov + t
        q = Ov[:, 0] * v[:, 0] + Ov[:, 1] * v[:, 1] + Ov[:, 2] * v[:, 2]
        nv = v[:, 0].sqr() + v[:, 1].sqr() + v[:, 2].sqr()
        rq = q / nv
        if test == "lo":
            infeas |= rq.hi < OLO          # lambda_min <= rq < 0.05 for every Omega in box
        else:
            infeas |= rq.lo > OHI
    # 2x2 minors: |Omega_ij| > sqrt(Omega_ii Omega_jj) throughout -> not PD
    for c, (i, j) in enumerate(PAIRS):
        m2 = IA(Olo[:, i, j], Ohi[:, i, j]).sqr().lo
        pd = up(Ohi[:, i, i] * Ohi[:, j, j])
        infeas |= m2 > pd
    status[infeas] = 1
    # ---------------- generalized Rayleigh prune: l >= fsat + tau if v^T Omega v / v^T R v
    # leaves [mu_lo, mu_hi] for some v (here unit vectors and e_p +- e_q)
    A = IA(np.broadcast_to(np.eye(3), (K, 3, 3)).copy() - Bhi, np.broadcast_to(np.eye(3), (K, 3, 3)).copy() - Blo)
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    SA = imatmul(Sb, A)
    R = imatmul(itranspose(A), SA)
    mu_lo, mu_hi = mu
    klp = np.zeros(K, bool)
    vecs = [np.eye(3)[v] for v in range(3)] + [np.eye(3)[p] + s * np.eye(3)[q] for (p, q) in PAIRS for s in (1, -1)]
    for v in vecs:
        nz = np.nonzero(v)[0]

        def quad(M):
            acc = None
            for a in nz:
                for b in nz:
                    t = M[:, a, b] * float(v[a] * v[b])
                    acc = t if acc is None else acc + t
            return acc
        qo = quad(O)
        qr = quad(R)
        qr = IA(np.maximum(qr.lo, 1e-300), np.maximum(qr.hi, 1e-300))
        ratio = qo / qr
        klp |= (ratio.hi < mu_lo) | (ratio.lo > mu_hi)
    for c in region.tdims:
        if c >= 3:
            p, q = PAIRS[c - 3]
            klp |= pair_prune(R, IA(Olo[:, p, q], Ohi[:, p, q]), p, q, mu_lo, mu_hi)
    status[(status == 0) & klp] = 2
    act = status == 0
    if not np.any(act):
        return lb, status, gmag
    idx = np.nonzero(act)[0]
    # ---------------- Omega^{-1} enclosure and gradients over the full box
    ok, W = neumann_inverse(Olo[idx], Ohi[idx])
    # nonsingular throughout the (connected) box and center certified PD => all PD
    ok &= cholesky_pd(IA.point(Oc[idx]))
    status[idx[~ok]] = 3
    idx = idx[ok]
    if not len(idx):
        return lb, status, gmag
    W = W[ok]
    Ab = A[idx]
    SAb = SA[idx]
    Rb = R[idx]
    Rb = intersect(Rb, itranspose(Rb))
    GB = -imatmul(SAb, W)                       # dl/dB (all entries)
    WRW = imatmul(imatmul(W, Rb), W)
    Mg = W - WRW
    Mg = intersect(Mg, itranspose(Mg))
    Kb = len(idx)
    Glo = np.zeros((Kb, 9)); Ghi = np.zeros((Kb, 9))
    for c, (i, j) in enumerate(region.fwd):
        Glo[:, c] = GB.lo[:, i, j]; Ghi[:, c] = GB.hi[:, i, j]
    for c, (i, j) in enumerate(PAIRS):
        Glo[:, 3 + c] = Mg.lo[:, i, j]; Ghi[:, 3 + c] = Mg.hi[:, i, j]
    for v in range(3):
        g = Mg[:, v, v] * 0.5
        Glo[:, 6 + v] = g.lo; Ghi[:, 6 + v] = g.hi
    G = IA(Glo, Ghi)
    gmag[idx] = G.mag()
    lo_b, hi_b = lo[idx], hi[idx]
    # ---------------- expansion point: t0 (closest to 0) and monotone corner for u
    x0 = 0.5 * (lo_b + hi_b)
    for c in region.tdims:
        x0[:, c] = np.clip(0.0, lo_b[:, c], hi_b[:, c])
    for c in region.udims:
        x0[:, c] = np.where(Glo[:, c] >= 0, lo_b[:, c], np.where(Ghi[:, c] <= 0, hi_b[:, c], x0[:, c]))
    l0, okp = loss_points(ctx, region, x0)
    mv = IA.point(l0)
    for c in region.udims:
        mv = mv + G[:, c] * (IA(lo_b[:, c], hi_b[:, c]) - IA.point(x0[:, c]))
    lbl = mv.lo
    lbl = np.where(okp, lbl, -np.inf)
    t0zero = np.ones(len(idx), bool)
    for c in region.tdims:
        t0zero &= (lo_b[:, c] <= 0) & (hi_b[:, c] >= 0)
    lbp = ctx.lb_prof.get((region.D, region.U), -np.inf)
    lbl = np.where(t0zero, np.maximum(lbl, lbp), lbl)
    # ---------------- t-coordinates: penalty + linear term, minimized at endpoints / 0
    tot = IA.point(lbl) + lam * float(nL)
    for c in region.tdims:
        a, b = lo_b[:, c], hi_b[:, c]
        cands = [a, b]
        best = None
        for tc in cands + [np.zeros_like(a)]:
            inside = (tc >= a) & (tc <= b)
            dt = IA.point(tc) - IA.point(x0[:, c])
            lin = (G[:, c] * dt).lo
            pen = (lam * IA.point(qlb(np.abs(tc), ctx.delta))).lo
            val = dn(pen + lin)
            val = np.where(inside, val, np.inf)
            best = val if best is None else np.minimum(best, val)
        tot = tot + IA(best, np.maximum(best, best))
    lb1 = tot.lo
    # ---------------- second-order bound: gradient on the slice t = t0, curvature C on the t-block
    C = t_hessian_negcurv(region, W, Rb, SAb, ctx.S)
    slo, shi = lo_b.copy(), hi_b.copy()
    for c in region.tdims:
        slo[:, c] = x0[:, c]; shi[:, c] = x0[:, c]
    sBlo, sBhi, sOlo, sOhi = build_BO(region, slo, shi)
    ok0, W0 = neumann_inverse(sOlo, sOhi)
    A0 = IA(np.broadcast_to(np.eye(3), (len(idx), 3, 3)).copy() - sBhi, np.broadcast_to(np.eye(3), (len(idx), 3, 3)).copy() - sBlo)
    SA0 = imatmul(IA.point(np.broadcast_to(ctx.S, (len(idx), 3, 3)).copy()), A0)
    R0 = imatmul(itranspose(A0), SA0)
    R0 = intersect(R0, itranspose(R0))
    GB0 = -imatmul(SA0, W0)
    Mg0 = W0 - imatmul(imatmul(W0, R0), W0)
    Mg0 = intersect(Mg0, itranspose(Mg0))
    G0lo = np.zeros((len(idx), 9)); G0hi = np.zeros((len(idx), 9))
    for c, (i, j) in enumerate(region.fwd):
        G0lo[:, c] = GB0.lo[:, i, j]; G0hi[:, c] = GB0.hi[:, i, j]
    for c, (i, j) in enumerate(PAIRS):
        G0lo[:, 3 + c] = Mg0.lo[:, i, j]; G0hi[:, 3 + c] = Mg0.hi[:, i, j]
    for v in range(3):
        g = Mg0[:, v, v] * 0.5
        G0lo[:, 6 + v] = g.lo; G0hi[:, 6 + v] = g.hi
    G0 = IA(G0lo, G0hi)
    # mean value on the slice for u, with corner choice by the slice gradient
    x1 = x0.copy()
    for c in region.udims:
        x1[:, c] = np.where(G0lo[:, c] >= 0, lo_b[:, c], np.where(G0hi[:, c] <= 0, hi_b[:, c], 0.5 * (lo_b[:, c] + hi_b[:, c])))
    l1, okp1 = loss_points(ctx, region, x1)
    mv1 = IA.point(l1)
    for c in region.udims:
        mv1 = mv1 + G0[:, c] * (IA(lo_b[:, c], hi_b[:, c]) - IA.point(x1[:, c]))
    lbl2 = np.where(okp1, mv1.lo, -np.inf)
    lbl2 = np.where(t0zero, np.maximum(lbl2, lbp), lbl2)
    tot2 = IA.point(lbl2) + lam * float(nL)
    Ci = IA.point(C)
    for c in region.tdims:
        a, b = lo_b[:, c], hi_b[:, c]
        best = None
        for tc in [a, b, np.zeros_like(a)]:
            inside = (tc >= a) & (tc <= b)
            dt = IA.point(tc) - IA.point(x0[:, c])
            lin = (G0[:, c] * dt).lo
            quad = (Ci * dt.sqr() * 0.5).hi
            pen = (lam * IA.point(qlb(np.abs(tc), ctx.delta))).lo
            val = dn(dn(pen + lin) - quad)
            val = np.where(inside, val, np.inf)
            best = val if best is None else np.minimum(best, val)
        tot2 = tot2 + IA(best, best)
    lb2 = np.where(ok0, tot2.lo, -np.inf)
    lb[idx] = np.maximum(lb1, lb2)
    return lb, status, gmag


def t_hessian_negcurv(region, W, Rb, SAb, Sm):
    """Certified C >= max(0, -lambda_min(H_tt)) over boxes, H_tt the Hessian of l_S in the
    t-coordinates (B entries and Omega pair entries), via interval entries and Gershgorin.
    W: enclosure of Omega^{-1}; Rb: R; SAb: S A; Sm: S (float)."""
    td = region.tdims
    m = len(td)
    K = W.shape[0]
    if m == 0:
        return np.zeros(K)
    WRW = imatmul(imatmul(W, Rb), W)
    WRW = intersect(WRW, itranspose(WRW))

    def WEW(p, q):
        # (W E W)_{ab} = W_ap W_qb + W_aq W_pb, E = e_p e_q^T + e_q e_p^T
        lo = np.zeros((K, 3, 3)); hi = np.zeros((K, 3, 3))
        M = IA(lo, hi)
        cols = []
        for a in range(3):
            row = []
            for b in range(3):
                row.append(W[:, a, p] * W[:, q, b] + W[:, a, q] * W[:, p, b])
            cols.append(row)
        return cols  # nested list of IA (K,)
    cache = {}

    def wew(c):
        if c not in cache:
            p, q = PAIRS[c - 3]
            cache[c] = WEW(p, q)
        return cache[c]
    H = [[None] * m for _ in range(m)]
    for a_, ca in enumerate(td):
        for b_, cb in enumerate(td):
            if b_ < a_:
                H[a_][b_] = H[b_][a_]
                continue
            if ca < 3 and cb < 3:
                (i, j), (k, l) = region.fwd[ca], region.fwd[cb]
                h = W[:, l, j] * float(Sm[i, k])
            elif ca < 3 or cb < 3:
                cB, cO = (ca, cb) if ca < 3 else (cb, ca)
                i, j = region.fwd[cB]
                X = wew(cO)
                h = None
                for b in range(3):
                    t = SAb[:, i, b] * X[b][j]
                    h = t if h is None else h + t
            else:
                X = wew(ca)
                r, s_ = PAIRS[cb - 3]
                t1 = (X[r][s_] + X[s_][r]) * (-0.5)
                Y = wew(cb)
                acc = None
                for a in range(3):
                    u1 = X[a][r] * WRW[:, s_, a] + X[a][s_] * WRW[:, r, a]
                    p, q = PAIRS[ca - 3]
                    u2 = Y[a][p] * WRW[:, q, a] + Y[a][q] * WRW[:, p, a]
                    t_ = u1 + u2
                    acc = t_ if acc is None else acc + t_
                h = t1 + acc * 0.5
            H[a_][b_] = h
    lmin = None
    for a_ in range(m):
        off = np.zeros(K)
        for b_ in range(m):
            if b_ != a_:
                off = up(off + H[a_][b_].mag())
        g = dn(H[a_][a_].lo - off)
        lmin = g if lmin is None else np.minimum(lmin, g)
    return np.maximum(0.0, -lmin)


def certify_region(ctx, region, max_boxes=3_000_000, max_iter=400, batch=200_000, debug=None):
    t0 = time.monotonic()
    nL = len(region.L)
    tau = ctx.s_star - float((ctx.lam * float(nL)).lo) - float(ctx.fsat.lo) - 1e-12
    mu = ctx.mu_bounds(max(tau, 0.0))
    lo0, hi0 = region.initial_box()
    lo = lo0[None, :].copy(); hi = hi0[None, :].copy()
    total = 0
    min_lb = math.inf
    counts = dict(bounded=0, infeasible=0, kl=0, noinv=0)
    it = 0
    stuck = False
    while len(lo):
        it += 1
        if it > max_iter or total > max_boxes:
            stuck = True
            break
        # process in batches to bound memory
        new_lo, new_hi = [], []
        for s0 in range(0, len(lo), batch):
            l_, h_ = lo[s0:s0 + batch], hi[s0:s0 + batch]
            total += len(l_)
            lb, st, gm = bound_boxes(ctx, region, l_, h_, mu)
            counts["infeasible"] += int(np.sum(st == 1))
            counts["kl"] += int(np.sum(st == 2))
            counts["noinv"] += int(np.sum(st == 3))
            done = (st == 1) | (st == 2) | ((st == 0) & (lb >= ctx.target))
            if np.any((st == 0) & done):
                min_lb = min(min_lb, float(np.min(lb[(st == 0) & done])))
            counts["bounded"] += int(np.sum((st == 0) & done))
            keep = ~done
            if not np.any(keep):
                continue
            l_, h_, gm, st_k = l_[keep], h_[keep], gm[keep], st[keep]
            w = h_ - l_
            # choose split dimension
            sens = w * gm
            # Omega scale for no-inverse boxes: relative widths of Omega entries
            dm = np.sqrt(np.maximum(l_[:, 6:9], OLO) * np.maximum(h_[:, 6:9], OLO))
            rel = np.zeros_like(w)
            for c, (i, j) in enumerate(PAIRS):
                rel[:, 3 + c] = w[:, 3 + c] / np.sqrt(dm[:, i] * dm[:, j])
            for v in range(3):
                rel[:, 6 + v] = w[:, 6 + v] / dm[:, v]
            for c in range(3):
                rel[:, c] = 0.25 * w[:, c]
            use_rel = (st_k == 3) | (np.max(sens, axis=1) <= 0) | ~np.all(np.isfinite(sens), axis=1)
            k = np.where(use_rel, np.argmax(rel, axis=1), np.argmax(np.where(np.isfinite(sens), sens, -1), axis=1))
            r = np.arange(len(l_))
            a, b = l_[r, k], h_[r, k]
            diag = k >= 6
            m = np.where(diag, np.sqrt(np.maximum(a, OLO) * b), 0.5 * (a + b))
            # split off-diagonal / coefficient ranges containing 0 at 0 when wide
            m = np.where(~diag & (a < 0) & (b > 0) & ((b - a) > 1e-3), np.where(np.abs(a + b) < 0.5 * (b - a), 0.0, m), m)
            m = np.where((m <= a) | (m >= b), 0.5 * (a + b), m)
            l2, h2 = l_.copy(), h_.copy()
            h_[r, k] = m
            l2[r, k] = m
            new_lo += [l_, l2]
            new_hi += [h_, h2]
        if new_lo:
            lo = np.concatenate(new_lo); hi = np.concatenate(new_hi)
        else:
            lo = np.zeros((0, 9)); hi = np.zeros((0, 9))
        # guard: tiny boxes that still fail
        if len(lo) and np.min(np.max(hi - lo, axis=1)) < 1e-13:
            stuck = True
            break
    if debug is not None:
        debug["lo"], debug["hi"], debug["mu"] = lo, hi, mu
    return dict(order="".join(str(v + 1) for v in region.order), L=",".join(map(str, region.L)),
                H_D=str(region.D), H_U=str(region.U), certified=not stuck, boxes=total, iterations=it,
                min_lb=min_lb, remaining=int(len(lo)), runtime_s=time.monotonic() - t0, **counts)


def regions_for(ctx):
    """All (order, L); returns (screened list, to-solve list)."""
    screened, pending = [], []
    for order in itertools.permutations(range(3)):
        for r in range(7):
            for L in itertools.combinations(range(6), r):
                lbs = float((ctx.fsat + ctx.lam * float(len(L))).lo)
                if lbs >= ctx.target:
                    screened.append((order, L, lbs))
                else:
                    pending.append(Region(ctx, order, L))
    return screened, pending

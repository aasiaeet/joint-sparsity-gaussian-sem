"""Certified bounds on the constrained support fits l_min(H; S) on three vertices (Section S18).

Notation as in ../global_selection/global_selection.py: B[i, j] != 0 is i -> j; a_v = e_v - sum_p B[p, v] e_p is column v of
A = I - B; R = A^T S A; Omega has zero pattern outside the bidirected set U.

Lower bound (relaxation of the spectral constraint).  For any (B, Omega) with support in H
and Omega positive definite,
    l_S(B, Omega) >= min_{Omega' > 0, pattern U} l_S(B, Omega') = 1/2 log h_H(B) + 3/2,
with the closed forms (det(I - B) = 1, so det R = det S):
  |U| = 0:  h = R_11 R_22 R_33                       (product of convex quadratics a_v^T S a_v)
  |U| = 1, pair {i,j}, third k:  h = Gram(a_i, a_j) R_kk = (n^T adj(S) n) R_kk,
            n = a_i x a_j = (x, y, 1) in coordinates (i, j, k),
            x = B_ki + B_ji B_kj,  y = B_kj + B_ij B_ki            (Cauchy-Binet / Jacobi)
  |U| = 2, hub j, missing pair {i,k}:  h = det S / (1 - rho^2),
            rho^2 = R_ik^2 / (R_ii R_kk)                 (covariance graph i<->j<->k = DAG i->j<-k)
  |U| = 3:  h = det S.
Each factor depends on its own disjoint set of B entries, so min h is the product of the
factor minima over the box |B| <= 2.  Factor minima are bounded below rigorously:
  * convex quadratics (a_v^T S a_v; n^T adj(S) n as a function of (x, y)) by linearization at
    the numerical box-constrained minimizer (valid by convexity), evaluated in interval
    arithmetic;
  * the Gram factor with a product term (x or y bilinear in B) by branch-and-bound over B
    with the convex quadratic minimized over the interval hull of (x, y) on each box;
  * rho^2 by branch-and-bound with the monotonicity-aware mean-value form.
The saturated value 1/2 log det S + 3/2 is a lower bound for every support (Fischer).

Upper bound.  The stored fit point (B, Omega) of each support, certified feasible in K
(|B| <= 2 exactly; 0.05 I <= Omega <= 10 I by interval Cholesky of Omega - 0.05 I and
10 I - Omega, after an inward shrink Omega <- (1-s) Omega + s I with s <= 1e-6 if needed),
evaluated in interval arithmetic.

Certified when  LB >= found - 1e-8  and  UB <= found + 1e-8.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ia import IA, adj3, cholesky_pd, det3, dn, matmul, transpose, up  # noqa: E402

P = 3
MB = 2.0
OLO, OHI = 0.05, 10.0
TOL_CERT = 1e-8


# ============================================================== helpers
class SampleIA:
    """Interval data derived from one sample covariance S."""

    def __init__(self, S):
        self.S = np.asarray(S, float)
        self.Si = IA.point(self.S)
        if not bool(cholesky_pd(self.Si)):
            raise RuntimeError("S not certified positive definite")
        self.detS = det3(self.Si)
        self.adjS = adj3(self.Si)
        lg = self.detS.log()
        self.fsat = lg * 0.5 + 1.5          # 1/2 log det S + 3/2
        # certified lower bound on lambda_min(S): S - sigma I certified PD
        sig = 0.9 * float(np.linalg.eigvalsh(self.S)[0])
        while not bool(cholesky_pd(self.Si - IA.point(sig * np.eye(P)))):
            sig *= 0.5
        self.lam_min_lb = sig


def loss_point(Si, B, O):
    """Interval enclosure of l_S(B, Omega) at a point (B, O floats)."""
    A = IA.point(np.eye(P) - B)  # exact: diagonal 1 - 0, off-diagonal 0 - b
    R = matmul(matmul(transpose(A), Si), A)
    Oi = IA.point(O)
    dO = det3(Oi)
    if not dO.lo > 0:
        raise RuntimeError("Omega not certified nonsingular")
    adjO = adj3(Oi)
    tr = None
    for i in range(P):
        for j in range(P):
            t = adjO[i, j] * R[j, i]
            tr = t if tr is None else tr + t
    return dO.log() * 0.5 + (tr / dO) * 0.5


def spectral_certified(O):
    Oi = IA.point(O)
    E = np.eye(P)
    lo_ok = bool(cholesky_pd(Oi - IA.point(OLO * E)))
    hi_ok = bool(cholesky_pd(IA.point(OHI * E) - Oi))
    return lo_ok and hi_ok


def certified_upper(sia, sup, B, O):
    """Rigorous upper bound on l_min(H; S) from a candidate point; returns (ub, s_used)."""
    mb, mo = sup_masks(sup)
    if np.any(B[~mb] != 0) or np.any(np.abs(B) > MB):
        return math.inf, None
    Ooff = O.copy()
    np.fill_diagonal(Ooff, 0)
    if np.any(Ooff[~mo] != 0) or np.any(O != O.T):
        return math.inf, None
    for s in [0.0, 1e-12, 1e-10, 1e-8, 1e-6]:
        Os = O if s == 0 else (1.0 - s) * O + s * np.eye(P)
        if spectral_certified(Os):
            return float(loss_point(sia.Si, B, Os).hi), s
    return math.inf, None


def sup_masks(sup):
    mb = np.zeros((P, P), bool)
    for i, j in sup.D:
        mb[i, j] = True
    mo = np.zeros((P, P), bool)
    for i, j in sup.U:
        mo[i, j] = mo[j, i] = True
    np.fill_diagonal(mo, True)
    return mb, mo


# ============================================================== convex box QP
def _qp_candidates(M, b, lo, hi):
    """Numerical minimizer of z^T M z + 2 b^T z over boxes (vectorized over the leading axis).
    M: (m, m) float or (K, m, m); b: (K, m); lo, hi: (K, m).  m in {1, 2}."""
    K, m = lo.shape
    M = np.broadcast_to(M, (K, m, m))
    best = np.full(K, np.inf)
    bestz = np.zeros((K, m))
    for pattern in itertools.product((0, 1, 2), repeat=m):  # 0 free, 1 at lo, 2 at hi
        z = np.zeros((K, m))
        fixed = [k for k in range(m) if pattern[k] != 0]
        free = [k for k in range(m) if pattern[k] == 0]
        for k in fixed:
            z[:, k] = lo[:, k] if pattern[k] == 1 else hi[:, k]
        if free:
            Mf = M[:, free][:, :, free]
            rhs = -b[:, free] - np.einsum("kij,kj->ki", M[:, free][:, :, fixed], z[:, fixed]) if fixed else -b[:, free]
            zf = np.linalg.solve(Mf, rhs[..., None])[..., 0]
            z[:, free] = zf
        feas = np.all((z >= lo - 1e-15) & (z <= hi + 1e-15), axis=1)
        z = np.clip(z, lo, hi)
        val = np.einsum("ki,kij,kj->k", z, M, z) + 2 * np.einsum("ki,ki->k", b, z)
        upd = feas & (val < best)
        best[upd] = val[upd]
        bestz[upd] = z[upd]
    return bestz


def qp_lower(Mi, bi, ci, lo, hi):
    """Rigorous lower bound of q(z) = z^T M z + 2 b^T z + c over boxes [lo, hi] (K, m), for
    the TRUE convex q whose coefficients are enclosed by the IA objects Mi (m, m),
    bi (m,) or (K, m), ci (scalar or (K,)).  Uses q(z) >= q(zt) + grad q(zt).(z - zt)."""
    K, m = lo.shape
    Mm = Mi.mid()
    bm = np.broadcast_to(bi.mid(), (K, m))
    zt = _qp_candidates(Mm, bm, lo, hi)
    Z = IA.point(zt)
    Mb = IA(np.broadcast_to(Mi.lo, (K, m, m)), np.broadcast_to(Mi.hi, (K, m, m)))
    bb = IA(np.broadcast_to(bi.lo, (K, m)).copy(), np.broadcast_to(bi.hi, (K, m)).copy())
    MZ = None
    for j in range(m):
        t = Mb[:, :, j] * Z[:, j:j + 1]
        MZ = t if MZ is None else MZ + t
    g = (MZ + bb) * 2.0                                 # gradient at zt, (K, m)
    q = None
    for i in range(m):
        t = Z[:, i] * (MZ[:, i] + bb[:, i] * 2.0)       # z^T M z + 2 b^T z
        q = t if q is None else q + t
    q = q + ci
    lin = None
    for i in range(m):
        t = g[:, i] * (IA(lo[:, i], hi[:, i]) - Z[:, i])
        lin = t if lin is None else lin + t
    return (q + lin).lo, zt


def qp_value(Mi, bi, ci, z):
    """Interval value of q at points z (K, m)."""
    K, m = z.shape
    Z = IA.point(z)
    Mb = IA(np.broadcast_to(Mi.lo, (K, m, m)), np.broadcast_to(Mi.hi, (K, m, m)))
    bb = IA(np.broadcast_to(bi.lo, (K, m)).copy(), np.broadcast_to(bi.hi, (K, m)).copy())
    q = None
    for i in range(m):
        acc = bb[:, i] * 2.0
        for j in range(m):
            acc = acc + Mb[:, i, j] * Z[:, j]
        t = Z[:, i] * acc
        q = t if q is None else q + t
    return q + ci


# ============================================================== generic vectorized B&B
def bnb(lb_fn, ub_fn, lo0, hi0, rtol=1e-12, atol=1e-14, max_boxes=400000, max_iter=200,
        ub0=math.inf, split_fn=None):
    """Minimize over the box [lo0, hi0].  lb_fn(lo, hi) -> rigorous lower bounds (K,);
    ub_fn(lo, hi) -> rigorous upper bounds of the function at feasible points (K,);
    ub0: a rigorous upper bound from a feasible point; split_fn(lo, hi) -> dimension to
    bisect per box (default: widest).  Returns dict(lb, ub, boxes, converged)."""
    lo = np.atleast_2d(np.asarray(lo0, float))
    hi = np.atleast_2d(np.asarray(hi0, float))
    ub = ub0
    lb_fathomed = math.inf
    total = 0
    it = 0
    while len(lo) and it < max_iter and total < max_boxes:
        it += 1
        total += len(lo)
        lbs = lb_fn(lo, hi)
        ubs = ub_fn(lo, hi)
        ub = min(ub, float(np.min(ubs)))
        tol = max(atol, rtol * abs(ub))
        keep = lbs < ub - tol
        if np.any(~keep):
            lb_fathomed = min(lb_fathomed, float(np.min(lbs[~keep])))
        lo, hi, lbs = lo[keep], hi[keep], lbs[keep]
        if not len(lo):
            break
        w = hi - lo
        k = np.argmax(w, axis=1) if split_fn is None else split_fn(lo, hi)
        r = np.arange(len(lo))
        midv = 0.5 * (lo[r, k] + hi[r, k])
        lo2, hi2 = lo.copy(), hi.copy()
        hi[r, k] = midv
        lo2[r, k] = midv
        lo = np.concatenate([lo, lo2])
        hi = np.concatenate([hi, hi2])
    if len(lo):
        lbs = lb_fn(lo, hi)
        lb = min(lb_fathomed, float(np.min(lbs)))
        return dict(lb=lb, ub=ub, boxes=total, converged=False)
    return dict(lb=lb_fathomed, ub=ub, boxes=total, converged=True)


# ============================================================== factors
def factor_quadratic(sia, v, parents):
    """min over |B_pv| <= 2 of a_v^T S a_v (convex).  Returns (lb, ub, zt)."""
    S = sia.S
    if not parents:
        return float(S[v, v]), float(S[v, v]), None
    pa = list(parents)
    m = len(pa)
    Mi = IA.point(S[np.ix_(pa, pa)])
    bi = IA.point(-S[pa, v])
    ci = IA.point(np.array(S[v, v]))
    lo = np.full((1, m), -MB)
    hi = np.full((1, m), MB)
    lb, zt = qp_lower(Mi, bi, ci, lo, hi)
    ubv = qp_value(Mi, bi, ci, zt).hi
    return float(lb[0]), float(ubv[0]), zt[0]


def _gram_setup(sia, i, j):
    k = 3 - i - j
    perm = [i, j, k]
    Ap = IA(sia.adjS.lo[np.ix_(perm, perm)], sia.adjS.hi[np.ix_(perm, perm)])
    Mi = Ap[:2, :2]
    bi = Ap[:2, 2]
    ci = Ap[2, 2]
    return k, Mi, bi, ci


def factor_gram(sia, i, j, D):
    """min over the B entries with head in {i, j} of Gram(a_i, a_j) = n^T adj(S) n."""
    k, Mi, bi, ci = _gram_setup(sia, i, j)
    names = [e for e in [(k, i), (k, j), (j, i), (i, j)] if e in D]
    has = {e: (e in D) for e in [(k, i), (k, j), (j, i), (i, j)]}
    pos = {e: n for n, e in enumerate(names)}

    def xy(lo, hi):
        # interval hull of (x, y) over B boxes
        K = lo.shape[0]
        zero = IA.point(np.zeros(K))

        def var(e):
            return IA(lo[:, pos[e]], hi[:, pos[e]]) if has[e] else zero
        x = var((k, i)) + var((j, i)) * var((k, j))
        y = var((k, j)) + var((i, j)) * var((k, i))
        return x, y

    def xy_point(z):
        K = z.shape[0]

        def var(e):
            return z[:, pos[e]] if has[e] else np.zeros(K)
        x = var((k, i)) + var((j, i)) * var((k, j))   # float, used only as a point in (x, y)
        y = var((k, j)) + var((i, j)) * var((k, i))
        return np.stack([x, y], axis=1)

    if not names:
        q = ci  # n = (0, 0, 1)
        return float(q.lo), float(q.hi), dict(boxes=0, converged=True)

    def lb_fn(lo, hi):
        x, y = xy(lo, hi)
        lb, _ = qp_lower(Mi, bi, ci, np.stack([x.lo, y.lo], 1), np.stack([x.hi, y.hi], 1))
        return lb

    def ub_fn(lo, hi):
        c = 0.5 * (lo + hi)
        # the point (x, y) is computed in floats from the B point c; to stay rigorous we
        # bound q over the tiny interval hull of (x, y) at the point c
        x, y = xy(c, c)
        K = len(c)
        Z = IA(np.stack([x.lo, y.lo], 1), np.stack([x.hi, y.hi], 1))
        q = None
        for a in range(2):
            acc = bi[a] * 2.0
            for b_ in range(2):
                acc = acc + Mi[a, b_] * Z[:, b_]
            t = Z[:, a] * acc
            q = t if q is None else q + t
        return (q + ci).hi

    def split_fn(lo, hi):
        K = lo.shape[0]
        zero = IA.point(np.zeros(K))
        one = IA.point(np.ones(K))

        def var(e):
            return IA(lo[:, pos[e]], hi[:, pos[e]]) if has[e] else zero
        sens = np.zeros((K, d))
        for e in names:
            # |dx/dz| + |dy/dz| over the box
            if e == (k, i):
                dx, dy = one, var((i, j))
            elif e == (k, j):
                dx, dy = var((j, i)), one
            elif e == (j, i):
                dx, dy = var((k, j)), zero
            else:  # (i, j)
                dx, dy = zero, var((k, i))
            sens[:, pos[e]] = (hi[:, pos[e]] - lo[:, pos[e]]) * (dx.mag() + dy.mag())
        return np.argmax(sens, axis=1)

    def fval(zz):
        c = np.atleast_2d(zz)
        return float(ub_fn(c, c)[0])

    d = len(names)
    ub0 = _local_ub(fval, d)
    res = bnb(lb_fn, ub_fn, np.full(d, -MB), np.full(d, MB), ub0=ub0, split_fn=split_fn)
    return res["lb"], res["ub"], res


def _local_ub(fval, d, seed=12345):
    """Best rigorous factor value over a few local minimizations (feasible points only)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    starts = [np.zeros(d)] + [rng.uniform(-MB, MB, d) for _ in range(6)]
    best = math.inf
    for x0 in starts:
        r = minimize(lambda z: fval(np.clip(z, -MB, MB)), x0, method="L-BFGS-B",
                     bounds=[(-MB, MB)] * d, options=dict(ftol=1e-15, gtol=1e-12, maxiter=500))
        best = min(best, fval(np.clip(r.x, -MB, MB)))
    return best


def factor_rho2(sia, i, k, D):
    """min over the B entries with head in {i, k} of rho^2 = R_ik^2 / (R_ii R_kk)."""
    S = sia.S
    Si = sia.Si
    names = sorted(e for e in D if e[1] in (i, k))
    sig = sia.lam_min_lb  # R_vv = a_v^T S a_v >= lambda_min(S) |a_v|^2 >= lambda_min(S)
    if not names:
        R = S
        r = IA.point(np.array(R[i, k])).sqr() / (IA.point(np.array(R[i, i])) * IA.point(np.array(R[k, k])))
        return float(r.lo), float(r.hi), dict(boxes=0, converged=True)
    d = len(names)

    def build(lo, hi):
        K = lo.shape[0]
        ai_lo = np.zeros((K, 3)); ai_hi = np.zeros((K, 3))
        ak_lo = np.zeros((K, 3)); ak_hi = np.zeros((K, 3))
        ai_lo[:, i] = ai_hi[:, i] = 1.0
        ak_lo[:, k] = ak_hi[:, k] = 1.0
        for n_, (p_, h_) in enumerate(names):
            if h_ == i:
                ai_lo[:, p_] = -hi[:, n_]; ai_hi[:, p_] = -lo[:, n_]
            else:
                ak_lo[:, p_] = -hi[:, n_]; ak_hi[:, p_] = -lo[:, n_]
        ai = IA(ai_lo, ai_hi); ak = IA(ak_lo, ak_hi)
        Sai = None; Sak = None
        for c in range(3):
            t1 = IA.point(np.broadcast_to(S[:, c], (K, 3))) * ai[:, c:c + 1]
            t2 = IA.point(np.broadcast_to(S[:, c], (K, 3))) * ak[:, c:c + 1]
            Sai = t1 if Sai is None else Sai + t1
            Sak = t2 if Sak is None else Sak + t2
        Rik = None; Rii = None; Rkk = None
        for c in range(3):
            a = ai[:, c] * Sak[:, c]; b = ai[:, c] * Sai[:, c]; cc = ak[:, c] * Sak[:, c]
            Rik = a if Rik is None else Rik + a
            Rii = b if Rii is None else Rii + b
            Rkk = cc if Rkk is None else Rkk + cc
        Rii = IA(np.maximum(Rii.lo, sig), np.maximum(Rii.hi, sig))
        Rkk = IA(np.maximum(Rkk.lo, sig), np.maximum(Rkk.hi, sig))
        return Sai, Sak, Rik, Rii, Rkk

    def value(lo, hi):
        _, _, Rik, Rii, Rkk = build(lo, hi)
        return Rik.sqr() / (Rii * Rkk)

    def grad(lo, hi):
        Sai, Sak, Rik, Rii, Rkk = build(lo, hi)
        p = Rik.sqr() / (Rii * Rkk)
        gs = []
        for n_, (p_, h_) in enumerate(names):
            if h_ == i:
                dRik = -Sak[:, p_]; dRii = Sai[:, p_] * (-2.0)
                g = (Rik * dRik * 2.0) / (Rii * Rkk) - p * (dRii / Rii)
            else:
                dRik = -Sai[:, p_]; dRkk = Sak[:, p_] * (-2.0)
                g = (Rik * dRik * 2.0) / (Rii * Rkk) - p * (dRkk / Rkk)
            gs.append(g)
        return gs

    def lb_fn(lo, hi):
        gs = grad(lo, hi)
        c = 0.5 * (lo + hi)
        for n_, g in enumerate(gs):
            c[:, n_] = np.where(g.lo >= 0, lo[:, n_], np.where(g.hi <= 0, hi[:, n_], c[:, n_]))
        v = value(c, c)
        lin = None
        for n_, g in enumerate(gs):
            t = g * (IA(lo[:, n_], hi[:, n_]) - IA.point(c[:, n_]))
            lin = t if lin is None else lin + t
        nat = value(lo, hi).lo
        return np.maximum((v + lin).lo, nat)

    def ub_fn(lo, hi):
        c = 0.5 * (lo + hi)
        return value(c, c).hi

    def split_fn(lo, hi):
        gs = grad(lo, hi)
        sens = np.stack([(hi[:, n_] - lo[:, n_]) * g.mag() for n_, g in enumerate(gs)], axis=1)
        return np.argmax(sens, axis=1)

    def fval(zz):
        c = np.atleast_2d(zz)
        return float(ub_fn(c, c)[0])

    ub0 = _local_ub(fval, d)
    res = bnb(lb_fn, ub_fn, np.full(d, -MB), np.full(d, MB), ub0=ub0, split_fn=split_fn)
    return res["lb"], res["ub"], res


# ============================================================== support lower bound
def support_lower(sia, sup, skip_rho2=False):
    """Rigorous lower bound on the spectral-unconstrained profile minimum of support H."""
    D = set(sup.D)
    U = list(sup.U)
    info = {}
    if len(U) == 3:
        h_lo = sia.detS.lo
        info["form"] = "U3"
    elif len(U) == 0:
        h = IA.point(1.0)
        for v in range(P):
            pa = [p for p in range(P) if (p, v) in D]
            lb, ub, _ = factor_quadratic(sia, v, pa)
            h = h * IA(lb, lb)
        h_lo = h.lo
        info["form"] = "U0"
    elif len(U) == 1:
        i, j = U[0]
        k = 3 - i - j
        lbg, ubg, rg = factor_gram(sia, i, j, D)
        pa = [p for p in range(P) if (p, k) in D]
        lbq, ubq, _ = factor_quadratic(sia, k, pa)
        h_lo = (IA(lbg, lbg) * IA(lbq, lbq)).lo
        info.update(form="U1", gram_boxes=rg["boxes"], gram_converged=rg["converged"])
    else:
        (i, k), = [pr for pr in [(0, 1), (0, 2), (1, 2)] if pr not in U]
        if skip_rho2:
            return -math.inf, dict(form="U2_skipped")
        lbr, ubr, rr = factor_rho2(sia, i, k, D)
        lbr = max(lbr, 0.0)
        if lbr >= 1.0:
            raise RuntimeError("rho^2 lower bound >= 1")
        h_lo = (sia.detS / (IA.point(1.0) - IA(lbr, lbr))).lo
        info.update(form="U2", rho2_lb=lbr, rho2_ub=ubr, rho2_boxes=rr["boxes"], rho2_converged=rr["converged"])
    if not h_lo > 0:
        return -math.inf, info
    f = IA(h_lo, h_lo).log() * 0.5 + 1.5
    return float(f.lo), info


def profile_value(S, sup, B):
    """Float profile value at B (for diagnostics only)."""
    A = np.eye(P) - B
    R = A.T @ S @ A
    U = list(sup.U)
    if len(U) == 3:
        return 0.5 * np.linalg.slogdet(S)[1] + 1.5
    if len(U) == 0:
        return 0.5 * np.sum(np.log(np.diag(R))) + 1.5
    if len(U) == 1:
        (i, j), = U
        k = 3 - i - j
        return 0.5 * (np.log(R[i, i] * R[j, j] - R[i, j] ** 2) + np.log(R[k, k])) + 1.5
    (i, k), = [pr for pr in [(0, 1), (0, 2), (1, 2)] if pr not in U]
    rho2 = R[i, k] ** 2 / (R[i, i] * R[k, k])
    return 0.5 * (np.linalg.slogdet(S)[1] - np.log1p(-rho2)) + 1.5

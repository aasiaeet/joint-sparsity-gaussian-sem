"""B-space branch and bound for the criterion bound, with an analytic inner bound over Omega.

Region R(pi, L) as in criterion_bounds.py.  Write T_B, T_Omega for the coordinates not in L, and
tau = s* - lambda|L| - fsat (fsat = 1/2 log det S + 3/2).  For fixed B put R = A^T S A,
M = R^{-1/2} Omega R^{-1/2} with eigenvalues mu_i, so that
    l_S(B, Omega) - fsat = 1/2 sum_i h(mu_i),   h(mu) = log mu + 1/mu - 1 >= 0.
Choose mu_hi in (1, 2) with h(mu_hi) >= 2 tau (certified).  Then:
  (1) if Omega is not <= mu_hi R, some mu_i > mu_hi, so l_S >= fsat + tau and
      F >= l_S + lambda|L| >= s* (up to 1e-12);
  (2) on the convex set {0 < Omega <= mu_hi R}, Omega -> l_S(B, Omega) is strongly convex:
      d^2 l[H, H] = tr(X^2 (M^{-1} - I/2)) >= m ||H||_R^2,  m = (1/mu_hi - 1/2) / mu_hi^2,
      ||H||_R^2 = tr(R^{-1} H R^{-1} H)  (X = M^{-1/2} R^{-1/2} H R^{-1/2} M^{-1/2}).
For a reference Omega~ (profile of support H(pi, L) at the box center, so that its
T_Omega entries are 0) with Omega~ <= mu_hi R(B) certified for all B in the box, and the
penalty bound lambda q(t) >= c |t| on |t| <= delta' (c = lambda / delta'), every region point
with Omega <= mu_hi R satisfies
    l_S + sum_{T_Omega} lambda q >= phi(B) - tr(G' R G' R) / (2 m),
where phi(B) = l_S(B, Omega~) = 1/2 log det Omega~ + 1/2 tr(Omega~^{-1} R(B)) (convex
quadratic in B), G = 1/2 (W - W R W) (W = Omega~^{-1}) is the matrix gradient at Omega~ and
G' equals G with its T_Omega entries soft-thresholded at c/2 (weak duality for the l1 term;
the minimum of the quadratic model over all symmetric Delta is -tr(G'RG'R)/(2m)).
Then over a box of B, with the T_B penalties:
    F >= lambda|L| + min_B [phi(B) + sum_{T_B} lambda q(B_k)] - max_B tr(G'RG'R)/(2m),
the first minimum bounded by the monotonicity-aware mean-value form in B (per T_B
coordinate jointly with the penalty, minimized at an endpoint or 0), the second maximum
by interval arithmetic.  Only the 3 forward B coordinates are branched.
Usable when tau < h(2)/2 = 0.0966 (mu_hi < 2); other regions use criterion_bounds.certify_region.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ia import IA, adj3, cholesky_pd, det3, dn, up  # noqa: E402
from criterion_bounds import PAIRS, MB, TOL, imatmul, intersect, itranspose, neumann_inverse, pair_prune, qlb  # noqa: E402


def profile_omega(R, U):
    """Closed-form profile Omega for pattern U (vectorized over (K,3,3) float R)."""
    K = R.shape[0]
    O = np.zeros_like(R)
    for v in range(3):
        O[:, v, v] = R[:, v, v]
    if len(U) == 3:
        return 0.5 * (R + np.swapaxes(R, 1, 2))
    if len(U) == 1:
        (i, j), = U
        O[:, i, j] = O[:, j, i] = R[:, i, j]
        return O
    if len(U) == 2:
        (i, k), = [pr for pr in PAIRS if pr not in U]
        j = 3 - i - k
        Rii, Rkk, Rik = R[:, i, i], R[:, k, k], R[:, i, k]
        det = Rii * Rkk - Rik ** 2
        b_i = (Rkk * R[:, i, j] - Rik * R[:, k, j]) / det
        b_k = (Rii * R[:, k, j] - Rik * R[:, i, j]) / det
        rss = R[:, j, j] - b_i * R[:, i, j] - b_k * R[:, k, j]
        O[:, i, j] = O[:, j, i] = b_i * Rii
        O[:, k, j] = O[:, j, k] = b_k * Rkk
        O[:, j, j] = b_i ** 2 * Rii + b_k ** 2 * Rkk + rss
        return O
    return O


def soft(lo, hi, c):
    """Interval soft-threshold at c >= 0 (monotone, so endpoints map to endpoints)."""
    f = lambda x: np.sign(x) * np.maximum(np.abs(x) - c, 0.0)
    return f(lo), f(hi)


def _half_h_tangent_min(x0, coef, sign, xmax):
    """Rigorous lower bound of min_{0 < x <= xmax} phi(x), phi(x) = 1/2 (log x + 1/x - 1) +
    sign * coef * x, for coef >= 0 (IA), xmax < 2 (phi convex there), via the tangent at the
    float point x0 in (0, xmax]: phi(x) >= phi(x0) + phi'(x0)(x - x0)."""
    X = IA.point(x0)
    one = IA.point(np.ones_like(x0))
    c = coef if sign > 0 else -coef
    phi = (X.log() + one / X - 1.0) * 0.5 + c * X
    dphi = (one / X - one / X.sqr()) * 0.5 + c
    lin_lo = dphi * (IA.point(np.zeros_like(x0)) - X)       # x -> 0
    lin_hi = dphi * (IA.point(np.full_like(x0, xmax)) - X)  # x -> xmax
    return dn(phi.lo + np.minimum(lin_lo.lo, lin_hi.lo))


def graded_pair_bound(ctx, R, p, q, mu_hi, kappas=np.geomspace(1e-3, 1e3, 41)):
    """Rigorous lower bound on (l_S - fsat) over all Omega with |Omega_pq| <= delta' and
    Omega <= mu_hi R (R: IA (K,3,3) over B-boxes).  With v+- = e_p +- s e_q,
    u+- = v^T R v, a = mu_min, b = mu_max of R^{-1/2} Omega R^{-1/2}:
      a u+ - b u- <= 4 s delta'  and  a u- - b u+ <= 4 s delta'   (since w+ - w- = 4 s Omega_pq),
      l_S - fsat >= 1/2 (h(a) + h(b)),  b <= mu_hi < 2.
    Weak duality with multiplier kappa >= 0 on the binding constraint gives
      >= min_a [h(a)/2 + kappa u_1 a] + min_{b <= mu_hi} [h(b)/2 - kappa u_2 b] - kappa 4 s delta'."""
    K = R.shape[0]
    Rm = 0.5 * (R.lo + R.hi)
    s = np.sqrt(np.maximum(Rm[:, p, p], 1e-12) / np.maximum(Rm[:, q, q], 1e-12))
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
    Si = IA.point(s)
    up_ = R[:, p, p] + Si * R[:, p, q] * 2.0 + Si.sqr() * R[:, q, q]
    um_ = R[:, p, p] - Si * R[:, p, q] * 2.0 + Si.sqr() * R[:, q, q]
    up_ = IA(np.maximum(up_.lo, 0.0), up_.hi)
    um_ = IA(np.maximum(um_.lo, 0.0), um_.hi)
    c0 = (Si * (4.0 * ctx.delta_hi)).hi
    pos = 0.5 * (up_.lo + up_.hi) >= 0.5 * (um_.lo + um_.hi)
    u1 = IA(np.where(pos, up_.lo, um_.lo), np.where(pos, up_.hi, um_.hi))   # coefficient of a
    u2 = IA(np.where(pos, um_.lo, up_.lo), np.where(pos, um_.hi, up_.hi))   # coefficient of b
    best = np.zeros(K)
    um1, um2 = 0.5 * (u1.lo + u1.hi), 0.5 * (u2.lo + u2.hi)

    def h2(x):
        return 0.5 * (np.log(x) + 1.0 / x - 1.0)
    # float dual values on a grid of kappa to pick kappa per box
    vals = []
    xb = np.linspace(1.0, mu_hi, 64)
    for kap in kappas:
        al = kap * um1
        a = 2.0 / (1.0 + np.sqrt(1.0 + 8.0 * al))
        va = h2(a) + al * a
        be = kap * um2
        vb = np.min(h2(xb)[None, :] - be[:, None] * xb[None, :], axis=1)
        vals.append(va + vb - kap * c0)
    vals = np.array(vals)
    kidx = np.argmax(vals, axis=0)
    kap = kappas[kidx]
    al_f = kap * um1
    a0 = 2.0 / (1.0 + np.sqrt(1.0 + 8.0 * al_f))
    be_f = kap * um2
    bvals = h2(xb)[None, :] - be_f[:, None] * xb[None, :]
    b0 = xb[np.argmin(bvals, axis=1)]
    Kap = IA.point(kap)
    # a-term: coefficient kappa u1 >= 0; phi increasing in the coefficient -> use lower end
    alpha = Kap * IA.point(u1.lo)
    ta = _half_h_tangent_min(np.minimum(a0, 1.0), IA(alpha.lo, alpha.lo), +1, 1.999)
    # b-term: -kappa u2 b decreasing in the coefficient -> use upper end
    beta = Kap * IA.point(u2.hi)
    tb = _half_h_tangent_min(np.minimum(b0, mu_hi), IA(beta.hi, beta.hi), -1, mu_hi)
    pen = (Kap * IA.point(c0)).hi
    val = dn(dn(ta + tb) - pen)
    return np.maximum(best, np.where(np.isfinite(val), val, 0.0))


def quad_kl_bound(ctx, region, R, mu_hi):
    """Joint (all small pairs) lower bound on l_S - fsat for Omega <= mu_hi R with
    |Omega_pq| <= delta' on the small pairs T.
    (i)  h(w) >= c (w - 1)^2 on (0, mu_hi], c = h(mu_hi) / (mu_hi - 1)^2 (h(w)/(w-1)^2 is
         nonincreasing on (1, inf) and h(w) >= (w-1)^2/2 on (0, 1]); hence
         l_S - fsat >= (c/2) tr(R^{-1} D R^{-1} D),  D = Omega - R.
    (ii) For any symmetric Y supported on T (Cauchy-Schwarz in the R-inner product):
         tr(R^{-1} D R^{-1} D) >= tr(Y D)^2 / tr(R Y R Y),
         tr(Y D) = 2 sum_T y_pq (Omega_pq - R_pq), minimized over |Omega_pq| <= delta'.
    Y is chosen numerically (float) as R^{-1} D* R^{-1} on T for the R-norm projection D*."""
    K = R.shape[0]
    TO = [c for c in range(3, 6) if c not in region.L]
    if not TO:
        return np.zeros(K)
    dp = ctx.delta_hi
    Rm = 0.5 * (R.lo + R.hi)
    try:
        Ri = np.linalg.inv(Rm)
    except np.linalg.LinAlgError:
        return np.zeros(K)
    basis = []
    for v in range(3):
        E = np.zeros((3, 3)); E[v, v] = 1.0; basis.append(E)
    for (p, q) in PAIRS:
        E = np.zeros((3, 3)); E[p, q] = E[q, p] = 1.0; basis.append(E)
    RiE = [np.einsum("kab,bc->kac", Ri, E) for E in basis]
    Q = np.empty((K, 6, 6))
    for i in range(6):
        for j in range(i, 6):
            Q[:, i, j] = Q[:, j, i] = np.einsum("kab,kba->k", RiE[i], RiE[j])
    Tc = [3 + (c - 3) for c in TO]
    Fc = [i for i in range(6) if i not in Tc]
    QFF = Q[:, Fc][:, :, Fc]; QFT = Q[:, Fc][:, :, Tc]; QTT = Q[:, Tc][:, :, Tc]
    try:
        X = np.linalg.solve(QFF, QFT)
    except np.linalg.LinAlgError:
        return np.zeros(K)
    Pm = QTT - np.einsum("kfi,kfj->kij", QFT, X)
    Pm = 0.5 * (Pm + np.swapaxes(Pm, 1, 2))
    rT = np.stack([Rm[:, PAIRS[c - 3][0], PAIRS[c - 3][1]] for c in TO], axis=1)
    m = len(TO)
    from support_bounds import _qp_candidates
    tlo = np.full((K, m), -dp); thi = np.full((K, m), dp)
    ok = np.all(np.linalg.eigvalsh(Pm) > 0, axis=1)
    Psafe = np.where(ok[:, None, None], Pm, np.eye(m)[None])
    tstar = _qp_candidates(Psafe, -np.einsum("kij,kj->ki", Psafe, rT), tlo, thi)
    dstar = tstar - rT
    dF = -np.einsum("kfj,kj->kf", X, dstar)
    Dm = np.zeros((K, 3, 3))
    for n_, i in enumerate(Fc):
        Dm += dF[:, n_, None, None] * basis[i][None]
    for n_, i in enumerate(Tc):
        Dm += dstar[:, n_, None, None] * basis[i][None]
    Yf = np.einsum("kab,kbc,kcd->kad", Ri, Dm, Ri)
    Y = np.zeros((K, 3, 3))
    for c in TO:
        p, q = PAIRS[c - 3]
        Y[:, p, q] = Y[:, q, p] = Yf[:, p, q]
    Yi = IA.point(Y)
    RY = imatmul(R, Yi)
    tr = None
    for a_ in range(3):
        for b_ in range(3):
            t = RY[:, a_, b_] * RY[:, b_, a_]
            tr = t if tr is None else tr + t
    Lsum = None
    for c in TO:
        p, q = PAIRS[c - 3]
        t = IA.point(Y[:, p, q]) * (IA(np.full(K, -dp), np.full(K, dp)) - R[:, p, q]) * 2.0
        Lsum = t if Lsum is None else Lsum + t
    Dlo = np.where(Lsum.lo > 0, Lsum.lo, np.where(Lsum.hi < 0, -Lsum.hi, 0.0))
    MH = IA.point(np.array(mu_hi))
    hmu = MH.log() + IA.point(np.array(1.0)) / MH - 1.0
    cc = float((hmu / (MH - 1.0).sqr()).lo)
    if not (cc > 0 and mu_hi > 1):
        return np.zeros(K)
    trh = np.where(tr.hi > 0, tr.hi, np.inf)
    val = (IA.point(Dlo).sqr() / IA.point(trh) * (0.5 * cc)).lo
    val = np.where(ok & np.isfinite(val), dn(val), 0.0)
    return np.maximum(val, 0.0)


def ref_solve(R, U, TO, c, dp, sweeps=40):
    """Float, approximate: minimize l_R(Omega) + c sum_T |Omega_pq| s.t. |Omega_pq| <= dp (pairs in
    TO) by exact-curvature coordinate descent, starting from the pattern profile.  Only used
    to choose the reference point; the bound built on it is valid for any reference."""
    K = R.shape[0]
    O = profile_omega(R, U)
    O = 0.5 * (O + np.swapaxes(O, 1, 2))
    Tset = {PAIRS[cc - 3] for cc in TO}
    coords = [("d", v, v) for v in range(3)] + [("p", p, q) for (p, q) in PAIRS]
    for _ in range(sweeps):
        for kind, p, q in coords:
            if kind == "p" and (p, q) not in Tset and (p, q) not in U:
                continue  # entries outside L_Omega and T do not exist in this region (none here)
            E = np.zeros((3, 3))
            if kind == "d":
                E[p, p] = 1.0
            else:
                E[p, q] = E[q, p] = 1.0
            W = np.linalg.inv(O)
            WR = W @ R
            G = 0.5 * (W - WR @ W)
            g = np.einsum("kab,ba->k", G, E)
            WE = W @ E
            WEWE = WE @ WE
            t1 = np.einsum("kaa->k", WEWE)
            t2 = np.einsum("kab,kba->k", WEWE, WR)
            H2 = -0.5 * t1 + t2
            H2 = np.where(H2 > 1e-12, H2, 1e-12 + np.abs(H2))
            x = O[:, p, q]
            u = x - g / H2
            if kind == "p" and (p, q) in Tset:
                thr = c / H2
                u = np.sign(u) * np.maximum(np.abs(u) - thr, 0.0)
                u = np.clip(u, -dp, dp)
            step = u - x
            for _bt in range(30):
                On = O.copy()
                On[:, p, q] = x + step
                On[:, q, p] = x + step
                okpd = np.linalg.eigvalsh(On)[:, 0] > 1e-10
                if np.all(okpd):
                    break
                step = np.where(okpd, step, 0.5 * step)
            O[:, p, q] = x + step
            O[:, q, p] = x + step
    return O


def bound_REF(ctx, region, lo, hi, mus):
    """Reference-solution bound.  Omega~ approximately solves the inner convex problem at the
    box center (small pairs allowed nonzero).  For |x| <= delta' and any z:
    c|x| >= 2 z x - max(0, 2|z| - c) delta'.  With z_pq = -G_pq(Omega~) on the small pairs,
    G' = G + Z, and strong convexity on {Omega <= mu' R}:
      l + sum_T lambda q(Omega_pq) >= l(Omega~) + sum_T [2 z Omega~_pq - max(0, 2|z| - c) delta']
                                      - tr(G' R G' R) / (2 m),
    uniformly over B in the box (G and l(B, Omega~) enclosed in interval arithmetic)."""
    K = lo.shape[0]
    lam = ctx.lam
    nL = len(region.L)
    fwd = region.fwd
    LB_dims = [c for c in range(3) if c in region.L]
    TB_dims = [c for c in range(3) if c not in region.L]
    TO = [c for c in range(3, 6) if c not in region.L]
    U = region.U
    dp = ctx.delta_hi
    Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        Blo[:, i, j] = lo[:, c]; Bhi[:, i, j] = hi[:, c]
    I3 = np.broadcast_to(np.eye(3), (K, 3, 3))
    A = IA(I3 - Bhi, I3 - Blo)
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    SA = imatmul(Sb, A)
    R = imatmul(itranspose(A), SA)
    R = intersect(R, itranspose(R))
    cB = 0.5 * (lo + hi)
    Bc = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        Bc[:, i, j] = cB[:, c]
    Ac = np.eye(3) - Bc
    Rc = np.einsum("kai,ab,kbj->kij", Ac, ctx.S, Ac)
    c_lo = float((lam / IA.point(np.array(dp))).lo)
    Ot = ref_solve(Rc, U, TO, c_lo, dp)
    Ot = 0.5 * (Ot + np.swapaxes(Ot, 1, 2))
    Oti = IA.point(Ot)
    valid = cholesky_pd(Oti)
    best_m = np.full(K, -np.inf)
    for m_R, mu in mus:
        okm = cholesky_pd(R * mu - Oti)
        best_m = np.where(okm & (m_R > best_m), m_R, best_m)
    valid &= np.isfinite(best_m)
    dO = det3(Oti)
    valid &= dO.lo > 0
    dOs = IA(np.where(dO.lo > 0, dO.lo, 1.0), np.where(dO.lo > 0, dO.hi, 1.0))
    W = adj3(Oti) / IA(dOs.lo[:, None, None], dOs.hi[:, None, None])
    W = intersect(W, itranspose(W))
    GB = -imatmul(SA, W)
    G = [GB[:, i, j] for (i, j) in fwd]
    x0 = cB.copy()
    for c in LB_dims:
        x0[:, c] = np.where(G[c].lo >= 0, lo[:, c], np.where(G[c].hi <= 0, hi[:, c], cB[:, c]))
    for c in TB_dims:
        x0[:, c] = np.clip(0.0, lo[:, c], hi[:, c])
    B0 = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        B0[:, i, j] = x0[:, c]
    A0 = IA.point(np.eye(3) - B0)
    R0 = imatmul(itranspose(A0), imatmul(Sb, A0))
    tr = None
    for a in range(3):
        for b in range(3):
            t = W[:, a, b] * R0[:, b, a]
            tr = t if tr is None else tr + t
    tot = dOs.log() * 0.5 + tr * 0.5 + lam * float(nL)
    for c in LB_dims:
        tot = tot + G[c] * (IA(lo[:, c], hi[:, c]) - IA.point(x0[:, c]))
    for c in TB_dims:
        a, b = lo[:, c], hi[:, c]
        best = None
        for tc in [a, b, np.zeros_like(a)]:
            inside = (tc >= a) & (tc <= b)
            dt = IA.point(tc) - IA.point(x0[:, c])
            lin = (G[c] * dt).lo
            pen = (lam * IA.point(qlb(np.abs(tc), ctx.delta))).lo
            val = np.where(inside, dn(pen + lin), np.inf)
            best = val if best is None else np.minimum(best, val)
        tot = tot + IA(best, best)
    WRW = imatmul(imatmul(W, R), W)
    Gm = (W - WRW) * 0.5
    Gm = intersect(Gm, itranspose(Gm))
    # z from the center gradient (float), fixed over the box
    Wc = np.linalg.inv(Ot)
    Gc = 0.5 * (Wc - Wc @ Rc @ Wc)
    Zlo, Zhi = Gm.lo.copy(), Gm.hi.copy()
    const = IA.point(np.zeros(K))
    for c in TO:
        p, q = PAIRS[c - 3]
        z = -Gc[:, p, q]
        Zi = IA.point(z)
        Zlo[:, p, q] = Zlo[:, q, p] = (Gm[:, p, q] + Zi).lo
        Zhi[:, p, q] = Zhi[:, q, p] = (Gm[:, p, q] + Zi).hi
        twoabs = IA.point(np.abs(z)) * 2.0
        excess = IA(np.maximum((twoabs - c_lo).lo, 0.0), np.maximum((twoabs - c_lo).hi, 0.0))
        const = const + Zi * IA.point(Ot[:, p, q]) * 2.0 - excess * dp
    Gp = IA(Zlo, Zhi)
    GR = imatmul(Gp, R)
    trq = None
    for a in range(3):
        for b in range(3):
            t = GR[:, a, b] * GR[:, b, a]
            trq = t if trq is None else trq + t
    mR = np.where(np.isfinite(best_m), best_m, 1.0)
    Q = (trq / IA.point(2.0 * mR)).hi
    Q = np.maximum(Q, 0.0)
    lb = dn((tot + const).lo - Q)
    return np.where(valid & np.isfinite(lb), lb, -np.inf)


def graded_bound_B(ctx, region, lo, hi, mu_hi):
    """F >= lambda|L| + fsat + max_pairs graded_pair_bound + sum_{T_B} lambda q(min |B_k|)."""
    K = lo.shape[0]
    Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(region.fwd):
        Blo[:, i, j] = lo[:, c]; Bhi[:, i, j] = hi[:, c]
    I3 = np.broadcast_to(np.eye(3), (K, 3, 3))
    A = IA(I3 - Bhi, I3 - Blo)
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    R = imatmul(itranspose(A), imatmul(Sb, A))
    R = intersect(R, itranspose(R))
    g = np.zeros(K)
    for c in range(3, 6):
        if c in region.L:
            continue
        p, q = PAIRS[c - 3]
        g = np.maximum(g, graded_pair_bound(ctx, R, p, q, mu_hi))
    g = np.maximum(g, quad_kl_bound(ctx, region, R, mu_hi))
    tot = ctx.fsat + ctx.lam * float(len(region.L)) + IA(g, g)
    for c in range(3):
        if c in region.L:
            continue
        mabs = np.where((lo[:, c] <= 0) & (hi[:, c] >= 0), 0.0, np.minimum(np.abs(lo[:, c]), np.abs(hi[:, c])))
        tot = tot + IA((ctx.lam * IA.point(qlb(mabs, ctx.delta))).lo, (ctx.lam * IA.point(qlb(mabs, ctx.delta))).lo)
    return tot.lo


def prune_B(ctx, region, lo, hi, mu_lo, mu_hi):
    """True where every point of the region with B in the box has F >= s*: some small pair
    (p,q) in T_Omega has |Omega_pq| <= delta' incompatible with mu_lo R <= Omega <= mu_hi R."""
    K = lo.shape[0]
    Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(region.fwd):
        Blo[:, i, j] = lo[:, c]; Bhi[:, i, j] = hi[:, c]
    I3 = np.broadcast_to(np.eye(3), (K, 3, 3))
    A = IA(I3 - Bhi, I3 - Blo)
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    R = imatmul(itranspose(A), imatmul(Sb, A))
    R = intersect(R, itranspose(R))
    out = np.zeros(K, bool)
    dp = ctx.delta_hi
    for c in range(3, 6):
        if c in region.L:
            continue
        p, q = PAIRS[c - 3]
        out |= pair_prune(R, IA(np.full(K, -dp), np.full(K, dp)), p, q, mu_lo, mu_hi)
    return out


def _profile_omega_ia(R, U):
    """Closed-form profile Omega(R) for pattern U, interval version (R: IA (K,3,3))."""
    K = R.shape[0]
    Z = IA.point(np.zeros((K, 3, 3)))
    lo, hi = Z.lo.copy(), Z.hi.copy()
    O = IA(lo, hi)

    def put(i, j, v):
        O.lo[:, i, j] = v.lo; O.hi[:, i, j] = v.hi
        if i != j:
            O.lo[:, j, i] = v.lo; O.hi[:, j, i] = v.hi
    if len(U) == 3:
        return R
    for v in range(3):
        put(v, v, R[:, v, v])
    if len(U) == 1:
        (i, j), = U
        put(i, j, R[:, i, j])
    elif len(U) == 2:
        (i, k), = [pr for pr in PAIRS if pr not in U]
        j = 3 - i - k
        det = R[:, i, i] * R[:, k, k] - R[:, i, k].sqr()
        if np.any(det.lo <= 0):
            return None
        b_i = (R[:, k, k] * R[:, i, j] - R[:, i, k] * R[:, k, j]) / det
        b_k = (R[:, i, i] * R[:, k, j] - R[:, i, k] * R[:, i, j]) / det
        rss = R[:, j, j] - b_i * R[:, i, j] - b_k * R[:, k, j]
        put(i, j, b_i * R[:, i, i])
        put(k, j, b_k * R[:, k, k])
        put(j, j, b_i.sqr() * R[:, i, i] + b_k.sqr() * R[:, k, k] + rss)
    return O


def _profile_value_ia(R, U, logdetS):
    """1/2 log det Omega^(R) + 3/2 (closed form; R: IA (K,3,3) at points)."""
    if len(U) == 3:
        return logdetS * 0.5 + 1.5
    if len(U) == 0:
        return (R[:, 0, 0].log() + R[:, 1, 1].log() + R[:, 2, 2].log()) * 0.5 + 1.5
    if len(U) == 1:
        (i, j), = U
        k = 3 - i - j
        g = R[:, i, i] * R[:, j, j] - R[:, i, j].sqr()
        return (g.log() + R[:, k, k].log()) * 0.5 + 1.5
    (i, k), = [pr for pr in PAIRS if pr not in U]
    one = IA.point(np.ones(R.shape[0]))
    rho2 = R[:, i, k].sqr() / (R[:, i, i] * R[:, k, k])
    return (logdetS - (one - rho2).log()) * 0.5 + 1.5


def bound_P(ctx, region, lo, hi, mus):
    """Profile bound.  For B in the box, if (a) Omega^(B) (closed-form profile of the
    region's bidirected pattern L_Omega) satisfies Omega^(B) <= mu' R(B) and (b) every
    small-pair score satisfies |dl/dOmega_pq| <= c = lambda/delta' at Omega^(B), then
    Omega^(B) solves the convex problem min_{Omega <= mu' R} l + c sum_T |Omega_pq| (KKT),
    so the inner minimum equals the profile value f(B).  Then
        F >= lambda|L| + min_B [f(B) + sum_{T_B} lambda q(B_k)],
    bounded by the mean-value form with df/dB = -(S A Omega^(B)^{-1}) (envelope theorem).
    Returns lb (-inf where (a) or (b) is not certified)."""
    K = lo.shape[0]
    lam = ctx.lam
    nL = len(region.L)
    fwd = region.fwd
    LB_dims = [c for c in range(3) if c in region.L]
    TB_dims = [c for c in range(3) if c not in region.L]
    TO = [c for c in range(3, 6) if c not in region.L]
    U = region.U
    Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        Blo[:, i, j] = lo[:, c]; Bhi[:, i, j] = hi[:, c]
    I3 = np.broadcast_to(np.eye(3), (K, 3, 3))
    A = IA(I3 - Bhi, I3 - Blo)
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    SA = imatmul(Sb, A)
    R = imatmul(itranspose(A), SA)
    R = intersect(R, itranspose(R))
    lbad = np.full(K, -np.inf)
    Oh = _profile_omega_ia(R, U)
    if Oh is None:
        return lbad
    ok, W = neumann_inverse(Oh.lo, Oh.hi)
    ok &= cholesky_pd(IA.point(0.5 * Oh.lo + 0.5 * Oh.hi))
    # (a) validity with the smallest admissible mu'
    valid = np.zeros(K, bool)
    for m_R, mu in mus:
        valid |= cholesky_pd(R * mu - Oh)
    ok &= valid
    # (b) small-pair scores
    WRW = imatmul(imatmul(W, R), W)
    Mg = W - WRW
    Mg = intersect(Mg, itranspose(Mg))
    c_lo = float((lam / IA.point(np.array(ctx.delta_hi))).lo)
    for c in TO:
        p, q = PAIRS[c - 3]
        ok &= Mg.mag()[:, p, q] <= c_lo
    if not np.any(ok):
        return lbad
    GB = -imatmul(SA, W)
    G = [GB[:, i, j] for (i, j) in fwd]
    cB = 0.5 * (lo + hi)
    x0 = cB.copy()
    for c in LB_dims:
        x0[:, c] = np.where(G[c].lo >= 0, lo[:, c], np.where(G[c].hi <= 0, hi[:, c], cB[:, c]))
    for c in TB_dims:
        x0[:, c] = np.clip(0.0, lo[:, c], hi[:, c])
    B0 = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        B0[:, i, j] = x0[:, c]
    A0 = IA.point(np.eye(3) - B0)
    R0 = imatmul(itranspose(A0), imatmul(Sb, A0))
    R0 = intersect(R0, itranspose(R0))
    f0 = _profile_value_ia(R0, U, ctx.logdetS)
    tot = f0 + lam * float(nL)
    for c in LB_dims:
        tot = tot + G[c] * (IA(lo[:, c], hi[:, c]) - IA.point(x0[:, c]))
    for c in TB_dims:
        a, b = lo[:, c], hi[:, c]
        best = None
        for tc in [a, b, np.zeros_like(a)]:
            inside = (tc >= a) & (tc <= b)
            dt = IA.point(tc) - IA.point(x0[:, c])
            lin = (G[c] * dt).lo
            pen = (lam * IA.point(qlb(np.abs(tc), ctx.delta))).lo
            val = np.where(inside, dn(pen + lin), np.inf)
            best = val if best is None else np.minimum(best, val)
        tot = tot + IA(best, best)
    return np.where(ok, tot.lo, -np.inf)


def bound_B_multi(ctx, region, lo, hi, mus):
    """Best of bound_B over a list of (m_R, mu') pairs (each mu' >= mu_hi(tau), < 2).
    Any mu' in [mu_hi(tau), 2) is admissible: Omega not <= mu' R still forces l_S >= fsat + tau."""
    best = None
    for m_R, mu in mus:
        lb, valid, gm, Q = bound_B(ctx, region, lo, hi, m_R, mu)
        if best is None:
            best = [lb, valid, gm, Q]
        else:
            upd = lb > best[0]
            best[0] = np.where(upd, lb, best[0])
            best[1] = best[1] | valid
            best[2] = np.where(upd[:, None], gm, best[2])
            best[3] = np.where(upd, Q, best[3])
    return best


def bound_B(ctx, region, lo, hi, m_R, mu_hi):
    """Rigorous lower bounds of F over B-boxes (K, 3) of region (pi, L)."""
    K = lo.shape[0]
    lam = ctx.lam
    nL = len(region.L)
    fwd = region.fwd
    LB_dims = [c for c in range(3) if c in region.L]
    TB_dims = [c for c in range(3) if c not in region.L]
    TO = [c for c in range(3, 6) if c not in region.L]
    U = region.U
    # interval B, A, R over the boxes
    Blo = np.zeros((K, 3, 3)); Bhi = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        Blo[:, i, j] = lo[:, c]; Bhi[:, i, j] = hi[:, c]
    I3 = np.broadcast_to(np.eye(3), (K, 3, 3))
    A = IA(I3 - Bhi, I3 - Blo)
    Sb = IA.point(np.broadcast_to(ctx.S, (K, 3, 3)).copy())
    SA = imatmul(Sb, A)
    R = imatmul(itranspose(A), SA)
    R = intersect(R, itranspose(R))
    # reference Omega~ at the box center (float, then treated as an exact point)
    cB = 0.5 * (lo + hi)
    Bc = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        Bc[:, i, j] = cB[:, c]
    Ac = np.eye(3) - Bc
    Rc = np.einsum("kai,ab,kbj->kij", Ac, ctx.S, Ac)
    Ot = profile_omega(Rc, U)
    Ot = 0.5 * (Ot + np.swapaxes(Ot, 1, 2))
    for c in TO:
        p, q = PAIRS[c - 3]
        Ot[:, p, q] = Ot[:, q, p] = 0.0
    Oti = IA.point(Ot)
    valid = cholesky_pd(Oti)
    # Omega~ <= mu_hi R(B) for every B in the box
    valid &= cholesky_pd(R * mu_hi - Oti)
    dO = det3(Oti)
    dOs = IA(np.where(dO.lo > 0, dO.lo, 1.0), np.where(dO.lo > 0, dO.hi, 1.0))
    valid &= dO.lo > 0
    W = adj3(Oti) / IA(dOs.lo[:, None, None], dOs.hi[:, None, None])
    W = intersect(W, itranspose(W))
    # phi gradient in B over the box: d phi / d B = -(S A W)
    GB = -imatmul(SA, W)
    G = []
    for c, (i, j) in enumerate(fwd):
        G.append(GB[:, i, j])
    # expansion point: L_B dims monotone corner, T_B dims closest to 0
    x0 = cB.copy()
    for c in LB_dims:
        x0[:, c] = np.where(G[c].lo >= 0, lo[:, c], np.where(G[c].hi <= 0, hi[:, c], cB[:, c]))
    for c in TB_dims:
        x0[:, c] = np.clip(0.0, lo[:, c], hi[:, c])
    B0 = np.zeros((K, 3, 3))
    for c, (i, j) in enumerate(fwd):
        B0[:, i, j] = x0[:, c]
    A0 = IA.point(np.eye(3) - B0)
    R0 = imatmul(itranspose(A0), imatmul(Sb, A0))
    tr = None
    for a in range(3):
        for b in range(3):
            t = W[:, a, b] * R0[:, b, a]
            tr = t if tr is None else tr + t
    phi0 = dOs.log() * 0.5 + tr * 0.5
    tot = phi0 + lam * float(nL)
    for c in LB_dims:
        tot = tot + G[c] * (IA(lo[:, c], hi[:, c]) - IA.point(x0[:, c]))
    for c in TB_dims:
        a, b = lo[:, c], hi[:, c]
        best = None
        for tc in [a, b, np.zeros_like(a)]:
            inside = (tc >= a) & (tc <= b)
            dt = IA.point(tc) - IA.point(x0[:, c])
            lin = (G[c] * dt).lo
            pen = (lam * IA.point(qlb(np.abs(tc), ctx.delta))).lo
            val = np.where(inside, dn(pen + lin), np.inf)
            best = val if best is None else np.minimum(best, val)
        tot = tot + IA(best, best)
    # quadratic-model correction over Omega
    WRW = imatmul(imatmul(W, R), W)
    Gm = (W - WRW) * 0.5
    Gm = intersect(Gm, itranspose(Gm))
    cthr = float((lam / IA.point(np.array(ctx.delta_hi))).lo) * 0.5
    glo, ghi = Gm.lo.copy(), Gm.hi.copy()
    for c in TO:
        p, q = PAIRS[c - 3]
        sl, sh = soft(glo[:, p, q], ghi[:, p, q], cthr)
        glo[:, p, q] = glo[:, q, p] = sl
        ghi[:, p, q] = ghi[:, q, p] = sh
    Gp = IA(glo, ghi)
    GR = imatmul(Gp, R)
    trq = None
    for a in range(3):
        for b in range(3):
            t = GR[:, a, b] * GR[:, b, a]
            trq = t if trq is None else trq + t
    Q = (trq / (2.0 * m_R)).hi
    Q = np.maximum(Q, 0.0)
    lb = dn(tot.lo - Q)
    lb = np.where(valid, lb, -np.inf)
    gmag = np.stack([g.mag() for g in G], axis=1)
    return lb, valid, gmag, Q


def certify_region_B(ctx, region, max_boxes=2_000_000, max_iter=300, batch=100_000, debug=None, use_ref=True):
    t0 = time.monotonic()
    nL = len(region.L)
    tau = ctx.s_star - float((ctx.lam * float(nL)).lo) - float(ctx.fsat.lo) - 1e-12
    if tau >= 0.0966 or tau <= 1e-12:
        return None
    mu_lo, mu_hi = ctx.mu_bounds(max(tau, 1e-300))
    if not mu_hi < 1.9:
        return None
    mus = []
    for mu in sorted({mu_hi, min(1.9, mu_hi * 1.05), min(1.9, max(mu_hi, 1.3)), 1.6, 1.9}):
        if mu < mu_hi:
            continue
        MH = IA.point(np.array(mu))
        mR = ((IA.point(np.array(1.0)) / MH - 0.5) / MH.sqr()).lo
        if mR > 0:
            mus.append((mR, mu))
    dp = ctx.delta_hi
    lo = np.array([[(-MB if c in region.L else -dp) for c in range(3)]])
    hi = np.array([[(MB if c in region.L else dp) for c in range(3)]])
    total = 0
    npruned = 0
    ngraded = 0
    nref = 0
    min_lb = math.inf
    it = 0
    stuck = False
    while len(lo):
        it += 1
        if it > max_iter or total > max_boxes:
            stuck = True
            break
        nlo, nhi = [], []
        for s0 in range(0, len(lo), batch):
            l_, h_ = lo[s0:s0 + batch], hi[s0:s0 + batch]
            total += len(l_)
            pr = prune_B(ctx, region, l_, h_, mu_lo, mu_hi)
            npruned += int(pr.sum())
            lb = np.full(len(l_), -np.inf)
            valid = np.zeros(len(l_), bool)
            gm = np.zeros((len(l_), 3))
            act = ~pr
            if np.any(act):
                lbP = bound_P(ctx, region, l_[act], h_[act], mus)
                lb[act] = lbP
                valid[act] = np.isfinite(lbP)
                need = np.nonzero(act)[0][lbP < ctx.target]
                if len(need):
                    lbB, vB, gB, _ = bound_B_multi(ctx, region, l_[need], h_[need], mus)
                    lb[need] = np.maximum(lb[need], lbB)
                    valid[need] |= vB
                    gm[need] = gB
            need = np.nonzero(~pr & (lb < ctx.target))[0]
            if len(need):
                lbg = graded_bound_B(ctx, region, l_[need], h_[need], mu_hi)
                ngraded += int(np.sum(lbg >= ctx.target))
                lb[need] = np.maximum(lb[need], lbg)
            need = np.nonzero(~pr & (lb < ctx.target))[0]
            if len(need) and use_ref:
                lbr = bound_REF(ctx, region, l_[need], h_[need], mus)
                nref += int(np.sum(lbr >= ctx.target))
                lb[need] = np.maximum(lb[need], lbr)
                valid[need] |= np.isfinite(lbr)
            done = pr | (lb >= ctx.target)
            lb = np.where(pr, np.inf, lb)
            if np.any(done):
                min_lb = min(min_lb, float(np.min(lb[done])))
            keep = ~done
            if not np.any(keep):
                continue
            l_, h_, gm = l_[keep], h_[keep], gm[keep]
            w = h_ - l_
            sens = w * np.where(np.isfinite(gm), gm, 1e300) + 1e-300 * w
            # invalid boxes (profile/reference bound not certified): split by width relative
            # to a scale of 0.25 for large coordinates and delta'/4 for small ones
            # sensitivity of R(B) to each coordinate (heuristic, float) plus the penalty slope
            # 2 lambda / delta for small coordinates
            Bc_ = np.zeros((len(l_), 3, 3))
            cc_ = 0.5 * (l_ + h_)
            for c, (i, j) in enumerate(region.fwd):
                Bc_[:, i, j] = cc_[:, c]
            SAc = np.einsum("ab,kbj->kaj", ctx.S, np.eye(3) - Bc_)
            sR = np.zeros_like(w)
            for c, (p_, q_) in enumerate(region.fwd):
                # dR_ab/dB_pq = -d_aq (SA)_pb - d_bq (SA)_pa  ->  sum_ab |.| ~ 2 sum_b |(SA)_pb|
                sR[:, c] = 2.0 * np.sum(np.abs(SAc[:, p_, :]), axis=1)
                if c not in region.L:
                    sR[:, c] += 2.0
            k = np.where(valid[keep], np.argmax(sens, axis=1), np.argmax(w * sR, axis=1))
            r = np.arange(len(l_))
            a, b = l_[r, k], h_[r, k]
            m = 0.5 * (a + b)
            m = np.where((a < 0) & (b > 0) & (np.abs(a + b) < 0.5 * (b - a)) & ((b - a) > 1e-3), 0.0, m)
            l2, h2 = l_.copy(), h_.copy()
            h_[r, k] = m
            l2[r, k] = m
            nlo += [l_, l2]
            nhi += [h_, h2]
        lo = np.concatenate(nlo) if nlo else np.zeros((0, 3))
        hi = np.concatenate(nhi) if nhi else np.zeros((0, 3))
        if len(lo) and np.min(np.max(hi - lo, axis=1)) < 1e-14:
            stuck = True
            break
    if debug is not None:
        debug["lo"], debug["hi"], debug["mu"] = lo, hi, (mu_lo, mu_hi, mus)
    return dict(order="".join(str(v + 1) for v in region.order), L=",".join(map(str, region.L)),
                H_D=str(region.D), H_U=str(region.U), method="B-space", certified=not stuck,
                boxes=total, iterations=it, min_lb=min_lb, remaining=int(len(lo)),
                mu_hi=mu_hi, tau=tau, pair_pruned=npruned, graded_certified=ngraded, ref_certified=nref, runtime_s=time.monotonic() - t0)

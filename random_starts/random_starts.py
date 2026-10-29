"""Random starts of the alternating algorithm on three vertices.

Section 7.3 of the main article and Section S18 of the Supplementary Material.
Algorithm 1 (exact B-step by dynamic programming over vertex orders, eq. (6),
followed by the Omega-step evaluated at the updated B, eq. (7)) is started from
random feasible points, from B = 0, Omega = diag(S), and from the refits of the
50 best supports, on the three designs of Section 7.1.  For each run we record
whether the end point, thresholded at delta, is a minimum-edge support.

Conventions as in ../global_selection/global_selection.py: parent-row
parametrization Sigma = (I-B)^{-T} Omega (I-B)^{-1}; B[i, j] != 0 is i -> j;
Omega[i, j] != 0 (i < j) is i <-> j; loss
l_S = 0.5 log det Omega + 0.5 tr(Omega^{-1} (I-B)^T S (I-B)); quasi-MCP
q_delta(t) = 2|t|/delta - t^2/delta^2 for |t| <= delta and 1 beyond; the
penalty sums q_delta over directed coordinates and unordered off-diagonal
Omega entries.

The cycle (penalty, scalar_prox, arrow_prox, NoiseProx, loss_grad) is that of
../local_convergence/alternating.py with a configurable spectral box.  Data
generation, the support enumeration and the constrained-likelihood solver for
the refit starts come from ../global_selection/global_selection.py, whose
results/population.json must exist.

All computations are in floating point; none of them is a certificate.

Usage:
    python3 random_starts.py [--workers W]           main study (tuning checks,
                                                      step calibration, 63,900 runs)
    python3 random_starts.py --small-starts           starts ten times smaller
    python3 random_starts.py --resummarize            rebuild summary tables from runs
    python3 random_starts.py --reuse-steps --datasets 1 --out DIR
                                                      fast check: first sample of each
                                                      cell with the stored step sizes
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import gzip
import itertools
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "global_selection"))
import global_selection as gs  # noqa: E402

RESULTS = HERE / "results"
POPULATION = HERE.parent / "global_selection" / "results" / "population.json"
P = gs.P
PAIRS = gs.PAIRS
OFFDIAG = ~np.eye(P, dtype=bool)
IU = np.triu_indices(P, 1)

CONFIG = dict(
    study="random starts of the alternating algorithm (Algorithm 1), p = 3",
    p=P,
    designs=list(gs.CONFIG["designs"]),
    design_parameters={k: v for k, v in gs.CONFIG["designs"].items()},
    coefficient_bound=gs.CONFIG["coefficient_bound"],
    omega_spectrum=gs.CONFIG["omega_spectrum"],
    lam=0.01,
    delta=0.1,
    # The magnitude-only sufficient condition in the proof of Theorem 5 (delta below the
    # smallest e-th largest edge magnitude on the fiber, then lambda below eta_k/(e-k))
    # does not hold for G3 at delta = 0.1: its fiber contains an exact representation
    # whose third-largest edge magnitude is 0.0757.  The tuning is therefore checked with
    # the penalty-aware bound eta_pen (see admissibility()).  The scan records the
    # magnitude-only lambda_0(delta) for each design.
    delta_scan=[0.1, 0.05, 0.04, 0.03, 0.02],
    # Step sizes, set from the Lipschitz calibration (see calibrate()).
    alpha=None,
    beta=None,
    ns=[250, 1000, 4000],
    datasets=100,
    random_starts=20,
    refit_starts=50,
    max_cycles=20000,
    step_tolerance=1e-8,
    reference_tolerance=1e-8,
    Ks=[1, 2, 5, 10, 20],
    start_B_sd=0.5,
    start_omega_offdiag_sd=0.5,
    start_omega_min_corr_eig=0.5,
    start_omega_shrink=0.8,
    master_seed=gs.CONFIG["master_seed"],
    data_seed_rule="numpy default_rng(SeedSequence([master_seed, design_index, n, dataset])) "
                   "(rule of the global-selection study, whose sample sizes differ from these)",
    refit_solver_seed_rule="numpy default_rng(SeedSequence([master_seed, design_index, n, dataset, 1]))",
    start_seed_rule="numpy default_rng(SeedSequence([master_seed, design_index, n, dataset, 2, start]))",
    threshold="support = coordinates with |x| >= delta (also recorded: raw nonzero and |x| >= delta/2)",
    covariance="centered empirical covariance, divisor n (gs.sample_covariance)",
)


# ----------------------------------------------------------------------------
# One cycle of Algorithm 1, eqs. (6) and (7)
# ----------------------------------------------------------------------------

def penalty(x, delta):
    a = np.minimum(np.abs(x) / delta, 1.0)
    return 2 * a - a * a


def scalar_prox(z, gamma, lam, delta, bound=np.inf):
    """Unique scalar quasi-MCP prox for gamma < delta^2 / (2 lambda)."""
    if not 0 < gamma < delta * delta / (2 * lam):
        raise ValueError("Scalar proximal model is not strongly convex")
    v = np.abs(z)
    cut = 2 * gamma * lam / delta
    slope = 1 - 2 * gamma * lam / delta ** 2
    ans = np.where(v <= cut, 0, np.where(v < delta, (v - cut) / slope, v))
    return np.clip(np.sign(z) * ans, -bound, bound)


_MASKS = None


def _mask_table(p):
    global _MASKS
    if _MASKS is None or _MASKS[0] != p:
        tab = []
        for mask in range(1, 1 << p):
            nodes = [i for i in range(p) if mask & (1 << i)]
            tab.append((mask, [(j, mask ^ (1 << j), [i for i in nodes if i != j]) for j in nodes]))
        _MASKS = (p, tab)
    return _MASKS[1]


def arrow_prox(z, gamma, lam, delta, bound=2.0):
    """B-step (6): exact optimization over all acyclic B by dynamic programming over vertex subsets."""
    p = len(z)
    vals = scalar_prox(z, gamma, lam, delta, bound)
    np.fill_diagonal(vals, 0)
    gain = z * z / (2 * gamma) - ((vals - z) ** 2 / (2 * gamma) + lam * penalty(vals, delta))
    np.fill_diagonal(gain, 0)
    dp = np.full(1 << p, -np.inf)
    last = np.zeros(1 << p, dtype=int)
    dp[0] = 0
    for mask, items in _mask_table(p):
        for j, prev, others in items:
            score = dp[prev] + sum(gain[i, j] for i in others)
            if score > dp[mask]:
                dp[mask], last[mask] = score, j
    order, mask = [], (1 << p) - 1
    while mask:
        j = int(last[mask])
        order.append(j)
        mask ^= 1 << j
    order.reverse()
    out = np.zeros_like(z)
    for k, j in enumerate(order):
        for i in order[:k]:
            out[i, j] = vals[i, j]
    return out, float(dp[-1]), order


class NoiseProx:
    """Omega-step (7): scalar solution when inside the spectral box, else a conic solve."""

    def __init__(self, p, gamma, lam, delta, floor, ceiling):
        self.p, self.gamma, self.lam, self.delta = p, gamma, lam, delta
        self.floor, self.ceiling = floor, ceiling
        self.statuses = {}
        self.problem = None

    def conic(self, z):
        import cvxpy as cp
        if self.problem is None:
            self.z = cp.Parameter((self.p, self.p), symmetric=True)
            self.o = cp.Variable((self.p, self.p), symmetric=True)
            g, l, d = self.gamma, self.lam, self.delta
            terms = [cp.sum_squares(cp.diag(self.o) - cp.diag(self.z)) / (2 * g)]
            for i in range(self.p):
                for j in range(i + 1, self.p):
                    t = self.o[i, j]
                    terms += [(1 / g - l / d ** 2) * cp.square(t) - 2 * self.z[i, j] * t / g
                              + 2 * l * cp.abs(t) / d
                              + l * cp.square(cp.pos(cp.abs(t) - d)) / d ** 2]
            self.problem = cp.Problem(cp.Minimize(sum(terms)),
                                      [self.o >> self.floor * np.eye(self.p),
                                       self.o << self.ceiling * np.eye(self.p)])
            assert self.problem.is_dcp()
        self.z.value = (z + z.T) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.problem.solve(solver="CLARABEL", tol_gap_abs=1e-12,
                               tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=200)
        status = self.problem.status
        self.statuses[status] = self.statuses.get(status, 0) + 1
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("Noise proximal solve: " + status)
        o = np.asarray(self.o.value)
        o = (o + o.T) / 2
        # Tiny infeasibility from the interior-point tolerance: pull the
        # spectrum back into the box (changes o by at most ~1e-10).
        ev, U = np.linalg.eigh(o)
        if ev[0] < self.floor or ev[-1] > self.ceiling:
            self.statuses["spectrum_repaired"] = self.statuses.get("spectrum_repaired", 0) + 1
            o = (U * np.clip(ev, self.floor, self.ceiling)) @ U.T
        return o

    def __call__(self, z):
        z = (z + z.T) / 2
        # Each unordered pair appears twice in the Frobenius square: scalar gamma is beta/2,
        # threshold beta*lambda/delta.
        out = scalar_prox(z, self.gamma / 2, self.lam, self.delta)
        np.fill_diagonal(out, np.diag(z))
        eig = np.linalg.eigvalsh(out)
        if eig[0] >= self.floor and eig[-1] <= self.ceiling:
            self.statuses["analytic_interior"] = self.statuses.get("analytic_interior", 0) + 1
            return out
        return self.conic(z)


def loss_grad(b, o, s):
    a = np.eye(len(b)) - b
    inv = np.linalg.inv(o)
    residual = a.T @ s @ a
    value = 0.5 * (np.linalg.slogdet(o)[1] + np.trace(inv @ residual))
    gb = -s @ a @ inv
    go = 0.5 * (inv - inv @ residual @ inv)
    return float(value), gb, (go + go.T) / 2


def pen(b, o, delta):
    return float(penalty(b[OFFDIAG], delta).sum() + penalty(o[IU], delta).sum())


def criterion(b, o, s, lam, delta):
    return loss_grad(b, o, s)[0] + lam * pen(b, o, delta)


# ----------------------------------------------------------------------------
# One run of Algorithm 1
# ----------------------------------------------------------------------------

def run_alternating(s, b0, o0, cfg, prox=None):
    lam, delta, alpha, beta = cfg["lam"], cfg["delta"], cfg["alpha"], cfg["beta"]
    M = cfg["coefficient_bound"]
    lo, hi = cfg["omega_spectrum"]
    if prox is None:
        prox = NoiseProx(P, beta, lam, delta, lo, hi)
    b, o = b0.copy(), o0.copy()
    started = time.monotonic()
    val, gb, go = loss_grad(b, o, s)
    pb, po = penalty(b[OFFDIAG], delta).sum(), penalty(o[IU], delta).sum()
    f = val + lam * (pb + po)
    f_start = f
    max_increase = -np.inf
    curv_b = curv_o = 0.0          # largest realized curvature x step (descent needs < 1)
    min_eig = np.linalg.eigvalsh(o)[0]
    status = "cycle_cap"
    step = np.nan
    t = 0
    orders = set()
    n_conic0 = sum(v for k, v in prox.statuses.items() if k != "analytic_interior")
    for t in range(1, cfg["max_cycles"] + 1):
        bn, _, order = arrow_prox(b - alpha * gb, alpha, lam, delta, M)
        val_mid, _, go_mid = loss_grad(bn, o, s)
        db = bn - b
        nb2 = float(np.sum(db * db))
        if nb2 > 1e-10:
            curv_b = max(curv_b, alpha * 2 * (val_mid - val - float(np.sum(gb * db))) / nb2)
        on = prox(o - beta * go_mid)
        val_new, gb_new, go_new = loss_grad(bn, on, s)
        do = on - o
        no2 = float(np.sum(do * do))
        if no2 > 1e-10:
            curv_o = max(curv_o, beta * 2 * (val_new - val_mid - float(np.sum(go_mid * do))) / no2)
        f_new = val_new + lam * (penalty(bn[OFFDIAG], delta).sum() + penalty(on[IU], delta).sum())
        max_increase = max(max_increase, f_new - f)
        step = math.sqrt(nb2 + no2)
        b, o, val, gb, go, f = bn, on, val_new, gb_new, go_new, f_new
        if t <= 3 or t % 500 == 0:
            min_eig = min(min_eig, np.linalg.eigvalsh(o)[0])
        if step < cfg["step_tolerance"]:
            status = "step_tolerance"
            orders.add(tuple(order))
            break
    min_eig = min(min_eig, np.linalg.eigvalsh(o)[0])
    n_conic = sum(v for k, v in prox.statuses.items() if k != "analytic_interior") - n_conic0
    return b, o, dict(objective=float(f), start_objective=float(f_start), cycles=t, status=status,
                      final_step=step, max_objective_increase=float(max_increase),
                      max_curvature_ratio_B=float(curv_b), max_curvature_ratio_Omega=float(curv_o),
                      min_omega_eig=float(min_eig), noise_conic_calls=int(n_conic),
                      runtime_seconds=time.monotonic() - started)


def describe(b, o, cfg, emin, e_star):
    delta = cfg["delta"]
    out = {}
    for tag, cut in (("raw", 0.0), ("thr", delta), ("half", delta / 2)):
        mb = (np.abs(b) >= cut) & (np.abs(b) > 0) & OFFDIAG
        mo = (np.abs(o) >= cut) & (np.abs(o) > 0) & OFFDIAG
        sup = gs.support_from_masks(mb, mo)
        out[f"support_{tag}"] = sup.name
        if tag == "thr":
            out.update(support_count=sup.count, is_emin=sup.index in emin, bowed=sup.bowed,
                       more_than_e_star=sup.count > e_star, fewer_than_e_star=sup.count < e_star,
                       support_index=sup.index)
    coords = np.r_[np.abs(b[OFFDIAG]), np.abs(o[IU])]
    nz = coords[coords > 0]
    out["min_nonzero_coord"] = float(nz.min()) if len(nz) else np.nan
    out["n_coords_in_(0,delta)"] = int(np.sum((coords > 0) & (coords < delta)))
    out["B_at_bound"] = bool(np.any(np.abs(b) >= cfg["coefficient_bound"] - 1e-9))
    ev = np.linalg.eigvalsh(o)
    lo, hi = cfg["omega_spectrum"]
    out["Omega_at_spectral_bound"] = bool(ev[0] <= lo + 1e-7 or ev[-1] >= hi - 1e-7)
    return out


# ----------------------------------------------------------------------------
# Starts
# ----------------------------------------------------------------------------

def random_start(S, rng, cfg):
    lo, hi = cfg["omega_spectrum"]
    order = rng.permutation(P)
    B = np.zeros((P, P))
    for a in range(P):
        for c in range(a + 1, P):
            B[order[a], order[c]] = rng.normal(scale=cfg["start_B_sd"])
    n_clip = int(np.sum(np.abs(B) > cfg["coefficient_bound"]))
    B = np.clip(B, -cfg["coefficient_bound"], cfg["coefficient_bound"])
    d = np.sqrt(np.diag(S))
    G = np.zeros((P, P))
    for i, j in PAIRS:
        G[i, j] = G[j, i] = rng.normal(scale=cfg["start_omega_offdiag_sd"])
    scale = 1.0
    while True:
        C = np.eye(P) + scale * G
        O = C * np.outer(d, d)
        ev = np.linalg.eigvalsh(O)
        if np.linalg.eigvalsh(C)[0] >= cfg["start_omega_min_corr_eig"] and ev[0] >= lo and ev[-1] <= hi:
            break
        scale *= cfg["start_omega_shrink"]
        if scale < 1e-6:
            scale = 0.0
            O = np.diag(np.diag(S))
            break
    return B, O, dict(start_order="".join(str(v + 1) for v in order), start_omega_scale=scale,
                      start_B_clipped=n_clip)


# ----------------------------------------------------------------------------
# Admissibility of the fixed tuning (population)
# ----------------------------------------------------------------------------

FWD = {od: [(od[a], od[c]) for a in range(P) for c in range(a + 1, P)]
       for od in itertools.permutations(range(P))}


def _coords_exact(x, Sigma, od):
    B = np.zeros((P, P))
    for v, e in zip(x, FWD[od]):
        B[e] = v
    A = np.eye(P) - B
    O = A.T @ Sigma @ A
    return np.r_[x, [O[pr] for pr in PAIRS]], O


def delta0(Sigma, e, cfg, rng, starts=12):
    """min over exact representations in K of the e-th largest |edge coordinate|.

    Every exact representation with order od has Omega = (I-B)^T Sigma (I-B) for
    its forward B, so the fiber is parametrized by the forward B entries.  The e-th
    largest of 6 magnitudes equals min over sets T of e-1 'free' coordinates of the
    max over the others; each (od, T) is an epigraph problem solved by SLSQP.
    """
    M = cfg["coefficient_bound"]
    lo, hi = cfg["omega_spectrum"]
    best = (np.inf, None)
    for od in FWD:
        for T in itertools.combinations(range(6), e - 1):
            rest = [k for k in range(6) if k not in T]

            def cons(z):
                c, O = _coords_exact(z[:3], Sigma, od)
                ev = np.linalg.eigvalsh(O)
                return np.r_[z[3] - c[rest], z[3] + c[rest], ev - lo, hi - ev]
            for k in range(starts):
                x0 = rng.uniform(-M, M, 3) if k else np.zeros(3)
                c0, _ = _coords_exact(x0, Sigma, od)
                z0 = np.r_[x0, np.max(np.abs(c0[rest])) + 1e-3]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = minimize(lambda z: z[3], z0, method="SLSQP",
                                 bounds=[(-M, M)] * 3 + [(0, None)],
                                 constraints=[dict(type="ineq", fun=cons)],
                                 options=dict(ftol=1e-14, maxiter=500))
                if np.min(cons(r.x)) < -1e-9:
                    continue
                c, _ = _coords_exact(r.x[:3], Sigma, od)
                val = float(np.sort(np.abs(c))[::-1][e - 1])
                if val < best[0]:
                    best = (val, dict(order="".join(str(v + 1) for v in od),
                                      B_forward=r.x[:3].tolist(), coords=c.tolist()))
    return best


def eta_k(Sigma, k, delta, cfg, rng, starts=6):
    """inf of l(theta) - l_* over theta in K with at most k edge coordinates of
    magnitude >= delta (the others constrained to [-delta, delta])."""
    M = cfg["coefficient_bound"]
    lo, hi = cfg["omega_spectrum"]
    lstar = gs.saturated_value(Sigma)
    best = (np.inf, None)
    for od in FWD:
        fwd = FWD[od]
        for T in itertools.combinations(range(6), k):
            def unpack(z):
                B = np.zeros((P, P))
                for v, e in zip(z[:3], fwd):
                    B[e] = v
                O = np.diag(z[3:6]).astype(float)
                for v, pr in zip(z[6:], PAIRS):
                    O[pr] = O[pr[::-1]] = v
                return B, O

            def fun(z):
                B, O = unpack(z)
                if np.linalg.eigvalsh(O)[0] <= 1e-8:
                    return 1e10, np.zeros_like(z)
                v, gb, go = loss_grad(B, O, Sigma)
                return v, np.r_[[gb[e] for e in fwd], np.diag(go), [2 * go[pr] for pr in PAIRS]]

            def cons(z):
                ev = np.linalg.eigvalsh(unpack(z)[1])
                return np.r_[ev - lo, hi - ev]
            bounds = []
            for c in range(6):
                lim = (M if c < 3 else hi) if c in T else delta
                bounds.append((-lim, lim))
            bounds = bounds[:3] + [(lo, hi)] * 3 + bounds[3:]
            for s in range(starts):
                x = rng.uniform(-1, 1, 3) * np.array([b[1] for b in bounds[:3]]) * (s > 0)
                Bx = np.zeros((P, P))
                for v, e in zip(x, fwd):
                    Bx[e] = v
                A = np.eye(P) - Bx
                R = A.T @ Sigma @ A
                dg = np.clip(np.diag(R), lo + 1e-3, hi - 1e-3)
                off = np.array([np.clip(R[pr], *bounds[6 + q]) for q, pr in enumerate(PAIRS)])
                for sh in (1, .5, .25, 0):
                    z0 = np.r_[x, dg, sh * off]
                    if np.min(cons(z0)) > 0:
                        break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = minimize(fun, z0, jac=True, method="SLSQP", bounds=bounds,
                                 constraints=[dict(type="ineq", fun=cons)],
                                 options=dict(ftol=1e-15, maxiter=1000))
                if np.min(cons(r.x)) < -1e-9:
                    continue
                v = gs.loss(*unpack(r.x), Sigma) - lstar
                if v < best[0]:
                    names = [f"B{fwd[c][0]+1}{fwd[c][1]+1}" if c < 3 else f"O{PAIRS[c-3][0]+1}{PAIRS[c-3][1]+1}"
                             for c in T]
                    best = (float(v), dict(order="".join(str(q + 1) for q in od), free=names,
                                           z=r.x.tolist()))
    return best


def eta_pen(Sigma, k, delta, lam, cfg, rng, starts=2):
    """Penalty-aware version of eta_k: inf over theta in K with at most k edge coordinates
    of magnitude >= delta of  l(theta) - l_* + lam * sum_{small j} q_delta(x_j).
    Since P_delta(theta) >= k' + sum_small q over the k' <= k saturated coordinates, the
    population minimizer set equals E_min whenever eta_pen(k) > lam (e - k) for every
    k < e (points with >= e saturated coordinates score >= l_* + lam e, with equality
    exactly on E_min).  Small coordinates carry a fixed sign so that q is smooth."""
    M = cfg["coefficient_bound"]
    lo, hi = cfg["omega_spectrum"]
    lstar = gs.saturated_value(Sigma)
    best = (np.inf, None)
    zidx = [0, 1, 2, 6, 7, 8]          # edge coordinate c -> position in z
    for od in FWD:
        fwd = FWD[od]

        def unpack(z):
            B = np.zeros((P, P))
            for v, e in zip(z[:3], fwd):
                B[e] = v
            O = np.diag(z[3:6]).astype(float)
            for v, pr in zip(z[6:], PAIRS):
                O[pr] = O[pr[::-1]] = v
            return B, O

        def cons(z):
            ev = np.linalg.eigvalsh(unpack(z)[1])
            return np.r_[ev - lo, hi - ev]
        for T in itertools.combinations(range(6), k):
            small = [c for c in range(6) if c not in T]
            for signs in itertools.product([-1.0, 1.0], repeat=len(small)):
                sp = [zidx[c] for c in small]
                sg = np.array(signs)

                def fun(z):
                    B, O = unpack(z)
                    if np.linalg.eigvalsh(O)[0] <= 1e-8:
                        return 1e10, np.zeros_like(z)
                    v, gb, go = loss_grad(B, O, Sigma)
                    g = np.r_[[gb[e] for e in fwd], np.diag(go), [2 * go[pr] for pr in PAIRS]]
                    x = z[sp]
                    v += lam * float(np.sum(2 * sg * x / delta - x * x / delta ** 2))
                    g[sp] += lam * (2 * sg / delta - 2 * x / delta ** 2)
                    return v, g
                bounds = [(-M, M)] * 3 + [(lo, hi)] * 3 + [(-hi, hi)] * 3
                for c, s_ in zip(small, signs):
                    bounds[zidx[c]] = (0.0, delta) if s_ > 0 else (-delta, 0.0)
                for s in range(starts):
                    x = np.array([rng.uniform(*bounds[q]) if s > 0 else float(np.clip(0.0, *bounds[q]))
                                  for q in range(3)])
                    Bx = np.zeros((P, P))
                    for v, e in zip(x, fwd):
                        Bx[e] = v
                    A = np.eye(P) - Bx
                    R = A.T @ Sigma @ A
                    dg = np.clip(np.diag(R), lo + 1e-3, hi - 1e-3)
                    off = np.array([np.clip(R[pr], *bounds[6 + q]) for q, pr in enumerate(PAIRS)])
                    for sh in (1, .5, .25, 0):
                        z0 = np.r_[x, dg, np.clip(sh * off, [b[0] for b in bounds[6:]], [b[1] for b in bounds[6:]])]
                        if np.min(cons(z0)) > 0:
                            break
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = minimize(fun, z0, jac=True, method="SLSQP", bounds=bounds,
                                     constraints=[dict(type="ineq", fun=cons)],
                                     options=dict(ftol=1e-15, maxiter=1000))
                    if np.min(cons(r.x)) < -1e-9:
                        continue
                    xs = r.x[sp]
                    v = gs.loss(*unpack(r.x), Sigma) - lstar + lam * float(np.sum(penalty(xs, delta)))
                    if v < best[0]:
                        names = [f"B{fwd[c][0]+1}{fwd[c][1]+1}" if c < 3 else f"O{PAIRS[c-3][0]+1}{PAIRS[c-3][1]+1}"
                                 for c in T]
                        best = (float(v), dict(order="".join(str(q + 1) for q in od), free=names,
                                               z=r.x.tolist()))
    return best


def admissibility(cfg):
    out = {}
    pop = json.loads((POPULATION).read_text())["designs"]
    for di, d in enumerate(cfg["designs"]):
        B, O = gs.design_parameters(d)
        Sig = gs.covariance(B, O)
        e = pop[d]["e_star"]
        rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, 0, 0, 7]))
        d0, d0_arg = delta0(Sig, e, cfg, rng)
        etas = {}
        for k in range(e):
            v, arg = eta_k(Sig, k, cfg["delta"], cfg, rng)
            etas[k] = dict(eta=v, per_missing_edge=v / (e - k), argmin=arg)
        lam0 = min(x["per_missing_edge"] for x in etas.values())
        sups = {s["support"]: s for s in json.loads((POPULATION).read_text())["supports"]}
        l0_gaps = [(g / (e - sups[s]["count"]), s) for s, g in pop[d]["gaps"].items()
                   if g > 1e-10 and sups[s]["count"] < e]
        emin_min = min(x["min_abs_coordinate"] for x in pop[d]["E_min"])
        scan = {}
        for dl in cfg["delta_scan"]:
            et = [eta_k(Sig, k, dl, cfg, rng)[0] for k in range(e)]
            scan[str(dl)] = dict(eta=et, lambda0=min(x / (e - k) for k, x in enumerate(et)))
        tight = {}
        for k in range(e):
            v, arg = eta_pen(Sig, k, cfg["delta"], cfg["lam"], cfg, rng)
            tight[k] = dict(value=v, required=cfg["lam"] * (e - k), margin=v - cfg["lam"] * (e - k), argmin=arg)
        out[d] = dict(e_star=e, lambda0_scan=scan, penalty_aware_check=tight,
                      penalty_aware_ok=bool(all(x["margin"] > 0 for x in tight.values())), delta0_fiber_eth_largest=d0, delta0_argmin=d0_arg,
                      smallest_active_magnitude_E_min=emin_min,
                      eta_k_at_delta=etas, lambda0_at_delta=lam0,
                      l0_per_missing_edge_gap=min(l0_gaps)[0], l0_per_missing_edge_gap_support=min(l0_gaps)[1],
                      delta_ok=bool(cfg["delta"] < min(d0, emin_min)),
                      lambda_ok=bool(cfg["lam"] < lam0))
    return out


# ----------------------------------------------------------------------------
# Lipschitz constants
# ----------------------------------------------------------------------------

def hess_omega_norm(B, O, S):
    """Operator norm (Frobenius) of the Omega-Hessian of l_S at (B, O), exact formula."""
    inv = np.linalg.inv(O)
    A = np.eye(P) - B
    R = A.T @ S @ A
    basis = []
    for i in range(P):
        E = np.zeros((P, P)); E[i, i] = 1
        basis.append(E)
    for i, j in PAIRS:
        E = np.zeros((P, P)); E[i, j] = E[j, i] = 1 / math.sqrt(2)
        basis.append(E)
    H = np.zeros((6, 6))
    for a, Ea in enumerate(basis):
        for c, Ec in enumerate(basis):
            H[a, c] = (-0.5 * np.trace(inv @ Ea @ inv @ Ec)
                       + 0.5 * np.trace(inv @ Ea @ inv @ Ec @ inv @ R)
                       + 0.5 * np.trace(inv @ Ec @ inv @ Ea @ inv @ R))
    return float(np.max(np.abs(np.linalg.eigvalsh((H + H.T) / 2))))


def all_covariances(cfg):
    out = {}
    for di, d in enumerate(cfg["designs"]):
        B, O = gs.design_parameters(d)
        Sig = gs.covariance(B, O)
        for n in cfg["ns"]:
            for rep in range(cfg["datasets"]):
                rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep]))
                out[(d, n, rep)] = gs.sample_covariance(Sig, n, rng)
    return out


def lipschitz_K(covs, cfg):
    """K-wide constants.  B-block: the B-Hessian of l_S is H -> S H Omega^{-1}, so
    L_B = lambda_max(S) / lambda_min(Omega), maximized on K at lambda_min = m.
    Omega-block: the bound 1/(2 m^2) + lambda_max(R)/m^3 with R = (I-B)^T S (I-B)."""
    m = cfg["omega_spectrum"][0]
    M = cfg["coefficient_bound"]
    lmax_S = max(np.linalg.eigvalsh(S)[-1] for S in covs.values())
    # largest ||I - B||_2 over acyclic B with |B_ij| <= M: attained at a vertex of the box
    # for some order; the order only permutes, so use the upper-triangular pattern.
    best = 0.0
    for signs in itertools.product([-1, 1], repeat=3):
        Bm = np.zeros((P, P))
        Bm[0, 1], Bm[0, 2], Bm[1, 2] = M * np.array(signs)
        best = max(best, np.linalg.norm(np.eye(P) - Bm, 2))
    lmax_R = lmax_S * best ** 2
    return dict(lambda_max_S_over_datasets=float(lmax_S), L_B_K=float(lmax_S / m),
                max_norm_I_minus_B=float(best), L_Omega_K_bound=float(1 / (2 * m * m) + lmax_R / m ** 3))


def calibrate(covs, cfg, n_pilot=4):
    """Estimate L_Omega on the region the algorithm visits: Hessian norms at every
    start used by the study on pilot datasets (deterministic, 20 random, refits of the
    50 best supports) and along pilot trajectories from the first five starts."""
    rows = []
    pilot_cfg = dict(cfg)
    pilot_cfg["beta"] = 0.02
    pilot_cfg["max_cycles"] = 20000
    Lmax = 0.0
    where = None
    for di, d in enumerate(cfg["designs"]):
        for n in cfg["ns"]:
            for rep in range(n_pilot):
                S = covs[(d, n, rep)]
                rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep, 1]))
                fits, _ = gs.lmin_all(S, rng)
                crit = np.array([criterion(f.B, f.O, S, cfg["lam"], cfg["delta"]) for f in fits])
                for i in np.argsort(crit, kind="stable")[:cfg["refit_starts"]]:
                    f = fits[i]
                    h = hess_omega_norm(f.B, f.O, S)
                    if h > Lmax:
                        Lmax, where = h, f"refit {d} n={n} rep={rep}"
                for k in range(cfg["random_starts"] + 1):
                    if k == 0:
                        B0, O0 = np.zeros((P, P)), np.diag(np.diag(S))
                    else:
                        srng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep, 2, k]))
                        B0, O0, _ = random_start(S, srng, cfg)
                    h = hess_omega_norm(B0, O0, S)
                    if h > Lmax:
                        Lmax, where = h, f"start {d} n={n} rep={rep} k={k}"
                    if k <= 4:
                        # pilot trajectory: record the Omega-Hessian at iterates
                        b, o, info, traj = run_traced(S, B0, O0, pilot_cfg)
                        hm = max(hess_omega_norm(bb, oo, S) for bb, oo in traj)
                        rows.append(dict(design=d, n=n, rep=rep, start=k, hess_max=hm, cycles=info["cycles"],
                                         status=info["status"], min_eig=info["min_omega_eig"]))
                        if hm > Lmax:
                            Lmax, where = hm, f"trajectory {d} n={n} rep={rep} k={k}"
    return dict(L_Omega_estimate=Lmax, attained_at=where, pilot_datasets_per_cell=n_pilot,
                pilot_beta=pilot_cfg["beta"], pilot_runs=rows)


def run_traced(S, B0, O0, cfg, every=50):
    """Same cycle as run_alternating, returning iterates every `every` cycles."""
    c2 = dict(cfg)
    traj = []
    b, o = B0.copy(), O0.copy()
    total = 0
    info = None
    while total < cfg["max_cycles"]:
        c2["max_cycles"] = every
        traj.append((b.copy(), o.copy()))
        b, o, info = run_alternating(S, b, o, c2)
        total += info["cycles"]
        if info["status"] == "step_tolerance":
            break
    traj.append((b, o))
    info["cycles"] = total
    return b, o, info, traj


# ----------------------------------------------------------------------------
# One dataset
# ----------------------------------------------------------------------------

def run_dataset(job):
    d, di, n, rep, S, pop_d, cfg = job
    t0 = time.monotonic()
    emin = set(pop_d["E_min_indices"])
    e_star = pop_d["e_star"]
    lam, delta = cfg["lam"], cfg["delta"]
    prox = NoiseProx(P, cfg["beta"], lam, delta, *cfg["omega_spectrum"])
    rows = []
    base = dict(design=d, n=n, dataset=rep)

    def record(kind, k, B0, O0, extra):
        b, o, info = run_alternating(S, B0, O0, cfg, prox)
        r = dict(base, start_type=kind, start=k, **extra, **info, **describe(b, o, cfg, emin, e_star))
        rows.append(r)

    # deterministic start
    record("deterministic", 0, np.zeros((P, P)), np.diag(np.diag(S)), {})
    # random starts
    for k in range(1, cfg["random_starts"] + 1):
        srng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep, 2, k]))
        B0, O0, ex = random_start(S, srng, cfg)
        record("random", k, B0, O0, ex)
    # refit starts: constrained fits of all 200 supports, then the 50 best by the
    # fixed-tuning criterion at the refit
    rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep, 1]))
    fits, repairs = gs.lmin_all(S, rng)
    lmin = np.array([f.value for f in fits])
    crit = np.array([criterion(f.B, f.O, S, lam, delta) for f in fits])
    counts = np.array([s.count for s in gs.SUPPORTS])
    l0 = lmin + lam * counts
    top = np.argsort(crit, kind="stable")[:cfg["refit_starts"]]
    for rank, i in enumerate(top):
        record("refit", rank + 1, fits[i].B, fits[i].O,
               dict(start_support=gs.SUPPORTS[i].name, start_criterion=float(crit[i])))
    objs = np.array([r["objective"] for r in rows])
    ref = float(objs.min())
    tol = cfg["reference_tolerance"]
    for r in rows:
        r["gap_to_reference"] = r["objective"] - ref
        r["attains_reference"] = bool(r["objective"] <= ref + tol)
    at_ref = [r for r in rows if r["attains_reference"]]
    l0_best = float(l0.min())
    l0_sel = [int(i) for i in np.where(l0 <= l0_best + 1e-12)[0]]
    ds = dict(base, reference_objective=ref,
              reference_supports=";".join(sorted({r["support_thr"] for r in at_ref})),
              reference_all_emin=all(r["is_emin"] for r in at_ref),
              reference_attained_by_random=any(r["start_type"] == "random" for r in at_ref),
              reference_attained_by_deterministic=any(r["start_type"] == "deterministic" for r in at_ref),
              reference_attained_by_refit=any(r["start_type"] == "refit" for r in at_ref),
              l0_fixed_tuning_min=l0_best,
              l0_fixed_tuning_selected=";".join(gs.SUPPORTS[i].name for i in l0_sel),
              l0_fixed_tuning_selected_all_emin=all(i in emin for i in l0_sel),
              best_refit_criterion=float(crit.min()),
              best_refit_support=gs.SUPPORTS[int(np.argmin(crit))].name,
              reference_minus_l0_min=ref - l0_best,
              reference_minus_best_refit_criterion=ref - float(crit.min()),
              hierarchy_repairs=len(repairs),
              sample_min_eig=float(np.linalg.eigvalsh(S)[0]),
              sample_max_eig=float(np.linalg.eigvalsh(S)[-1]),
              noise_prox_statuses=json.dumps(prox.statuses),
              runtime_seconds=time.monotonic() - t0)
    return rows, ds


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

def summarize(rows, dsets, cfg):
    wil = gs.wilson
    out = []
    for d in cfg["designs"]:
        for n in cfg["ns"]:
            R = [r for r in rows if r["design"] == d and r["n"] == n]
            rnd = [r for r in R if r["start_type"] == "random"]
            det = [r for r in R if r["start_type"] == "deterministic"]
            ref = [r for r in R if r["start_type"] == "refit"]
            D = sorted({r["dataset"] for r in R})
            m = len(D)
            row = dict(design=d, n=n, datasets=m, random_runs=len(rnd))
            k = sum(r["is_emin"] for r in rnd)
            lo, hi = wil(k, len(rnd))
            row.update(single_start_emin=k / len(rnd), single_start_emin_lo=lo, single_start_emin_hi=hi)
            k = sum(r["attains_reference"] for r in rnd)
            row.update(single_start_reference=k / len(rnd))
            by = {}
            for r in rnd:
                by.setdefault(r["dataset"], []).append(r)
            for K in cfg["Ks"]:
                win_e, win_r = 0, 0
                for ds in D:
                    first = sorted(by[ds], key=lambda r: r["start"])[:K]
                    best = min(first, key=lambda r: (r["objective"], r["start"]))
                    win_e += best["is_emin"]
                    win_r += best["attains_reference"]
                lo, hi = wil(win_e, m)
                row[f"bestof{K}_emin"] = win_e / m
                row[f"bestof{K}_emin_lo"] = lo
                row[f"bestof{K}_emin_hi"] = hi
                row[f"bestof{K}_reference"] = win_r / m
            row.update(
                single_bowed=sum(r["bowed"] for r in rnd) / len(rnd),
                single_more_than_e_star=sum(r["more_than_e_star"] for r in rnd) / len(rnd),
                single_bowed_or_more=sum(r["bowed"] or r["more_than_e_star"] for r in rnd) / len(rnd),
                single_fewer_than_e_star=sum(r["fewer_than_e_star"] for r in rnd) / len(rnd),
                single_e_star_not_emin=sum((r["support_count"] == cfg_e(d)) and not r["is_emin"] for r in rnd) / len(rnd),
                deterministic_emin=sum(r["is_emin"] for r in det) / len(det),
                deterministic_reference=sum(r["attains_reference"] for r in det) / len(det),
                refit_start_emin=sum(r["is_emin"] for r in ref) / max(len(ref), 1),
                reference_all_emin=sum(x["reference_all_emin"] for x in dsets if x["design"] == d and x["n"] == n) / m,
                reference_attained_by_random=sum(x["reference_attained_by_random"] for x in dsets
                                                 if x["design"] == d and x["n"] == n) / m,
                l0_fixed_tuning_emin=sum(x["l0_fixed_tuning_selected_all_emin"] for x in dsets
                                         if x["design"] == d and x["n"] == n) / m,
                median_cycles_random=float(np.median([r["cycles"] for r in rnd])),
                max_cycles_random=int(max(r["cycles"] for r in rnd)),
                cycle_cap_hits_all_runs=int(sum(r["status"] == "cycle_cap" for r in R)),
                max_objective_increase_all_runs=float(max(r["max_objective_increase"] for r in R)),
                max_curvature_ratio_B=float(max(r["max_curvature_ratio_B"] for r in R)),
                max_curvature_ratio_Omega=float(max(r["max_curvature_ratio_Omega"] for r in R)),
                noise_conic_calls=int(sum(r["noise_conic_calls"] for r in R)),
                median_runtime_random_s=float(np.median([r["runtime_seconds"] for r in rnd])),
            )
            row["runs_with_coords_in_(0_delta)"] = int(sum(r["n_coords_in_(0,delta)"] > 0 for r in R))
            # delta/2 threshold, bowed end points that are not E_min, unconverged runs,
            # and the number of independent starts needed for a 95% chance of success
            emin_names = set(cfg_emin(d))
            p1 = row["single_start_emin"]
            row.update(
                single_start_emin_half_threshold=sum(r["support_half"] in emin_names for r in rnd) / len(rnd),
                single_bowed_not_emin=sum(r["bowed"] and not r["is_emin"] for r in rnd) / len(rnd),
                single_not_emin_bowed_or_more=sum((r["bowed"] or r["more_than_e_star"]) and not r["is_emin"]
                                                  for r in rnd) / len(rnd),
                random_cycle_cap_fraction=sum(r["status"] == "cycle_cap" for r in rnd) / len(rnd),
                random_emin_among_cycle_cap=(sum(r["is_emin"] for r in rnd if r["status"] == "cycle_cap") /
                                             max(1, sum(r["status"] == "cycle_cap" for r in rnd))),
                starts_for_95pct_if_independent=(math.ceil(math.log(0.05) / math.log(1 - p1)) if 0 < p1 < 1 else None),
                deterministic_support=";".join(sorted({r["support_thr"] for r in det})),
            )
            out.append(row)
    return out


_E = {}
_EM = {}


def cfg_emin(d):
    if not _EM:
        pop = json.loads((POPULATION).read_text())["designs"]
        _EM.update({k: v["E_min_names"] for k, v in pop.items()})
    return _EM[d]


def _open(path, mode="rt"):
    return (gzip.open if str(path).endswith(".gz") else open)(path, mode, newline="")


def _load(path):
    def conv(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v
    with _open(path) as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def resummarize(tag="", out=RESULTS):
    cfg = json.loads((out / f"config{tag}.json").read_text())
    rows, dsets = _load(out / f"runs{tag}.csv.gz"), _load(out / f"datasets{tag}.csv")
    summ = summarize(rows, dsets, cfg)
    if tag:
        for row in summ:
            for key in list(row):
                if key.startswith(("refit_start", "reference_all", "reference_attained")):
                    row.pop(key)
    write_csv(out / f"summary{tag}.csv", summ)
    print("summary rewritten", tag, len(rows), len(dsets))


def cfg_e(d):
    if not _E:
        pop = json.loads((POPULATION).read_text())["designs"]
        _E.update({k: v["e_star"] for k, v in pop.items()})
    return _E[d]


def _jd(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with _open(path, "wt") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def small_starts(cfg, pop, workers, out=RESULTS):
    """Random starts ten times smaller (B sd 0.05, Omega off-diagonal sd 0.05 on the
    correlation scale), the first 30 datasets per cell, no refit starts.  Steps and
    tunings are read from config.json of the main study; the reference minimum of each
    dataset is the smaller of the main-study reference and the best run here."""
    import multiprocessing as mp
    t0 = time.monotonic()
    main_cfg = json.loads((RESULTS / "config.json").read_text())
    cfg = dict(cfg, alpha=main_cfg["alpha"], beta=main_cfg["beta"], start_B_sd=0.05,
               start_omega_offdiag_sd=0.05, refit_starts=0, datasets=30,
               variant="small random starts")
    ref_main = {}
    with open(RESULTS / "datasets.csv") as f:
        for r in csv.DictReader(f):
            ref_main[(r["design"], int(r["n"]), int(r["dataset"]))] = float(r["reference_objective"])
    covs = all_covariances(cfg)
    jobs = [(d, di, n, rep, covs[(d, n, rep)],
              dict(E_min_indices=pop[d]["E_min_indices"], e_star=pop[d]["e_star"]), cfg)
             for di, d in enumerate(cfg["designs"]) for n in cfg["ns"] for rep in range(cfg["datasets"])]
    rows, dsets = [], []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, (r, ds) in enumerate(pool.imap_unordered(run_dataset, jobs, chunksize=1)):
            key = (ds["design"], ds["n"], ds["dataset"])
            ref = min(ref_main[key], ds["reference_objective"])
            for x in r:
                x["gap_to_reference"] = x["objective"] - ref
                x["attains_reference"] = bool(x["objective"] <= ref + cfg["reference_tolerance"])
            ds["reference_objective_main"] = ref_main[key]
            ds["variant_below_main_reference"] = bool(ds["reference_objective"] < ref_main[key] - cfg["reference_tolerance"])
            rows.extend(r)
            dsets.append(ds)
            if k % 40 == 0:
                print(f"small {k+1}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["design"], r["n"], r["dataset"], r["start_type"], r["start"]))
    dsets.sort(key=lambda r: (r["design"], r["n"], r["dataset"]))
    write_csv(out / "runs_small_starts.csv.gz", rows)
    write_csv(out / "datasets_small_starts.csv", dsets)
    summ = summarize(rows, dsets, cfg)
    for row in summ:
        for key in list(row):
            if key.startswith("refit_start") or key.startswith("reference_all") or key.startswith("reference_attained"):
                row.pop(key)
    write_csv(out / "summary_small_starts.csv", summ)
    (out / "config_small_starts.json").write_text(json.dumps(
        dict(cfg, runtime_seconds=time.monotonic() - t0), indent=2, default=_jd) + "\n")
    print("small done", len(rows), f"{time.monotonic()-t0:.0f}s", flush=True)


def floor2(x):
    e = math.floor(math.log10(x)) - 1
    return round(math.floor(x / 10 ** e) * 10 ** e, 12)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--datasets", type=int, default=None, help="samples per cell (default 100)")
    ap.add_argument("--small-starts", action="store_true", help="starts ten times smaller")
    ap.add_argument("--resummarize", action="store_true", help="rebuild summaries from the run tables")
    ap.add_argument("--calibrate-only", action="store_true", help="stop after the tuning checks and step calibration")
    ap.add_argument("--reuse-steps", action="store_true",
                    help="skip the tuning checks and calibration; take alpha and beta from results/config.json")
    ap.add_argument("--out", default=str(RESULTS), help="output directory (default: results/)")
    args = ap.parse_args()
    out = Path(args.out)
    t0 = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG)
    if args.datasets is not None:
        cfg["datasets"] = args.datasets
    pop = json.loads((POPULATION).read_text())["designs"]
    if args.resummarize:
        resummarize(out=out)
        return resummarize("_small_starts", out=out)
    if args.small_starts:
        return small_starts(cfg, pop, args.workers, out=out)
    covs = all_covariances(cfg)
    t_adm = t_cal = 0.0
    if args.reuse_steps:
        stored = json.loads((RESULTS / "config.json").read_text())
        cfg["alpha"], cfg["beta"] = stored["alpha"], stored["beta"]
        print("alpha", cfg["alpha"], "beta", cfg["beta"], "(stored)", flush=True)
    else:
        # tuning admissibility
        t_adm = time.monotonic()
        adm = admissibility(cfg)
        t_adm = time.monotonic() - t_adm
        print(json.dumps({d: dict(delta0=a["delta0_fiber_eth_largest"], lambda0=a["lambda0_at_delta"],
                                  delta_ok=a["delta_ok"], lambda_ok=a["lambda_ok"])
                          for d, a in adm.items()}, default=_jd), flush=True)
        # Lipschitz constants and steps
        LK = lipschitz_K(covs, cfg)
        cap = cfg["delta"] ** 2 / (2 * cfg["lam"])
        cfg["alpha"] = floor2(min(0.9 / LK["L_B_K"], 0.99 * cap))
        t_cal = time.monotonic()
        cal = calibrate(covs, cfg)
        t_cal = time.monotonic() - t_cal
        alpha = cfg["alpha"]
        beta = floor2(min(0.5 / cal["L_Omega_estimate"], 0.99 * cap))
        cfg["alpha"], cfg["beta"] = alpha, beta
        steps = dict(L_K=LK, L_Omega_calibration={k: v for k, v in cal.items() if k != "pilot_runs"},
                     alpha=alpha, beta=beta, prox_cap_delta2_over_2lambda=cap,
                     rule="alpha = 0.9 / L_B(K) (exact K-wide B-block constant), beta = 0.5 / (estimated L_Omega "
                          "on the visited region), each rounded down to two significant digits and "
                          "capped at 0.99 delta^2/(2 lambda)")
        write_csv(out / "calibration_pilot.csv", cal["pilot_runs"])
        cfg_out = dict(cfg, admissibility=adm, steps=steps)
        (out / "config.json").write_text(json.dumps(cfg_out, indent=2, default=_jd) + "\n")
        print("alpha", alpha, "beta", beta, "L_B_K", LK["L_B_K"], "L_Omega_est", cal["L_Omega_estimate"],
              cal["attained_at"], f"admissibility {t_adm:.0f}s calibration {t_cal:.0f}s", flush=True)
        if args.calibrate_only:
            return
    jobs = []
    for di, d in enumerate(cfg["designs"]):
        for n in cfg["ns"]:
            for rep in range(cfg["datasets"]):
                jobs.append((d, di, n, rep, covs[(d, n, rep)],
                              dict(E_min_indices=pop[d]["E_min_indices"], e_star=pop[d]["e_star"]), cfg))
    # interleave cells so that a partial run covers every cell
    jobs.sort(key=lambda t: (t[3], t[0], t[2]))
    rows, dsets = [], []
    t_sim = time.monotonic()
    part = out / "runs.partial.csv"
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for k, (r, ds) in enumerate(pool.imap_unordered(run_dataset, jobs, chunksize=1)):
            rows.extend(r)
            dsets.append(ds)
            if k % 40 == 0 or k == len(jobs) - 1:
                print(f"{k+1}/{len(jobs)} {time.monotonic()-t_sim:.0f}s", flush=True)
            if k % 150 == 149:
                write_csv(part, rows)
    t_sim = time.monotonic() - t_sim
    rows.sort(key=lambda r: (r["design"], r["n"], r["dataset"], ["deterministic", "random", "refit"].index(r["start_type"]), r["start"]))
    dsets.sort(key=lambda r: (r["design"], r["n"], r["dataset"]))
    write_csv(out / "runs.csv.gz", rows)
    write_csv(out / "datasets.csv", dsets)
    write_csv(out / "summary.csv", summarize(rows, dsets, cfg))
    if part.exists():
        part.unlink()
    meta = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                cpu_count=os.cpu_count(), workers=args.workers,
                runtime_seconds=dict(total=time.monotonic() - t0, admissibility=t_adm, calibration=t_cal,
                                     simulation=t_sim,
                                     dataset_cpu_sum=float(sum(x["runtime_seconds"] for x in dsets))),
                n_runs=len(rows), n_datasets=len(dsets))
    try:
        import cvxpy
        meta["versions"]["cvxpy"] = cvxpy.__version__
    except Exception:
        pass
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=_jd) + "\n")
    print("done", len(rows), f"{time.monotonic()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

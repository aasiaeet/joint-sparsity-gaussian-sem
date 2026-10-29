"""Global selection on three vertices by exhaustive support enumeration.

Section 7.1 of the main article and Section S18 of the Supplementary Material.
For every acyclic mixed support H on three vertices and every sample, compute
the constrained likelihood minimum l_min(H; S_n) over the domain K and select
the support minimizing l_min(H; S_n) + lambda_n |H| with
lambda_n = delta_n = log(n)/n (Theorems 3, 5 and 9).

Conventions.  Parent-row parametrization X = B^T X + eps,
Sigma = Phi(B, Omega) = (I-B)^{-T} Omega (I-B)^{-1}; B[i, j] != 0 is the
directed edge i -> j; Omega[i, j] != 0 (i < j) is the bidirected edge i <-> j.
Loss l_S(B, Omega) = 0.5 log det Omega + 0.5 tr(Omega^{-1} (I-B)^T S (I-B)).
Edge count c = #directed + #unordered bidirected.  Quasi-MCP
q_delta(t) = 2|t|/delta - t^2/delta^2 on |t| <= delta and 1 beyond.

Constrained likelihood.  For a support H = (D, U) and fixed B on D, the
minimum over Omega with zero pattern outside U has a closed form on three
vertices: |U| = 0, 1, 3 give block-diagonal Omega = blocks of
R = (I-B)^T S (I-B); |U| = 2 with missing pair {i, k} is the covariance
graph i <-> j <-> k, whose model equals that of the DAG i -> j <- k, so the
inner minimum is the regression fit of j on (i, k) in R.  The profile over
the at most three free entries of B is minimized by multistart L-BFGS-B
with the envelope gradient.  When the profiled Omega violates the spectral
box, a joint SLSQP solve with eigenvalue constraints is used instead.
When U is empty the exact regression MLE is used.

All computations are in floating point; none of them is a certificate
(see ../certify_global_selection for the interval-arithmetic certification).

Usage:
    python3 global_selection.py [--workers W] [--quick] [--out DIR]

--quick runs every 97th sample (44 of 4,200) as a fast check; use --out to
write its results outside results/.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # 3x3 algebra: threads only add overhead

import argparse
import gzip
import itertools
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
P = 3
PAIRS = [(0, 1), (0, 2), (1, 2)]
DIRECTED = [(i, j) for i in range(P) for j in range(P) if i != j]

CONFIG = dict(
    study="global selection by exhaustive enumeration, p = 3",
    p=P,
    n_supports_expected=200,
    coefficient_bound=2.0,
    omega_spectrum=[0.05, 10.0],
    designs={
        "G1": dict(label="chain", B={"1->2": 0.8, "2->3": -0.7}, Omega_offdiag={},
                   Omega_diag=[1, 1, 1], e_star=2),
        "G2": dict(label="confounded pair", B={"1->2": 0.7}, Omega_offdiag={"2<->3": 0.5},
                   Omega_diag=[1, 1, 1], e_star=2),
        "G3": dict(label="front-door", B={"1->2": 0.7, "2->3": 0.6},
                   Omega_offdiag={"1<->3": 0.4}, Omega_diag=[1, 1, 1], e_star=3),
    },
    ns=[100, 200, 400, 800, 1600, 3200, 6400],
    replications=200,
    tuning="lambda_n = delta_n = log(n)/n",
    threshold="delta_n / 2 (Theorem 9(ii))",
    master_seed=20260925,
    data_seed_rule="numpy default_rng(SeedSequence([master_seed, design_index, n, replication]))",
    solver_seed_rule="numpy default_rng(SeedSequence([master_seed, design_index, n, replication, 1]))",
    covariance="centered empirical covariance, divisor n",
    exact_fit_tolerance=1e-10,
    tie_tolerances=[1e-12, 1e-9],
    solver=dict(method="profile over B (closed-form Omega step) + L-BFGS-B",
                random_starts=8, start_box=[-2.0, 2.0], lbfgs_ftol=1e-15,
                lbfgs_gtol=1e-11, lbfgs_maxiter=2000,
                spectral_fallback="joint SLSQP with eigenvalue constraints",
                hierarchy_repair="re-solve H warm-started at the optimum of any H' in H that fits better by >1e-10"),
    class_invariant=dict(jacobian_draws=6, rank_rel_tol=1e-8, vanish_tol=1e-10,
                         constraints=["rho12", "rho13", "rho23", "rho12.3", "rho13.2", "rho23.1"],
                         class_seed=4242),
    validation=dict(random_S=24, validation_seed=777, monotone_tol=1e-9, dag_tol=1e-9,
                    saturated_tol=1e-9),
    equivalence_check=dict(top_supports=6, descent_max_iter=3000, descent_step_min=1e-14,
                           lower_tol=1e-10,
                           description="prox-gradient descent of l_S + lambda P_delta over all forward B "
                                       "entries of each topological order compatible with the start "
                                       "support plus all Omega entries; exact scalar prox; backtracking "
                                       "with sufficient decrease and spectral feasibility"),
)


# ----------------------------------------------------------------------------
# Supports
# ----------------------------------------------------------------------------

def _acyclic(D):
    adj = {i: [j for (a, j) in D if a == i] for i in range(P)}
    state = [0] * P

    def dfs(v):
        state[v] = 1
        for w in adj[v]:
            if state[w] == 1 or (state[w] == 0 and not dfs(w)):
                return False
        state[v] = 2
        return True
    return all(state[v] == 2 or dfs(v) for v in range(P))


def _orders(D):
    out = []
    for perm in itertools.permutations(range(P)):
        pos = {v: k for k, v in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in D):
            out.append(perm)
    return out


def edge_name(e, directed=True):
    i, j = e
    return f"{i+1}->{j+1}" if directed else f"{i+1}<->{j+1}"


@dataclass(frozen=True)
class Support:
    index: int
    D: tuple
    U: tuple
    orders: tuple = field(compare=False)

    @property
    def count(self):
        return len(self.D) + len(self.U)

    @property
    def name(self):
        parts = [edge_name(e) for e in self.D] + [edge_name(e, False) for e in self.U]
        return ",".join(parts) if parts else "empty"

    @property
    def bowed(self):
        und = {tuple(sorted(e)) for e in self.D}
        return any(u in und for u in self.U)

    def mask(self):
        mb = np.zeros((P, P), bool)
        for i, j in self.D:
            mb[i, j] = True
        mo = np.zeros((P, P), bool)
        for i, j in self.U:
            mo[i, j] = mo[j, i] = True
        return mb, mo


def enumerate_supports():
    Ds = []
    for m in range(1 << len(DIRECTED)):
        D = tuple(e for k, e in enumerate(DIRECTED) if m >> k & 1)
        if _acyclic(D):
            Ds.append(D)
    Us = [tuple(pr for k, pr in enumerate(PAIRS) if m >> k & 1) for m in range(1 << len(PAIRS))]
    raw = sorted(((D, U) for D in Ds for U in Us), key=lambda t: (len(t[0]) + len(t[1]), len(t[0]), t[0], t[1]))
    sups = [Support(k, D, U, tuple(_orders(D))) for k, (D, U) in enumerate(raw)]
    return sups, len(Ds), len(Us)


SUPPORTS, N_DIRECTED_ACYCLIC, N_BIDIRECTED = enumerate_supports()
NAME_TO_INDEX = {s.name: s.index for s in SUPPORTS}


def support_from_masks(mb, mo):
    D = tuple(e for e in DIRECTED if mb[e])
    U = tuple(pr for pr in PAIRS if mo[pr])
    for s in SUPPORTS:
        if s.D == D and s.U == U:
            return s
    raise ValueError("support not acyclic")


def nested_pairs():
    """All (small, big) index pairs with small strictly contained in big."""
    out = []
    for a in SUPPORTS:
        sa, ua = set(a.D), set(a.U)
        for b in SUPPORTS:
            if a.index != b.index and sa <= set(b.D) and ua <= set(b.U):
                out.append((a.index, b.index))
    return np.array(out, dtype=int)


NESTED = nested_pairs()


# ----------------------------------------------------------------------------
# Model maps, loss, penalty
# ----------------------------------------------------------------------------

def covariance(B, O):
    C = np.linalg.inv(np.eye(len(B)) - B)
    return C.T @ O @ C


def loss_grad(B, O, S):
    A = np.eye(len(B)) - B
    inv = np.linalg.inv(O)
    R = A.T @ S @ A
    val = 0.5 * (np.linalg.slogdet(O)[1] + np.trace(inv @ R))
    gb = -S @ A @ inv
    go = 0.5 * (inv - inv @ R @ inv)
    return float(val), gb, (go + go.T) / 2


def loss(B, O, S):
    A = np.eye(len(B)) - B
    R = A.T @ S @ A
    sign, ld = np.linalg.slogdet(O)
    if sign <= 0:
        return np.inf
    return float(0.5 * (ld + np.trace(np.linalg.solve(O, R))))


def qmcp(x, delta):
    a = np.minimum(np.abs(x) / delta, 1.0)
    return 2 * a - a * a


def criterion(B, O, S, lam, delta):
    iu = np.triu_indices(P, 1)
    return loss(B, O, S) + lam * (qmcp(B[~np.eye(P, dtype=bool)], delta).sum() + qmcp(O[iu], delta).sum())


def saturated_value(S):
    return 0.5 * np.linalg.slogdet(S)[1] + P / 2


# ----------------------------------------------------------------------------
# Constrained likelihood l_min(H; S)
# ----------------------------------------------------------------------------

def inner_omega(R, U):
    """Closed-form minimizer of l over Omega with zero pattern outside U (p = 3)."""
    k = len(U)
    if k == 0:
        return np.diag(np.diag(R))
    if k == 3:
        return (R + R.T) / 2
    O = np.diag(np.diag(R))
    if k == 1:
        i, j = U[0]
        O[i, j] = O[j, i] = R[i, j]
        return O
    (i, k_), = [pr for pr in PAIRS if pr not in U]
    j = 3 - i - k_
    idx = [i, k_]
    beta = np.linalg.solve(R[np.ix_(idx, idx)], R[idx, j])
    rss = R[j, j] - R[j, idx] @ beta
    O[i, j] = O[j, i] = beta[0] * R[i, i]
    O[k_, j] = O[j, k_] = beta[1] * R[k_, k_]
    O[j, j] = beta[0] ** 2 * R[i, i] + beta[1] ** 2 * R[k_, k_] + rss
    return O


def regression_mle(S, D):
    """Exact Gaussian DAG MLE (U empty): per-node OLS, residual variances."""
    B = np.zeros((P, P))
    om = np.zeros(P)
    for j in range(P):
        pa = [i for (i, jj) in D if jj == j]
        if pa:
            beta = np.linalg.solve(S[np.ix_(pa, pa)], S[pa, j])
            B[pa, j] = beta
            om[j] = S[j, j] - S[j, pa] @ beta
        else:
            om[j] = S[j, j]
    O = np.diag(om)
    return 0.5 * float(np.sum(np.log(om))) + P / 2, B, O


def _profile(x, S, D, U):
    B = np.zeros((P, P))
    for v, e in zip(x, D):
        B[e] = v
    A = np.eye(P) - B
    R = A.T @ S @ A
    O = inner_omega(R, U)
    sign, ld = np.linalg.slogdet(O)
    if sign <= 0:
        return 1e10, np.zeros(len(x))
    inv = np.linalg.inv(O)
    val = 0.5 * (ld + np.trace(inv @ R))
    gb = -S @ A @ inv
    return float(val), np.array([gb[e] for e in D])


@dataclass
class Fit:
    value: float
    B: np.ndarray
    O: np.ndarray
    method: str
    n_starts: int = 0
    n_at_best: int = 0
    fallback: bool = False
    b_bound_touch: bool = False
    b_bound_binding: bool = False
    spec_bound_touch: bool = False


def _spectral_ok(O, lo, hi, tol=0.0):
    ev = np.linalg.eigvalsh(O)
    return ev[0] >= lo - tol and ev[-1] <= hi + tol


def _joint_constrained(S, sup, starts_B, cfg):
    """Joint SLSQP over (B_D, diag Omega, Omega_U) with spectral constraints."""
    lo, hi = cfg["omega_spectrum"]
    M = cfg["coefficient_bound"]
    D, U = sup.D, sup.U
    nd, nu = len(D), len(U)

    def unpack(z):
        B = np.zeros((P, P))
        for v, e in zip(z[:nd], D):
            B[e] = v
        O = np.diag(z[nd:nd + P]).astype(float)
        for v, (i, j) in zip(z[nd + P:], U):
            O[i, j] = O[j, i] = v
        return B, O

    def fun(z):
        B, O = unpack(z)
        if np.linalg.eigvalsh(O)[0] <= 1e-8:
            return 1e10, np.zeros_like(z)
        val, gb, go = loss_grad(B, O, S)
        g = np.r_[[gb[e] for e in D], np.diag(go), [2 * go[i, j] for i, j in U]]
        return val, g

    def cons(z):
        ev = np.linalg.eigvalsh(unpack(z)[1])
        return np.r_[ev - lo, hi - ev]

    bounds = [(-M, M)] * nd + [(lo, hi)] * P + [(-hi, hi)] * nu
    best = None
    for Bs in starts_B:
        A = np.eye(P) - Bs
        R = A.T @ S @ A
        Oin = inner_omega(R, U)
        d = np.clip(np.diag(Oin), lo + 1e-3, hi - 1e-3)
        off = np.array([Oin[i, j] for i, j in U])
        # shrink the off-diagonal part until the start is feasible
        for s in [1.0, .8, .6, .4, .2, .1, 0.0]:
            z0 = np.r_[[np.clip(Bs[e], -M, M) for e in D], d, s * off]
            if np.min(cons(z0)) > 0:
                break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(fun, z0, jac=True, method="SLSQP", bounds=bounds,
                         constraints=[dict(type="ineq", fun=cons)],
                         options=dict(ftol=1e-15, maxiter=2000))
        if np.min(cons(r.x)) < -1e-9:
            continue
        val = loss(*unpack(r.x), S)
        if best is None or val < best[0]:
            best = (val, *unpack(r.x))
    return best


def solve_support(S, sup, rng, cfg=CONFIG, force_numeric=False, extra_starts=()):
    lo, hi = cfg["omega_spectrum"]
    M = cfg["coefficient_bound"]
    scfg = cfg["solver"]
    D, U = sup.D, sup.U
    if not D:
        O = inner_omega(S, U)
        fit = Fit(loss(np.zeros((P, P)), O, S), np.zeros((P, P)), O, "closed_form_noB")
    elif not U and not force_numeric:
        val, B, O = regression_mle(S, D)
        fit = Fit(val, B, O, "regression_mle")
        if np.max(np.abs(B)) > M:
            fit = None  # coefficient bound would bind; use numeric path
    else:
        fit = None
    if fit is None:
        _, B_ols, _ = regression_mle(S, D)
        starts = [np.zeros(len(D)), np.clip([B_ols[e] for e in D], -M, M)]
        starts += [rng.uniform(*scfg["start_box"], size=len(D)) for _ in range(scfg["random_starts"])]
        starts += [np.array([Bx[e] for e in D]) for Bx in extra_starts]
        results = []
        for x0 in starts:
            r = minimize(_profile, x0, args=(S, D, U), jac=True, method="L-BFGS-B",
                         bounds=[(-M, M)] * len(D),
                         options=dict(ftol=scfg["lbfgs_ftol"], gtol=scfg["lbfgs_gtol"],
                                      maxiter=scfg["lbfgs_maxiter"], maxcor=20))
            results.append((float(r.fun), r.x))
        vals = np.array([v for v, _ in results])
        gbest = float(vals.min())

        def build(x):
            Bx = np.zeros((P, P))
            for v, e in zip(x, D):
                Bx[e] = v
            A = np.eye(P) - Bx
            return Bx, inner_omega(A.T @ S @ A, U)

        # Bowed supports have flat directions: several starts reach the same
        # value at different points.  Prefer the best start whose profiled
        # Omega lies in the spectral box.
        feas = []
        for v, x in results:
            Bx, Ox = build(x)
            if _spectral_ok(Ox, lo, hi):
                feas.append((loss(Bx, Ox, S), x, Bx, Ox))
        feas.sort(key=lambda t: t[0])
        n_at_best = int(np.sum(vals <= gbest + 1e-9))
        if feas and feas[0][0] <= gbest + 1e-11:
            v, x, B, O = feas[0]
            fit = Fit(v, B, O, "profile_lbfgsb", n_starts=len(starts), n_at_best=n_at_best)
        else:
            # every best start leaves the spectral box: joint constrained solve
            Bstarts = [build(x)[0] for _, x in sorted(results, key=lambda t: t[0])[:4]]
            Bstarts.append(np.zeros((P, P)))
            best = _joint_constrained(S, sup, Bstarts, cfg)
            cands = []
            if best is not None:
                cands.append((best[0], best[1], best[2], "joint_slsqp_spectral"))
            if feas:
                cands.append((feas[0][0], feas[0][2], feas[0][3], "profile_lbfgsb_feasible"))
            v, B, O, meth = min(cands, key=lambda t: t[0])
            fit = Fit(v, B, O, meth, n_starts=len(starts), n_at_best=n_at_best, fallback=True)
        x = np.array([B[e] for e in D])
        touch = np.abs(x) >= M - 1e-7
        if touch.any():
            fit.b_bound_touch = True
            _, g = _profile(x, S, D, U)
            # binding: the gradient pushes outward at an active coordinate
            fit.b_bound_binding = bool(np.any(touch & (np.sign(x) * g < -1e-7)))
    ev = np.linalg.eigvalsh(fit.O)
    fit.spec_bound_touch = bool(ev[0] <= lo + 1e-7 or ev[-1] >= hi - 1e-7)
    if np.any(np.abs(fit.B) >= M - 1e-7):
        fit.b_bound_touch = True
    if not _spectral_ok(fit.O, lo, hi, 1e-9):
        raise RuntimeError(f"infeasible fit on {sup.name}")
    return fit


def lmin_all(S, rng, cfg=CONFIG, repair=True):
    fits = [solve_support(S, s, rng, cfg) for s in SUPPORTS]
    repairs = []
    if repair:
        for _ in range(3):
            vals = np.array([f.value for f in fits])
            bad = NESTED[vals[NESTED[:, 0]] < vals[NESTED[:, 1]] - 1e-10]
            if len(bad) == 0:
                break
            for big in np.unique(bad[:, 1]):
                smalls = bad[bad[:, 1] == big, 0]
                before = fits[big].value
                starts = [fits[s].B for s in smalls]
                new = solve_support(S, SUPPORTS[big], rng, cfg, extra_starts=starts)
                if new.value < fits[big].value:
                    fits[big] = new
                repairs.append(dict(support=SUPPORTS[big].name, before=before, after=fits[big].value))
    return fits, repairs


# ----------------------------------------------------------------------------
# Algebraic class invariant
# ----------------------------------------------------------------------------

CONSTRAINTS = CONFIG["class_invariant"]["constraints"]


def _constraint_values(Sig):
    d = np.sqrt(np.diag(Sig))
    corr = Sig / np.outer(d, d)
    K = np.linalg.inv(Sig)
    kd = np.sqrt(np.diag(K))
    pc = -K / np.outer(kd, kd)
    return np.array([corr[0, 1], corr[0, 2], corr[1, 2], pc[0, 1], pc[0, 2], pc[1, 2]])


def random_parameter(sup, rng):
    while True:
        B = np.zeros((P, P))
        for e in sup.D:
            B[e] = rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)
        O = np.diag(rng.uniform(0.6, 1.6, P))
        for i, j in sup.U:
            O[i, j] = O[j, i] = rng.choice([-1, 1]) * rng.uniform(0.15, 0.5)
        if np.linalg.eigvalsh(O)[0] > 0.1:
            return B, O


def jacobian(B, O, sup):
    C = np.linalg.inv(np.eye(P) - B)
    Sig = C.T @ O @ C
    iu = np.triu_indices(P)
    cols = []
    for (i, j) in sup.D:
        E = np.zeros((P, P)); E[i, j] = 1
        cols.append((C.T @ E.T @ Sig + Sig @ E @ C)[iu])
    for i in range(P):
        E = np.zeros((P, P)); E[i, i] = 1
        cols.append((C.T @ E @ C)[iu])
    for (i, j) in sup.U:
        E = np.zeros((P, P)); E[i, j] = E[j, i] = 1
        cols.append((C.T @ E @ C)[iu])
    return np.column_stack(cols)


def algebraic_class(sup, cfg=CONFIG):
    c = cfg["class_invariant"]
    rng = np.random.default_rng(np.random.SeedSequence([c["class_seed"], sup.index]))
    rank = 0
    vmax = np.zeros(6)
    for _ in range(c["jacobian_draws"]):
        B, O = random_parameter(sup, rng)
        s = np.linalg.svd(jacobian(B, O, sup), compute_uv=False)
        rank = max(rank, int(np.sum(s > c["rank_rel_tol"] * s[0])))
        vmax = np.maximum(vmax, np.abs(_constraint_values(covariance(B, O))))
    vanish = tuple(n for n, v in zip(CONSTRAINTS, vmax) if v < c["vanish_tol"])
    return dict(dim=rank, vanishing=list(vanish), key=f"dim{rank}|" + ",".join(vanish),
                n_params=P + sup.count, smallest_nonvanishing=float(np.min(vmax[vmax >= c["vanish_tol"]])) if np.any(vmax >= c["vanish_tol"]) else None)


RELAXED = json.loads(json.dumps(CONFIG))
RELAXED["coefficient_bound"] = 1e3
RELAXED["omega_spectrum"] = [1e-9, 1e9]

CLASSES = None


def all_classes():
    global CLASSES
    if CLASSES is None:
        CLASSES = [algebraic_class(s) for s in SUPPORTS]
    return CLASSES


# ----------------------------------------------------------------------------
# Designs and data
# ----------------------------------------------------------------------------

def _parse_edge(s):
    if "<->" in s:
        a, b = s.split("<->")
    else:
        a, b = s.split("->")
    return int(a) - 1, int(b) - 1


def design_parameters(name, cfg=CONFIG):
    d = cfg["designs"][name]
    B = np.zeros((P, P))
    for k, v in d["B"].items():
        B[_parse_edge(k)] = v
    O = np.diag(np.array(d["Omega_diag"], float))
    for k, v in d["Omega_offdiag"].items():
        i, j = _parse_edge(k)
        O[i, j] = O[j, i] = v
    return B, O


def design_support(name):
    B, O = design_parameters(name)
    mo = np.abs(O) > 0
    np.fill_diagonal(mo, False)
    return support_from_masks(np.abs(B) > 0, mo)


def sample_covariance(Sigma, n, rng):
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, P)) @ L.T
    X -= X.mean(axis=0)
    return X.T @ X / n


# ----------------------------------------------------------------------------
# Local descent of the actual quasi-MCP criterion
# ----------------------------------------------------------------------------

def scalar_prox_exact(z, gamma, lam, delta, bound=np.inf):
    """Global minimizer of (x-z)^2/(2 gamma) + lam q_delta(x) on [-bound, bound]."""
    z = np.asarray(z, float)
    s = np.where(z >= 0, 1.0, -1.0)
    v = np.abs(z)
    k = 1 / gamma - 2 * lam / delta ** 2
    if k > 0:
        mid = np.clip((v / gamma - 2 * lam / delta) / k, 0, delta)
    else:
        mid = np.full_like(v, delta)
    cands = np.stack([np.zeros_like(v), mid, np.full_like(v, delta), np.maximum(v, delta)])
    cands = np.minimum(cands, bound)
    h = (cands - v) ** 2 / (2 * gamma) + lam * qmcp(cands, delta)
    return s * cands[np.argmin(h, axis=0), np.arange(len(v))]


def quasi_mcp_descent(S, B0, O0, order, lam, delta, cfg=CONFIG):
    lo, hi = cfg["omega_spectrum"]
    M = cfg["coefficient_bound"]
    ec = cfg["equivalence_check"]
    fwd = [(order[a], order[b]) for a in range(P) for b in range(a + 1, P)]
    B = np.zeros((P, P))
    for e in fwd:
        B[e] = B0[e]
    if np.any(np.abs(B0 - B) > 0):
        raise ValueError("start not compatible with order")
    O = O0.copy()

    def pack(B, O):
        return np.r_[[B[e] for e in fwd], np.diag(O), [O[pr] for pr in PAIRS]]

    def unpack(x):
        Bx = np.zeros((P, P))
        for v, e in zip(x[:3], fwd):
            Bx[e] = v
        Ox = np.diag(x[3:6]).astype(float)
        for v, (i, j) in zip(x[6:], PAIRS):
            Ox[i, j] = Ox[j, i] = v
        return Bx, Ox

    def F(x):
        Bx, Ox = unpack(x)
        ev = np.linalg.eigvalsh(Ox)
        if ev[0] < lo or ev[-1] > hi or np.max(np.abs(x[:3])) > M:
            return np.inf
        return loss(Bx, Ox, S) + lam * (qmcp(x[:3], delta).sum() + qmcp(x[6:], delta).sum())

    def grad(x):
        Bx, Ox = unpack(x)
        _, gb, go = loss_grad(Bx, Ox, S)
        return np.r_[[gb[e] for e in fwd], np.diag(go), [2 * go[pr] for pr in PAIRS]]

    x = pack(B, O)
    f = F(x)
    f0 = f
    gamma = 1e-2
    it = 0
    small = 0
    for it in range(ec["descent_max_iter"]):
        g = grad(x)
        accepted = False
        while gamma >= ec["descent_step_min"]:
            z = x - gamma * g
            xn = z.copy()
            xn[:3] = scalar_prox_exact(z[:3], gamma, lam, delta, M)
            xn[6:] = scalar_prox_exact(z[6:], gamma, lam, delta)
            fn = F(xn)
            step = xn - x
            if fn <= f - 1e-4 * step @ step / (2 * gamma):
                accepted = True
                break
            gamma /= 2
        if not accepted:
            break
        dec = f - fn
        x, f = xn, fn
        gamma = min(gamma * 2, 10.0)
        small = small + 1 if (dec < 1e-15 or np.max(np.abs(step)) < 1e-12) else 0
        if small >= 5:
            break
    return f0, f, it, unpack(x)


# ----------------------------------------------------------------------------
# One dataset
# ----------------------------------------------------------------------------

def emin_names(pop, design):
    return set(pop[design]["E_min_names"])


def run_dataset(job):
    design, di, n, rep, pop_info, cfg = job
    t0 = time.monotonic()
    master = cfg["master_seed"]
    rng_data = np.random.default_rng(np.random.SeedSequence([master, di, n, rep]))
    rng_solve = np.random.default_rng(np.random.SeedSequence([master, di, n, rep, 1]))
    Bt, Ot = design_parameters(design, cfg)
    Sigma = covariance(Bt, Ot)
    S = sample_covariance(Sigma, n, rng_data)
    lam = delta = math.log(n) / n
    fits, repairs = lmin_all(S, rng_solve, cfg)
    vals = np.array([f.value for f in fits])
    counts = np.array([s.count for s in SUPPORTS])
    score = vals + lam * counts
    order = np.argsort(score, kind="stable")
    best = float(score[order[0]])
    # Exact ties are expected (supports with the same covariance model have the
    # same constrained likelihood); pick the first tied support in the fixed
    # enumeration order so that the choice does not depend on rounding.
    k = int(min(np.where(score <= best + 1e-12)[0]))
    ties12 = [int(i) for i in np.where(score <= best + 1e-12)[0]]
    ties9 = [int(i) for i in np.where(score <= best + 1e-9)[0]]
    classes = all_classes()
    emin = set(pop_info["E_min_indices"])
    gen_key = pop_info["generating_class"]
    in_class = np.array([c["key"] == gen_key for c in classes])
    sel = SUPPORTS[k]
    fit = fits[k]
    mb, mo = sel.mask()

    def min_coord_ratio(i):
        s, f = SUPPORTS[i], fits[i]
        c = [abs(f.B[e]) for e in s.D] + [abs(f.O[pr]) for pr in s.U]
        return float(min(c) / delta) if c else np.inf
    min_ratio = min_coord_ratio(k)
    ties_min_ratio = min(min_coord_ratio(i) for i in ties12)
    # thresholded support of the global-minimizer candidate (refit on selected support)
    tb = (np.abs(fit.B) >= delta / 2) & mb
    to = (np.abs(fit.O) >= delta / 2) & mo
    th = support_from_masks(tb, to)
    # equivalence check: descent of the actual quasi-MCP criterion
    tops = [k] + [int(i) for i in order if int(i) != k][:cfg["equivalence_check"]["top_supports"] - 1]
    crit_sel = criterion(fit.B, fit.O, S, lam, delta)
    best_crit = crit_sel
    best_from = sel.name
    n_desc = 0
    max_iters_hit = 0
    for i in tops:
        s = SUPPORTS[i]
        for od in s.orders:
            f0, f1, iters, _ = quasi_mcp_descent(S, fits[i].B, fits[i].O, od, lam, delta, cfg)
            n_desc += 1
            max_iters_hit += int(iters >= cfg["equivalence_check"]["descent_max_iter"] - 1)
            if f1 < best_crit:
                best_crit, best_from = f1, f"{s.name}|order{''.join(str(v+1) for v in od)}"
    found_lower = bool(best_crit < best - cfg["equivalence_check"]["lower_tol"])
    equiv_pass = bool(min_ratio >= 1.0 and not found_lower)
    e_star = cfg["designs"][design]["e_star"]
    non_emin = np.array([i not in emin for i in range(len(SUPPORTS))])
    margin = float(score[non_emin].min() - score[~non_emin].min())
    runner = next(int(i) for i in order if score[i] > best + 1e-9)
    row = dict(
        design=design, n=n, replication=rep,
        data_seed=f"SeedSequence([{master},{di},{n},{rep}])",
        lambda_n=lam, delta_n=delta,
        selected_support=sel.name, selected_count=sel.count,
        selected_is_emin=k in emin, selected_in_generating_class=bool(in_class[k]),
        selected_bowed=sel.bowed,
        thresholded_support=th.name, thresholded_count=th.count,
        thresholded_is_emin=th.index in emin,
        thresholded_in_generating_class=bool(in_class[th.index]),
        fewer_than_e_star=th.count < e_star, more_than_e_star=th.count > e_star,
        n_ties_1e12=len(ties12), n_ties_1e9=len(ties9),
        ties_1e12=";".join(SUPPORTS[i].name for i in ties12),
        ties_1e9_all_emin=all(i in emin for i in ties9),
        ties_1e9_all_in_class=bool(all(in_class[i] for i in ties9)),
        ties_1e9_any_emin=any(i in emin for i in ties9),
        runner_up_support=SUPPORTS[runner].name, runner_up_gap=float(score[runner] - best),
        emin_margin=margin,
        l0_min_score=float(best), selected_lmin=float(vals[k]),
        refit_min_coord_over_delta=min_ratio, ties_min_coord_over_delta=ties_min_ratio,
        selected_fit_method=fit.method,
        criterion_at_refit=float(crit_sel),
        descent_best_criterion=float(best_crit), descent_best_from=best_from,
        descent_runs=n_desc, descent_iter_cap_hits=max_iters_hit,
        descent_found_lower=found_lower,
        equivalence_check_pass=equiv_pass,
        n_spectral_fallbacks=int(sum(f.fallback for f in fits)),
        n_b_bound_touch=int(sum(f.b_bound_touch for f in fits)),
        n_b_bound_binding=int(sum(f.b_bound_binding for f in fits)),
        n_spec_bound_touch=int(sum(f.spec_bound_touch for f in fits)),
        selected_b_bound_touch=fit.b_bound_touch, selected_spec_bound_touch=fit.spec_bound_touch,
        hierarchy_repairs=len(repairs),
        hierarchy_violation_after_repair=float(np.max(vals[NESTED[:, 1]] - vals[NESTED[:, 0]])),
        sample_min_eig=float(np.linalg.eigvalsh(S)[0]),
        runtime_seconds=time.monotonic() - t0,
    )
    extra = dict(design=design, n=n, replication=rep, lmin=vals.tolist(), repairs=repairs)
    return row, extra


# ----------------------------------------------------------------------------
# Population analysis
# ----------------------------------------------------------------------------

def population(cfg=CONFIG):
    classes = all_classes()
    out = {}
    for di, name in enumerate(cfg["designs"]):
        B, O = design_parameters(name, cfg)
        Sig = covariance(B, O)
        lstar = saturated_value(Sig)
        rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, 0, 0, 9]))
        fits, repairs = lmin_all(Sig, rng, cfg)
        gaps = np.array([f.value - lstar for f in fits])
        tol = cfg["exact_fit_tolerance"]
        fitting = [i for i in range(len(SUPPORTS)) if gaps[i] <= tol]
        cmin = min(SUPPORTS[i].count for i in fitting)
        emin = [i for i in fitting if SUPPORTS[i].count == cmin]
        gsup = design_support(name)
        gkey = classes[gsup.index]["key"]
        e_star = cfg["designs"][name]["e_star"]
        nonfit_le = [i for i in range(len(SUPPORTS)) if gaps[i] > tol and SUPPORTS[i].count <= e_star]
        near = [dict(support=SUPPORTS[i].name, gap=float(gaps[i])) for i in range(len(SUPPORTS))
                if tol < gaps[i] < 1e-6]
        rfits, _ = lmin_all(Sig, rng, RELAXED)
        rgaps = np.array([f.value - lstar for f in rfits])
        fit_outside_only = [dict(support=SUPPORTS[i].name, count=SUPPORTS[i].count, gap_in_K=float(gaps[i]),
                                 relaxed_gap=float(rgaps[i]), relaxed_max_abs_B=float(np.max(np.abs(rfits[i].B))),
                                 relaxed_omega_eig=[float(e) for e in np.linalg.eigvalsh(rfits[i].O)[[0, -1]]],
                                 in_generating_class=classes[i]["key"] == classes[design_support(name).index]["key"])
                            for i in range(len(SUPPORTS)) if rgaps[i] <= tol < gaps[i]]
        out[name] = dict(
            label=cfg["designs"][name]["label"], B=B.tolist(), Omega=O.tolist(), Sigma=Sig.tolist(),
            l_star=lstar, generating_support=gsup.name, e_star=e_star,
            generating_class=gkey, generating_class_dim=classes[gsup.index]["dim"],
            generating_class_vanishing=classes[gsup.index]["vanishing"],
            generating_support_gap=float(gaps[gsup.index]),
            n_exactly_fitting=len(fitting), minimal_fitting_count=cmin,
            minimal_count_equals_e_star=cmin == e_star,
            E_min_indices=emin, E_min_names=[SUPPORTS[i].name for i in emin],
            E_min=[dict(support=SUPPORTS[i].name, count=SUPPORTS[i].count, bowed=SUPPORTS[i].bowed,
                        dim=classes[i]["dim"], vanishing=classes[i]["vanishing"],
                        in_generating_class=classes[i]["key"] == gkey, gap=float(gaps[i]),
                        B=fits[i].B.tolist(), Omega=fits[i].O.tolist(),
                        min_abs_coordinate=float(min(
                            [abs(fits[i].B[e]) for e in SUPPORTS[i].D] +
                            [abs(fits[i].O[pr]) for pr in SUPPORTS[i].U])),
                        ) for i in emin],
            all_E_min_in_generating_class=all(classes[i]["key"] == gkey for i in emin),
            n_generating_class_supports_with_e_star_edges=sum(
                1 for i, s in enumerate(SUPPORTS) if s.count == e_star and classes[i]["key"] == gkey),
            smallest_gap_nonfitting_count_le_e_star=float(gaps[nonfit_le].min()) if nonfit_le else None,
            smallest_gap_support=SUPPORTS[nonfit_le[int(np.argmin(gaps[nonfit_le]))]].name if nonfit_le else None,
            largest_gap_fitting=float(gaps[fitting].max()),
            near_fitting_anomalies=near, hierarchy_repairs=repairs,
            fit_exactly_only_outside_K=fit_outside_only,
            bound_touch_supports=[dict(support=SUPPORTS[i].name, B_bound=fits[i].b_bound_touch,
                                       B_bound_binding=fits[i].b_bound_binding,
                                       spectral_bound=fits[i].spec_bound_touch, gap=float(gaps[i]))
                                  for i in range(len(SUPPORTS))
                                  if fits[i].b_bound_touch or fits[i].spec_bound_touch],
            E_min_touching_bound=[SUPPORTS[i].name for i in emin
                                  if fits[i].b_bound_touch or fits[i].spec_bound_touch],
            relaxed_minimal_fitting_count=int(min(SUPPORTS[i].count for i in range(len(SUPPORTS)) if rgaps[i] <= tol)),
            gaps={SUPPORTS[i].name: float(gaps[i]) for i in range(len(SUPPORTS))},
        )
    return out


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def random_S(rng):
    A = rng.standard_normal((P, P))
    Sig = A @ A.T + 0.5 * np.eye(P)
    d = np.sqrt(np.diag(Sig))
    Sig = Sig / np.outer(d, d)
    df = int(rng.integers(8, 40))
    Z = rng.multivariate_normal(np.zeros(P), Sig, size=df)
    Z -= Z.mean(axis=0)
    return Z.T @ Z / df


def validate(cfg=CONFIG):
    v = cfg["validation"]
    rng = np.random.default_rng(v["validation_seed"])
    classes = all_classes()
    records = []
    checks = {}
    # support count
    checks["support_count"] = dict(passed=len(SUPPORTS) == 200, supports=len(SUPPORTS),
                                   acyclic_directed=N_DIRECTED_ACYCLIC, bidirected_subsets=N_BIDIRECTED,
                                   nested_pairs=int(len(NESTED)))
    # inner closed form for |U| = 2 vs direct optimization over Omega
    err_inner = 0.0
    for _ in range(10):
        R = random_S(rng)
        for U in [(PAIRS[0], PAIRS[1]), (PAIRS[0], PAIRS[2]), (PAIRS[1], PAIRS[2])]:
            O = inner_omega(R, U)
            val = loss(np.zeros((P, P)), O, R)
            def fun(z):
                Ox = np.diag(z[:3]).astype(float)
                for vv, pr in zip(z[3:], U):
                    Ox[pr] = Ox[pr[::-1]] = vv
                return loss(np.zeros((P, P)), Ox, R)
            best = np.inf
            for s in range(4):
                z0 = np.r_[np.diag(R), [R[pr] * (0.5 * s / 3) for pr in U]]
                r = minimize(fun, z0, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000, maxfev=40000))
                best = min(best, r.fun)
            err_inner = max(err_inner, val - best)
    checks["inner_closed_form_vs_direct"] = dict(passed=err_inner < 1e-9, max_closed_minus_direct=err_inner)
    # profile gradient
    gerr = 0.0
    for _ in range(5):
        S = random_S(rng)
        for sup in SUPPORTS:
            if not sup.D:
                continue
            x = rng.uniform(-1, 1, len(sup.D))
            _, g = _profile(x, S, sup.D, sup.U)
            h = 1e-6
            num = np.array([(_profile(x + h * e, S, sup.D, sup.U)[0] - _profile(x - h * e, S, sup.D, sup.U)[0]) / (2 * h)
                            for e in np.eye(len(x))])
            gerr = max(gerr, float(np.max(np.abs(num - g))))
    checks["profile_envelope_gradient"] = dict(passed=gerr < 1e-6, max_abs_error=gerr)
    # (a), (b), (c) on random S (cold solver: no hierarchy repair)
    sat = [s.index for s in SUPPORTS if classes[s.index]["dim"] == 6]
    worst_sat, worst_dag, worst_mono = 0.0, 0.0, 0.0
    sat_fail, mono_fail = [], []
    bound_touch = 0
    fallback = 0
    for r in range(v["random_S"]):
        S = random_S(rng)
        srng = np.random.default_rng(np.random.SeedSequence([v["validation_seed"], r]))
        fits, _ = lmin_all(S, srng, cfg, repair=False)
        vals = np.array([f.value for f in fits])
        bound_touch += sum(f.b_bound_touch or f.spec_bound_touch for f in fits)
        fallback += sum(f.fallback for f in fits)
        ls = saturated_value(S)
        for i in sat:
            d = abs(vals[i] - ls)
            worst_sat = max(worst_sat, d)
            if d > v["saturated_tol"]:
                rf = solve_support(S, SUPPORTS[i], srng, RELAXED)
                outside = bool(np.max(np.abs(rf.B)) > cfg["coefficient_bound"] or
                               not _spectral_ok(rf.O, *cfg["omega_spectrum"]))
                sat_fail.append(dict(S=r, support=SUPPORTS[i].name, gap=float(vals[i] - ls),
                                     b_bound=fits[i].b_bound_touch, spec_bound=fits[i].spec_bound_touch,
                                     relaxed_gap=float(rf.value - ls), relaxed_fit_outside_K=outside,
                                     relaxed_max_abs_B=float(np.max(np.abs(rf.B))),
                                     relaxed_omega_eig=[float(e) for e in np.linalg.eigvalsh(rf.O)[[0, -1]]],
                                     domain_limited=bool(abs(rf.value - ls) <= v["saturated_tol"] and outside)))
        for s in SUPPORTS:
            if s.U or not s.D:
                continue
            num = solve_support(S, s, srng, cfg, force_numeric=True).value
            exact = regression_mle(S, s.D)[0]
            worst_dag = max(worst_dag, abs(num - exact))
        viol = vals[NESTED[:, 1]] - vals[NESTED[:, 0]]
        worst_mono = max(worst_mono, float(viol.max()))
        for (a, b), d in zip(NESTED, viol):
            if d > v["monotone_tol"]:
                mono_fail.append(dict(S=r, small=SUPPORTS[a].name, big=SUPPORTS[b].name, excess=float(d)))
    checks["a_saturated_attain"] = dict(passed=all(f["domain_limited"] for f in sat_fail),
                                        rule="every saturated support attains 0.5 log det S + p/2 within 1e-9, "
                                             "except where the exact saturated fit lies outside K (verified by "
                                             "re-solving on a relaxed domain)",
                                        n_domain_limited=sum(f["domain_limited"] for f in sat_fail),
                                        n_checked=len(sat) * v["random_S"], saturated_supports=len(sat), random_S=v["random_S"],
                                        max_abs_error=worst_sat, failures=sat_fail[:50], n_failures=len(sat_fail))
    checks["b_dag_numeric_vs_regression"] = dict(passed=worst_dag < v["dag_tol"], max_abs_error=worst_dag,
                                                 dag_supports=24, random_S=v["random_S"])
    checks["c_monotonicity"] = dict(passed=not mono_fail, nested_pairs=int(len(NESTED)), random_S=v["random_S"],
                                    max_violation=worst_mono, failures=mono_fail[:50], n_failures=len(mono_fail))
    checks["validation_bound_touch_count"] = bound_touch
    checks["validation_spectral_fallbacks"] = fallback
    # descent sanity: an added sub-threshold coordinate must be removed
    Bt, Ot = design_parameters("G1", cfg)
    Sig = covariance(Bt, Ot)
    lam_ = dl_ = 0.05
    Bs = Bt.copy(); Bs[0, 2] = 0.3 * dl_
    Os = Ot.copy(); Os[0, 1] = Os[1, 0] = 0.4 * dl_
    f0, f1, iters, (Bd, Od) = quasi_mcp_descent(Sig, Bs, Os, (0, 1, 2), lam_, dl_, cfg)
    target = saturated_value(Sig) + lam_ * 2
    checks["descent_removes_small_coordinates"] = dict(
        passed=bool(f1 <= target + 1e-9 and abs(Bd[0, 2]) == 0 and Od[0, 1] == 0),
        start=f0, end=f1, target=target, iterations=iters,
        final_B13=float(Bd[0, 2]), final_Omega12=float(Od[0, 1]))
    # scalar prox exact vs brute force
    perr = 0.0
    for _ in range(200):
        z = rng.uniform(-3, 3)
        gam, lam_, dl = 10 ** rng.uniform(-4, 0), 10 ** rng.uniform(-3, -1), 10 ** rng.uniform(-3, -1)
        x = scalar_prox_exact(np.array([z]), gam, lam_, dl, 2.0)[0]
        grid = np.linspace(-2, 2, 400001)
        h = lambda t: (t - z) ** 2 / (2 * gam) + lam_ * qmcp(t, dl)
        perr = max(perr, float(h(x) - np.min(h(np.r_[grid, 0.0, np.clip(z, -2, 2), dl, -dl]))))
    checks["scalar_prox_exact_vs_grid"] = dict(passed=perr < 1e-9, max_excess=perr)
    return checks


# ----------------------------------------------------------------------------
# Summary helpers
# ----------------------------------------------------------------------------

def _jd(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def write_csv(path, rows):
    """Write dict rows to csv; gzip-compressed when the name ends in .gz."""
    import csv
    fields = list(dict.fromkeys(k for r in rows for k in r))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, cfg=CONFIG):
    out = []
    for d in cfg["designs"]:
        for n in cfg["ns"]:
            rs = [r for r in rows if r["design"] == d and r["n"] == n]
            m = len(rs)
            k = sum(r["thresholded_is_emin"] for r in rs)
            lo, hi = wilson(k, m)
            out.append(dict(design=d, label=cfg["designs"][d]["label"], n=n, replications=m,
                            e_star=cfg["designs"][d]["e_star"],
                            prop_emin=k / m, emin_wilson_lo=lo, emin_wilson_hi=hi,
                            prop_generating_class=sum(r["thresholded_in_generating_class"] for r in rs) / m,
                            prop_fewer_than_e_star=sum(r["fewer_than_e_star"] for r in rs) / m,
                            prop_more_than_e_star=sum(r["more_than_e_star"] for r in rs) / m,
                            prop_equivalence_check_pass=sum(r["equivalence_check_pass"] for r in rs) / m,
                            prop_threshold_changed=sum(r["thresholded_support"] != r["selected_support"] for r in rs) / m,
                            median_runtime_s=float(np.median([r["runtime_seconds"] for r in rs]))))
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true", help="run every 97th sample only")
    ap.add_argument("--force", action="store_true", help="run even if a validation check fails")
    ap.add_argument("--out", default=str(RESULTS), help="output directory (default: results/)")
    args = ap.parse_args()
    out = Path(args.out)
    t0 = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    print("supports", len(SUPPORTS), "directed acyclic", N_DIRECTED_ACYCLIC, flush=True)
    t_val = time.monotonic()
    checks = validate()
    t_val = time.monotonic() - t_val
    print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else v) for k, v in checks.items()}, default=_jd), flush=True)
    (out / "validation.json").write_text(json.dumps(checks, indent=2, default=_jd) + "\n")
    required = ["support_count", "a_saturated_attain", "b_dag_numeric_vs_regression", "c_monotonicity",
                "inner_closed_form_vs_direct", "profile_envelope_gradient", "scalar_prox_exact_vs_grid",
                "descent_removes_small_coordinates"]
    if not all(checks[k]["passed"] for k in required) and not args.force:
        print("validation failed; study not run", flush=True)
        sys.exit(1)
    t_pop = time.monotonic()
    pop = population()
    t_pop = time.monotonic() - t_pop
    classes = all_classes()
    pop_out = dict(
        supports=[dict(index=s.index, support=s.name, count=s.count, bowed=s.bowed,
                       dim=classes[s.index]["dim"], vanishing=classes[s.index]["vanishing"])
                  for s in SUPPORTS],
        designs=pop)
    (out / "population.json").write_text(json.dumps(pop_out, indent=2, default=_jd) + "\n")
    for d, info in pop.items():
        print(d, "e_min", info["minimal_fitting_count"], info["E_min_names"], info["all_E_min_in_generating_class"], flush=True)
    jobs = []
    for di, d in enumerate(CONFIG["designs"]):
        for n in CONFIG["ns"]:
            for rep in range(CONFIG["replications"]):
                jobs.append((d, di, n, rep, dict(E_min_indices=pop[d]["E_min_indices"],
                                                   generating_class=pop[d]["generating_class"]), CONFIG))
    if args.quick:
        jobs = jobs[::97]
    rows, extras = [], []
    t_sim = time.monotonic()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for k, (row, extra) in enumerate(pool.imap_unordered(run_dataset, jobs, chunksize=2)):
            rows.append(row)
            extras.append(extra)
            if k % 200 == 0:
                print(f"{k+1}/{len(jobs)} {time.monotonic()-t_sim:.0f}s", flush=True)
    t_sim = time.monotonic() - t_sim
    rows.sort(key=lambda r: (r["design"], r["n"], r["replication"]))
    extras.sort(key=lambda r: (r["design"], r["n"], r["replication"]))
    write_csv(out / "runs.csv.gz", rows)
    write_csv(out / "summary.csv", summarize(rows))
    np.savez_compressed(out / "lmin_all_datasets.npz",
                        lmin=np.array([e["lmin"] for e in extras]),
                        design=np.array([e["design"] for e in extras]),
                        n=np.array([e["n"] for e in extras]),
                        replication=np.array([e["replication"] for e in extras]),
                        support_names=np.array([s.name for s in SUPPORTS]))
    meta = dict(
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        cpu_count=os.cpu_count(), workers=args.workers,
        runtime_seconds=dict(total=time.monotonic() - t0, validation=t_val, population=t_pop, simulation=t_sim,
                             dataset_cpu_sum=float(sum(r["runtime_seconds"] for r in rows))),
        n_datasets=len(rows), validation=checks,
        repairs_total=int(sum(len(e["repairs"]) for e in extras)),
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=_jd) + "\n")
    print("done", len(rows), f"{time.monotonic()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Minimum joint l1 norm over the exact representations of a covariance (Supplement S13).

Conventions follow Section 2 of the main article.  B[i, j] != 0 is the
directed edge i -> j, Omega[i, j] != 0 (i < j) is the bidirected edge i <-> j,
Sigma = Phi(B, Omega) = (I - B)^{-T} Omega (I - B)^{-1}.  For a fixed acyclic B
the error covariance of an exact representation is determined:
    Omega(B) = (I - B)^T Sigma (I - B),   Omega_ij = w_i^T Sigma w_j,  w_j = e_j - B[:, j],
and it is positive definite whenever Sigma is.  The joint l1 norm is
    R_1(B) = sum_{i != j} |B_ij| + sum_{i < j} |Omega_ij(B)|.
A support H = (D, U) (D acyclic directed coordinates, U unordered pairs) admits the
exact representations {B supported on D : Omega_ij(B) = 0 for every pair not in U}.

Solver S1 (this file): epigraph reformulation
    min sum(t)  s.t.  t_b >= +-b,  t_u >= +-Omega_U(b),  Omega_Z(b) = 0   (Z = pairs not in U)
solved by SLSQP with analytic Jacobians from many starts (zero, OLS on D, coarse grid,
uniform random), followed by an active-set polish (the terms that vanish at the
solution become equality constraints, the others keep their signs, and the smooth
problem is re-solved at tight tolerance).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", message="Values in x were outside bounds")


# ----------------------------------------------------------------------------
# Supports
# ----------------------------------------------------------------------------

def pairs_of(p):
    return [(i, j) for i in range(p) for j in range(i + 1, p)]


def directed_of(p):
    return [(i, j) for i in range(p) for j in range(p) if i != j]


def is_acyclic(D, p):
    adj = {i: [j for (a, j) in D if a == i] for i in range(p)}
    state = [0] * p

    def dfs(v):
        state[v] = 1
        for w in adj[v]:
            if state[w] == 1 or (state[w] == 0 and not dfs(w)):
                return False
        state[v] = 2
        return True
    return all(state[v] == 2 or dfs(v) for v in range(p))


@lru_cache(maxsize=None)
def acyclic_digraphs(p):
    dirs = directed_of(p)
    out = []
    for m in range(1 << len(dirs)):
        D = tuple(e for k, e in enumerate(dirs) if m >> k & 1)
        if is_acyclic(D, p):
            out.append(D)
    return out


@lru_cache(maxsize=None)
def bidirected_sets(p):
    prs = pairs_of(p)
    return [tuple(pr for k, pr in enumerate(prs) if m >> k & 1) for m in range(1 << len(prs))]


@lru_cache(maxsize=None)
def all_supports(p):
    """All acyclic mixed supports (D, U), bows allowed; 200 for p = 3, 34,752 for p = 4."""
    return [(D, U) for D in acyclic_digraphs(p) for U in bidirected_sets(p)]


def complete_supports(p):
    """One complete support per vertex order: all forward pairs directed, all pairs bidirected."""
    out = []
    for perm in itertools.permutations(range(p)):
        D = tuple((perm[a], perm[b]) for a in range(p) for b in range(a + 1, p))
        out.append((tuple(sorted(D)), tuple(pairs_of(p)), perm))
    return out


def support_name(D, U):
    parts = [f"{i+1}->{j+1}" for (i, j) in sorted(D)] + [f"{i+1}<->{j+1}" for (i, j) in sorted(U)]
    return ",".join(parts) if parts else "empty"


def parse_support(name):
    D, U = [], []
    if name == "empty":
        return (), ()
    for tok in name.split(","):
        if "<->" in tok:
            a, b = tok.split("<->")
            a, b = int(a) - 1, int(b) - 1
            U.append((min(a, b), max(a, b)))
        else:
            a, b = tok.split("->")
            D.append((int(a) - 1, int(b) - 1))
    return tuple(sorted(D)), tuple(sorted(U))


def is_bowfree(D, U):
    und = {tuple(sorted(e)) for e in D}
    return not any(u in und for u in U) and len(und) == len(D)


def topo_order(D, p):
    for perm in itertools.permutations(range(p)):
        pos = {v: k for k, v in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in D):
            return perm
    raise ValueError("cyclic")


# ----------------------------------------------------------------------------
# Model maps
# ----------------------------------------------------------------------------

def phi(B, O):
    p = len(B)
    C = np.linalg.inv(np.eye(p) - B)
    return C.T @ O @ C


def omega_of(B, S):
    W = np.eye(len(B)) - B
    return W.T @ S @ W


def standardize(S):
    d = 1.0 / np.sqrt(np.diag(S))
    return S * np.outer(d, d)


def r1_value(B, O):
    p = len(B)
    iu = np.triu_indices(p, 1)
    return float(np.abs(B).sum() + np.abs(O[iu]).sum())


def actual_support(B, O, tol=1e-9):
    p = len(B)
    D = tuple((i, j) for i in range(p) for j in range(p) if i != j and abs(B[i, j]) > tol)
    U = tuple((i, j) for (i, j) in pairs_of(p) if abs(O[i, j]) > tol)
    return D, U


# ----------------------------------------------------------------------------
# Algebraic invariants
# ----------------------------------------------------------------------------

def random_parameter(D, U, p, rng, coef=(0.3, 1.2), od=(0.15, 0.5)):
    while True:
        B = np.zeros((p, p))
        for e in D:
            B[e] = rng.choice([-1, 1]) * rng.uniform(*coef)
        O = np.diag(rng.uniform(0.6, 1.6, p))
        for i, j in U:
            O[i, j] = O[j, i] = rng.choice([-1, 1]) * rng.uniform(*od)
        if np.linalg.eigvalsh(O)[0] > 0.1:
            return B, O


def jacobian(B, O, D, U):
    p = len(B)
    C = np.linalg.inv(np.eye(p) - B)
    Sig = C.T @ O @ C
    iu = np.triu_indices(p)
    cols = []
    for (i, j) in D:
        E = np.zeros((p, p)); E[i, j] = 1
        cols.append((C.T @ E.T @ Sig + Sig @ E @ C)[iu])
    for i in range(p):
        E = np.zeros((p, p)); E[i, i] = 1
        cols.append((C.T @ E @ C)[iu])
    for (i, j) in U:
        E = np.zeros((p, p)); E[i, j] = E[j, i] = 1
        cols.append((C.T @ E @ C)[iu])
    return np.column_stack(cols)


@lru_cache(maxsize=None)
def constraint_names(p):
    names = []
    for (i, j) in pairs_of(p):
        rest = [k for k in range(p) if k not in (i, j)]
        for r in range(len(rest) + 1):
            for C in itertools.combinations(rest, r):
                names.append(("pc", i, j, C))
    if p >= 4:
        for quad in itertools.combinations(range(p), 4):
            a, b, c, d = quad
            # the three tetrads on four variables
            names.append(("tetrad", (a, b), (c, d), (a, c), (b, d)))
            names.append(("tetrad", (a, b), (c, d), (a, d), (b, c)))
            names.append(("tetrad", (a, c), (b, d), (a, d), (b, c)))
    return names


def constraint_label(c):
    if c[0] == "pc":
        _, i, j, C = c
        return f"rho{i+1}{j+1}" + ("." + "".join(str(k + 1) for k in C) if C else "")
    _, (a, b), (c_, d), (e, f), (g, h) = c
    return f"tet[{a+1}{b+1}*{c_+1}{d+1}-{e+1}{f+1}*{g+1}{h+1}]"


def constraint_values(S):
    """Correlations and partial correlations given every conditioning set, and tetrads of the correlation matrix."""
    p = len(S)
    R = standardize(S)
    out = []
    for c in constraint_names(p):
        if c[0] == "pc":
            _, i, j, C = c
            idx = [i, j] + list(C)
            K = np.linalg.inv(R[np.ix_(idx, idx)])
            out.append(-K[0, 1] / np.sqrt(K[0, 0] * K[1, 1]))
        else:
            _, (a, b), (c_, d), (e, f), (g, h) = c
            out.append(R[a, b] * R[c_, d] - R[e, f] * R[g, h])
    return np.array(out)


def invariant(D, U, p, seed, draws=4, rank_tol=1e-8, vanish_tol=1e-10):
    rng = np.random.default_rng(seed)
    rank = 0
    vmax = None
    for _ in range(draws):
        B, O = random_parameter(D, U, p, rng)
        s = np.linalg.svd(jacobian(B, O, D, U), compute_uv=False)
        rank = max(rank, int(np.sum(s > rank_tol * s[0])))
        v = np.abs(constraint_values(phi(B, O)))
        vmax = v if vmax is None else np.maximum(vmax, v)
    vanish = tuple(int(k) for k in np.flatnonzero(vmax < vanish_tol))
    nonvan = vmax[vmax >= vanish_tol]
    return rank, vanish, float(nonvan.min()) if len(nonvan) else np.inf


# ----------------------------------------------------------------------------
# Solver S1: epigraph SLSQP
# ----------------------------------------------------------------------------

@dataclass
class Problem:
    S: np.ndarray
    D: tuple
    U: tuple
    p: int = field(init=False)
    Z: tuple = field(init=False)

    def __post_init__(self):
        self.p = len(self.S)
        self.Z = tuple(pr for pr in pairs_of(self.p) if pr not in set(self.U))
        self.k = len(self.D)
        self.m = len(self.U)
        self.I = np.array([e[0] for e in self.D], int)
        self.J = np.array([e[1] for e in self.D], int)

    def B_of(self, b):
        B = np.zeros((self.p, self.p))
        if self.k:
            B[self.I, self.J] = b
        return B

    def pair_vals_jac(self, b, prs):
        """Omega entries on the pairs prs and their Jacobian with respect to b."""
        B = self.B_of(b)
        W = np.eye(self.p) - B
        SW = self.S @ W
        O = W.T @ SW
        if not prs:
            return np.zeros(0), np.zeros((0, self.k)), O
        K = np.array([q[0] for q in prs]); L = np.array([q[1] for q in prs])
        vals = O[K, L]
        if self.k == 0:
            return vals, np.zeros((len(prs), 0)), O
        Jm = -((K[:, None] == self.J[None, :]) * SW[self.I[None, :], L[:, None]]
               + (L[:, None] == self.J[None, :]) * SW[self.I[None, :], K[:, None]])
        return vals, Jm, O

    def r1(self, b):
        O = omega_of(self.B_of(b), self.S)
        return float(np.abs(b).sum() + sum(abs(O[q]) for q in self.U))

    def eq_resid(self, b):
        if not self.Z:
            return 0.0
        O = omega_of(self.B_of(b), self.S)
        return float(max(abs(O[q]) for q in self.Z))

    def terms(self, b):
        """All penalized terms: directed coefficients then bidirected Omega entries."""
        vu, _, _ = self.pair_vals_jac(b, list(self.U))
        return np.concatenate([b, vu])

    def terms_jac(self, b):
        vu, Ju, _ = self.pair_vals_jac(b, list(self.U))
        return np.concatenate([b, vu]), np.vstack([np.eye(self.k), Ju]) if (self.k + self.m) else np.zeros((0, self.k))


def _indep_rows(J, tol=1e-8):
    """Indices of a maximal set of linearly independent rows of J (QR with column pivoting on J^T)."""
    if J.shape[0] == 0 or J.shape[1] == 0:
        return []
    from scipy.linalg import qr
    _, R, piv = qr(J.T, pivoting=True, mode="economic")
    d = np.abs(np.diag(R))
    scale = max(1.0, d[0]) if len(d) else 1.0
    r = int(np.sum(d > tol * scale))
    return sorted(int(x) for x in piv[:r])


def _phase1(pb: Problem, b0, tol=1e-12):
    """Feasibility: least squares on Omega_Z(b) = 0 (handles redundant, consistent systems)."""
    from scipy.optimize import least_squares
    Z = list(pb.Z)

    def fun(b):
        return pb.pair_vals_jac(b, Z)[0]

    def jac(b):
        return pb.pair_vals_jac(b, Z)[1]
    method = "lm" if len(Z) >= pb.k else "trf"
    try:
        res = least_squares(fun, b0, jac=jac, method=method, xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=200 * (pb.k + 1))
    except Exception:
        return None
    b = res.x
    # Newton steps with the pseudo-inverse
    for _ in range(5):
        v, J, _ = pb.pair_vals_jac(b, Z)
        if np.max(np.abs(v)) < 1e-15:
            break
        b = b - np.linalg.pinv(J, rcond=1e-12) @ v
    if not np.all(np.isfinite(b)) or pb.eq_resid(b) > 1e-10:
        return None
    return b


def _slsqp_epigraph(pb: Problem, b0, eq_rows=None, ftol=1e-13, maxiter=400):
    k, m = pb.k, pb.m
    n = k + k + m
    U = list(pb.U)
    Z = list(pb.Z) if eq_rows is None else [pb.Z[i] for i in eq_rows]

    def obj(z):
        return float(z[k:].sum())

    def objg(z):
        g = np.zeros(n); g[k:] = 1.0
        return g

    def ineq(z):
        b = z[:k]; t = z[k:]
        vu, _, _ = pb.pair_vals_jac(b, U)
        return np.concatenate([t[:k] - b, t[:k] + b, t[k:] - vu, t[k:] + vu])

    def ineq_j(z):
        b = z[:k]
        _, Ju, _ = pb.pair_vals_jac(b, U)
        Jg = np.zeros((2 * k + 2 * m, n))
        Ik = np.eye(k)
        Jg[:k, :k] = -Ik; Jg[:k, k:2 * k] = Ik
        Jg[k:2 * k, :k] = Ik; Jg[k:2 * k, k:2 * k] = Ik
        if m:
            Jg[2 * k:2 * k + m, :k] = -Ju; Jg[2 * k:2 * k + m, 2 * k:] = np.eye(m)
            Jg[2 * k + m:, :k] = Ju; Jg[2 * k + m:, 2 * k:] = np.eye(m)
        return Jg

    def eq(z):
        vz, _, _ = pb.pair_vals_jac(z[:k], Z)
        return vz

    def eq_j(z):
        _, Jz, _ = pb.pair_vals_jac(z[:k], Z)
        J = np.zeros((len(Z), n)); J[:, :k] = Jz
        return J

    vu0, _, _ = pb.pair_vals_jac(b0, U)
    z0 = np.concatenate([b0, np.abs(b0) + 1e-3, np.abs(vu0) + 1e-3])
    cons = []
    if 2 * k + 2 * m:
        cons.append(dict(type="ineq", fun=ineq, jac=ineq_j))
    if Z:
        cons.append(dict(type="eq", fun=eq, jac=eq_j))
    res = minimize(obj, z0, jac=objg, constraints=cons, method="SLSQP",
                   options=dict(ftol=ftol, maxiter=maxiter))
    return res.x[:k]


def _polish(pb: Problem, b, act_tol=1e-6, ftol=1e-15):
    """Active-set polish: vanishing terms become equalities, the others keep their sign."""
    if pb.k == 0:
        return b
    t = pb.terms(b)
    act = np.flatnonzero(np.abs(t) < act_tol)
    inact = np.flatnonzero(np.abs(t) >= act_tol)
    sg = np.sign(t[inact])
    Zall = list(pb.Z)
    # independent subset of (active terms, zero constraints) at b
    _, Jt = pb.terms_jac(b)
    _, Jz, _ = pb.pair_vals_jac(b, Zall)
    Jeq = np.vstack([Jt[act], Jz])
    rows = _indep_rows(Jeq)
    na = len(act)
    act_rows = [act[i] for i in rows if i < na]
    z_rows = [Zall[i - na] for i in rows if i >= na]

    def obj(x):
        tt, _ = pb.terms_jac(x)
        return float(sg @ tt[inact])

    def objg(x):
        _, Jt_ = pb.terms_jac(x)
        return sg @ Jt_[inact]

    def eq(x):
        tt, _ = pb.terms_jac(x)
        vz, _, _ = pb.pair_vals_jac(x, z_rows)
        return np.concatenate([tt[act_rows], vz])

    def eq_j(x):
        _, Jt_ = pb.terms_jac(x)
        _, Jz_, _ = pb.pair_vals_jac(x, z_rows)
        return np.vstack([Jt_[act_rows], Jz_])

    def ineq(x):  # keep the signs of the inactive terms
        tt, _ = pb.terms_jac(x)
        return sg * tt[inact]

    def ineq_j(x):
        _, Jt_ = pb.terms_jac(x)
        return sg[:, None] * Jt_[inact]

    if len(rows) >= pb.k:
        # vertex-type solution: Newton on the k independent active equations
        x = b.copy()
        for _ in range(8):
            vals = np.concatenate([pb.terms_jac(x)[0][act_rows], pb.pair_vals_jac(x, z_rows)[0]])
            if np.max(np.abs(vals)) < 1e-16:
                break
            Jn = np.vstack([pb.terms_jac(x)[1][act_rows], pb.pair_vals_jac(x, z_rows)[1]])
            x = x - np.linalg.lstsq(Jn, vals, rcond=None)[0]
    else:
        cons = []
        if len(act_rows) + len(z_rows):
            cons.append(dict(type="eq", fun=eq, jac=eq_j))
        if len(inact):
            cons.append(dict(type="ineq", fun=ineq, jac=ineq_j))
        try:
            res = minimize(obj, b, jac=objg, constraints=cons, method="SLSQP",
                           options=dict(ftol=ftol, maxiter=40))
            x = res.x
        except Exception:
            return b
    if not np.all(np.isfinite(x)):
        return b
    if len(inact) and np.any(sg * pb.terms(x)[inact] < 0):
        return b
    # restore exact feasibility of the dropped (dependent) constraints, if needed
    if Zall and pb.eq_resid(x) > 1e-13:
        for _ in range(5):
            v, J, _ = pb.pair_vals_jac(x, Zall)
            x = x - np.linalg.pinv(J, rcond=1e-12) @ v
    if pb.eq_resid(x) <= max(pb.eq_resid(b), 1e-13) and pb.r1(x) <= pb.r1(b) + 1e-14:
        return x
    return b


def ols_start(S, D, p):
    B = np.zeros((p, p))
    for j in range(p):
        pa = [i for (i, jj) in D if jj == j]
        if pa:
            B[pa, j] = np.linalg.solve(S[np.ix_(pa, pa)], S[pa, j])
    return np.array([B[e] for e in D])


def make_starts(pb: Problem, rng, n_random, grid_levels=(-1.0, -0.5, 0.0, 0.5, 1.0), grid_max_dim=3,
                scale=1.5, extra=()):
    k = pb.k
    starts = [np.zeros(k), ols_start(pb.S, pb.D, pb.p)]
    if 0 < k <= grid_max_dim:
        starts += [np.array(g, float) for g in itertools.product(grid_levels, repeat=k)]
    starts += [rng.uniform(-scale, scale, k) for _ in range(n_random)]
    starts += [np.asarray(e, float) for e in extra]
    return starts


@dataclass
class Sol:
    value: float
    b: np.ndarray
    B: np.ndarray
    O: np.ndarray
    eq_resid: float
    support: tuple
    n_feasible: int
    local_values: list


def solve_support(S, D, U, rng, n_random=20, feas_tol=1e-9, keep_locals=False, **kw):
    """Minimize R_1 over exact representations with support contained in (D, U).

    Returns None if no start reaches a point with max |Omega_Z| <= feas_tol."""
    pb = Problem(S, tuple(D), tuple(U))
    p = pb.p
    if pb.k == 0:
        O = S.copy()
        if pb.eq_resid(np.zeros(0)) > feas_tol:
            return None
        v = pb.r1(np.zeros(0))
        sup = actual_support(np.zeros((p, p)), O)
        return Sol(v, np.zeros(0), np.zeros((p, p)), O, pb.eq_resid(np.zeros(0)), sup, 1, [v])
    best = None
    locals_ = []
    nfeas = 0
    for b0 in make_starts(pb, rng, n_random, **kw):
        rows = None
        if pb.Z:
            bf = _phase1(pb, b0)
            if bf is None:
                continue
            _, Jz, _ = pb.pair_vals_jac(bf, list(pb.Z))
            rows = _indep_rows(Jz)
            b0 = bf
        try:
            b = _slsqp_epigraph(pb, b0, eq_rows=rows)
        except Exception:
            b = b0
        if not np.all(np.isfinite(b)):
            b = b0
        if pb.Z and pb.eq_resid(b) > 1e-12:
            # restore exact feasibility by Gauss-Newton on all zero constraints
            for _ in range(6):
                v_, J_, _ = pb.pair_vals_jac(b, list(pb.Z))
                b = b - np.linalg.pinv(J_, rcond=1e-12) @ v_
            if not np.all(np.isfinite(b)) or pb.eq_resid(b) > 1e-9:
                b = b0
        b = _polish(pb, b)
        r = pb.eq_resid(b)
        if r > feas_tol:
            continue
        nfeas += 1
        v = pb.r1(b)
        if keep_locals:
            locals_.append((v, b.copy()))
        if best is None or v < best[0] - 1e-13:
            best = (v, b.copy(), r)
    if best is None:
        return None
    v, b, r = best
    B = pb.B_of(b)
    O = omega_of(B, S)
    sup = actual_support(B, O)
    return Sol(v, b, B, O, r, sup, nfeas, locals_)

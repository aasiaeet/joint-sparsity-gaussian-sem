"""Second implementation of Algorithm 1 (eqs. (6)-(7)) for p = 3.

Written from the equations of the paper, independently of ../random_starts.py, and used
to rerun the random-start study (Supplementary Section S18).  Differences in construction:
  * scalar prox: global minimum over an explicit candidate set of the piecewise objective
    (not the closed form), including the coefficient bound as a candidate;
  * B-step: enumeration of all 6 vertex orders (equivalent to the dynamic program for
    p = 3) with the per-entry gains w_ij;
  * Omega-step: per-pair scalar prox with gamma = beta/2 (the Frobenius term counts each
    off-diagonal pair twice, the penalty once), diagonal = z_ii; if the spectrum leaves
    [0.05, 10], a constrained convex solve (cvxpy) using the decomposition of q_delta into
    a convex part and a concave quadratic.
Conventions: X = B^T X + eps, Sigma = (I-B)^{-T} Omega (I-B)^{-1}, B[i,j] != 0 is i -> j.
"""
import itertools
import math

import numpy as np

P = 3
LAM, DELTA = 0.01, 0.1
ALPHA, BETA = 0.01, 0.0042
MB, LO, HI = 2.0, 0.05, 10.0
MAX_CYCLES, STEP_TOL = 20000, 1e-8
MASTER = 20260925
DESIGN_INDEX = {"G1": 0, "G2": 1, "G3": 2}
PAIRS = [(0, 1), (0, 2), (1, 2)]
DIRECTED = [(i, j) for i in range(P) for j in range(P) if i != j]
ORDERS = list(itertools.permutations(range(P)))
FORWARD = {o: [(o[a], o[c]) for a in range(P) for c in range(a + 1, P)] for o in ORDERS}
I3 = np.eye(P)


# ---------------------------------------------------------------- model
def design(name):
    B = np.zeros((P, P))
    O = np.eye(P)
    if name == "G1":
        B[0, 1], B[1, 2] = 0.8, -0.7
    elif name == "G2":
        B[0, 1] = 0.7
        O[1, 2] = O[2, 1] = 0.5
    elif name == "G3":
        B[0, 1], B[1, 2] = 0.7, 0.6
        O[0, 2] = O[2, 0] = 0.4
    return B, O


def sigma(B, O):
    Ai = np.linalg.inv(I3 - B)
    return Ai.T @ O @ Ai


def sample_S(name, n, dataset):
    Sig = sigma(*design(name))
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, DESIGN_INDEX[name], n, dataset]))
    X = rng.standard_normal((n, P)) @ np.linalg.cholesky(Sig).T
    X = X - X.mean(axis=0)
    return X.T @ X / n


def q(t, delta=DELTA):
    a = np.minimum(np.abs(t) / delta, 1.0)
    return a * (2.0 - a)


def penalty(B, O, delta=DELTA):
    return float(sum(q(B[e], delta) for e in DIRECTED) + sum(q(O[e], delta) for e in PAIRS))


def lik(B, O, S):
    A = I3 - B
    R = A.T @ S @ A
    sign, ld = np.linalg.slogdet(O)
    if sign <= 0:
        return np.inf
    return 0.5 * ld + 0.5 * float(np.trace(np.linalg.solve(O, R)))


def grads(B, O, S):
    A = I3 - B
    Oi = np.linalg.inv(O)
    R = A.T @ S @ A
    gB = -S @ A @ Oi
    gO = 0.5 * (Oi - Oi @ R @ Oi)
    return gB, 0.5 * (gO + gO.T)


def objective(B, O, S, lam=LAM, delta=DELTA):
    return lik(B, O, S) + lam * penalty(B, O, delta)


# ---------------------------------------------------------------- proximal maps
def prox_scalar(z, gamma, lam=LAM, delta=DELTA, bound=np.inf):
    """argmin_u (u - z)^2/(2 gamma) + lam q_delta(u), |u| <= bound, by candidate comparison."""
    v = abs(z)
    s = 1.0 if z >= 0 else -1.0
    k = 1.0 / gamma - 2.0 * lam / delta ** 2
    assert k > 0
    cands = [0.0, min(delta, bound), min(max(v, delta), bound)]
    mid = (v / gamma - 2.0 * lam / delta) / k        # stationary point of the (0, delta) piece
    if 0.0 < mid < delta:
        cands.append(min(mid, bound))
    vals = [(u - v) ** 2 / (2 * gamma) + lam * float(q(u, delta)) for u in cands]
    u = cands[int(np.argmin(vals))]
    return s * u, (v * v) / (2 * gamma) - min(vals)   # value, gain relative to u = 0


def b_step(Z, alpha=ALPHA, lam=LAM, delta=DELTA):
    val = np.zeros((P, P))
    gain = np.zeros((P, P))
    for (i, j) in DIRECTED:
        val[i, j], gain[i, j] = prox_scalar(Z[i, j], alpha, lam, delta, MB)
    tot = [sum(gain[e] for e in FORWARD[o]) for o in ORDERS]
    best = int(np.argmax(tot))
    srt = sorted(tot, reverse=True)
    tie = len(srt) > 1 and srt[0] - srt[1] <= 1e-15 * max(1.0, abs(srt[0]))
    U = np.zeros((P, P))
    for e in FORWARD[ORDERS[best]]:
        U[e] = val[e]
    if tie:  # does a tied order give a different matrix?
        for o, t in zip(ORDERS, tot):
            if srt[0] - t <= 1e-15 * max(1.0, abs(srt[0])):
                U2 = np.zeros((P, P))
                for e in FORWARD[o]:
                    U2[e] = val[e]
                if not np.array_equal(U2, U):
                    return U, ORDERS[best], True
    return U, ORDERS[best], False


_CVX = {}


def omega_constrained(Z, beta, lam, delta):
    """Omega-step with spectral constraints, using
    q = 2|t|/delta - t^2/delta^2 + (|t|-delta)_+^2/delta^2; the objective is strongly convex."""
    import cvxpy as cp
    key = (beta, lam, delta)
    if key not in _CVX:
        Zp = cp.Parameter((P, P), symmetric=True)
        V = cp.Variable((P, P), symmetric=True)
        obj = 0
        for i in range(P):
            obj += cp.square(V[i, i] - Zp[i, i]) / (2 * beta)
        for (i, j) in PAIRS:
            t = V[i, j]
            # 2 * (t - z)^2/(2 beta) - lam t^2/delta^2 expanded (constant dropped)
            obj += (1.0 / beta - lam / delta ** 2) * cp.square(t) - 2 * Zp[i, j] * t / beta
            obj += 2 * lam * cp.abs(t) / delta + lam * cp.square(cp.pos(cp.abs(t) - delta)) / delta ** 2
        prob = cp.Problem(cp.Minimize(obj), [V >> LO * np.eye(P), V << HI * np.eye(P)])
        _CVX[key] = (prob, Zp, V)
    import warnings
    from scipy.optimize import minimize
    prob, Zp, V = _CVX[key]
    Zs = 0.5 * (Z + Z.T)
    Zp.value = Zs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
    Vv = 0.5 * (V.value + V.value.T)
    w, U = np.linalg.eigh(Vv)
    V0 = (U * np.clip(w, LO, HI)) @ U.T

    def unpack(x):
        return np.array([[x[0], x[3], x[4]], [x[3], x[1], x[5]], [x[4], x[5], x[2]]])

    def phi(x):
        W = unpack(x)
        return float(np.sum((W - Zs) ** 2)) / (2 * beta) + lam * sum(float(q(W[e], delta)) for e in PAIRS)

    def cons(x):
        ev = np.linalg.eigvalsh(unpack(x))
        return np.r_[ev - LO, HI - ev]
    x0 = np.r_[np.diag(V0), V0[0, 1], V0[0, 2], V0[1, 2]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(phi, x0, method="SLSQP", constraints=[dict(type="ineq", fun=cons)],
                     options=dict(ftol=1e-16, maxiter=500))
    if np.min(cons(r.x)) >= 0 and r.fun < phi(x0):
        return unpack(r.x)
    return V0


def omega_step(Z, beta=BETA, lam=LAM, delta=DELTA, stats=None):
    V = np.diag(np.diag(Z)).astype(float)
    for (i, j) in PAIRS:
        zij = 0.5 * (Z[i, j] + Z[j, i])
        V[i, j] = V[j, i] = prox_scalar(zij, beta / 2.0, lam, delta)[0]
    w = np.linalg.eigvalsh(V)
    if w[0] >= LO and w[-1] <= HI:
        return V
    if stats is not None:
        stats["conic"] = stats.get("conic", 0) + 1
    return omega_constrained(Z, beta, lam, delta)


# ---------------------------------------------------------------- algorithm
def run(S, B0, O0, alpha=ALPHA, beta=BETA, lam=LAM, delta=DELTA, max_cycles=MAX_CYCLES, tol=STEP_TOL):
    B, O = B0.copy(), O0.copy()
    stats = {"conic": 0, "order_ties": 0}
    f0 = objective(B, O, S, lam, delta)
    fprev = f0
    max_inc = -np.inf
    gB, _ = grads(B, O, S)
    status, t = "cycle_cap", 0
    for t in range(1, max_cycles + 1):
        Bn, order, tie = b_step(B - alpha * gB, alpha, lam, delta)
        stats["order_ties"] += int(tie)
        _, gO = grads(Bn, O, S)
        On = omega_step(O - beta * gO, beta, lam, delta, stats)
        step = math.sqrt(float(np.sum((Bn - B) ** 2) + np.sum((On - O) ** 2)))
        B, O = Bn, On
        gB, _ = grads(B, O, S)
        if t % 250 == 0 or step < tol:
            f = objective(B, O, S, lam, delta)
            max_inc = max(max_inc, f - fprev)
            fprev = f
        if step < tol:
            status = "step_tolerance"
            break
    return B, O, dict(objective=objective(B, O, S, lam, delta), start_objective=f0, cycles=t, status=status,
                      final_step=step, conic=stats["conic"], order_ties=stats["order_ties"],
                      max_increase_sampled=max_inc)


# ---------------------------------------------------------------- starts
def random_start(S, name, n, dataset, k, B_sd=0.5, G_sd=0.5):
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, DESIGN_INDEX[name], n, dataset, 2, k]))
    order = rng.permutation(P)
    B = np.zeros((P, P))
    for a in range(P):
        for c in range(a + 1, P):
            B[order[a], order[c]] = rng.normal(0.0, B_sd)
    B = np.clip(B, -MB, MB)
    G = np.zeros((P, P))
    for (i, j) in PAIRS:
        G[i, j] = G[j, i] = rng.normal(0.0, G_sd)
    d = np.sqrt(np.diag(S))
    t = 1.0
    while t >= 1e-6:
        C = I3 + t * G
        O = np.outer(d, d) * C
        w = np.linalg.eigvalsh(O)
        if np.linalg.eigvalsh(C)[0] >= 0.5 and w[0] >= LO and w[-1] <= HI:
            return B, O, dict(start_order="".join(str(v + 1) for v in order), start_omega_scale=t)
        t *= 0.8
    return B, np.diag(np.diag(S)), dict(start_order="".join(str(v + 1) for v in order), start_omega_scale=0.0)


# ---------------------------------------------------------------- supports
def support_name(B, O, cut):
    D = [e for e in DIRECTED if abs(B[e]) > 0 and abs(B[e]) >= cut]
    U = [e for e in PAIRS if abs(O[e]) > 0 and abs(O[e]) >= cut]
    parts = [f"{i+1}->{j+1}" for i, j in D] + [f"{i+1}<->{j+1}" for i, j in U]
    return ",".join(parts) if parts else "empty"


def parse_support(name):
    D, U = [], []
    if name == "empty":
        return D, U
    for e in name.split(","):
        if "<->" in e:
            a, b = e.split("<->")
            U.append((int(a) - 1, int(b) - 1))
        else:
            a, b = e.split("->")
            D.append((int(a) - 1, int(b) - 1))
    return D, U


def acyclic(D):
    return any(all(o.index(i) < o.index(j) for i, j in D) for o in ORDERS)


def all_supports():
    out = []
    for m in range(1 << 6):
        D = [e for k, e in enumerate(DIRECTED) if m >> k & 1]
        if not acyclic(D):
            continue
        for u in range(1 << 3):
            U = [e for k, e in enumerate(PAIRS) if u >> k & 1]
            out.append((D, U))
    return out


def name_of(D, U):
    D = sorted(D, key=DIRECTED.index)
    U = sorted(U, key=PAIRS.index)
    parts = [f"{i+1}->{j+1}" for i, j in D] + [f"{i+1}<->{j+1}" for i, j in U]
    return ",".join(parts) if parts else "empty"

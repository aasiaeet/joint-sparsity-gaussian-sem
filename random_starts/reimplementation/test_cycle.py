"""Checks of cycle.py against brute force: scalar prox, B-step, Omega-step, gradients."""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import cycle as m

OUT = Path(__file__).resolve().parent / "results"

rng = np.random.default_rng(1)
out = []

# 1. scalar prox vs fine grid
worst = 0.0
for gamma in (m.ALPHA, m.BETA / 2, 0.1, 0.4):
    for z in np.r_[rng.uniform(-3, 3, 400), [0.0, 0.1, -0.1, 2.0, 2.5, -2.5, 1e-3]]:
        for bound in (np.inf, 2.0):
            u, gain = m.prox_scalar(z, gamma, bound=bound)
            lo = -bound if np.isfinite(bound) else -4
            hi = bound if np.isfinite(bound) else 4
            grid = np.linspace(lo, hi, 400001)
            f = (grid - z) ** 2 / (2 * gamma) + m.LAM * m.q(grid)
            fu = (u - z) ** 2 / (2 * gamma) + m.LAM * float(m.q(u))
            worst = max(worst, fu - f.min())
out.append(f"scalar prox: max excess over 4e5-point grid = {worst:.2e}")

# 2. B-step vs enumeration of all 25 DAG supports (separable optimum on each)
sups = sorted({tuple(D) for D, U in m.all_supports()})
assert len(sups) == 25
worst = 0.0
for _ in range(3000):
    Z = rng.normal(scale=rng.choice([0.005, 0.05, 0.5, 2.0]), size=(3, 3))
    np.fill_diagonal(Z, 0)
    U, order, tie = m.b_step(Z)

    def F(Um):
        return float(np.sum((Um - Z)[~np.eye(3, dtype=bool)] ** 2) / (2 * m.ALPHA)
                     + m.LAM * sum(m.q(Um[e]) for e in m.DIRECTED))
    best = np.inf
    for D in sups:
        Um = np.zeros((3, 3))
        for e in D:
            Um[e] = m.prox_scalar(Z[e], m.ALPHA, bound=m.MB)[0]
        best = min(best, F(Um))
    worst = max(worst, F(U) - best)
out.append(f"B-step: max excess over 25-DAG enumeration = {worst:.2e}")

# 3. Omega-step vs direct minimization (interior and constrained cases)
worst = 0.0
ncon = 0
for trial in range(300):
    A = rng.normal(size=(3, 3))
    Z = A @ A.T / 3 + (0.02 if trial % 3 == 0 else 0.5) * np.eye(3)
    if trial % 5 == 0:
        Z = Z * 6  # push towards the ceiling
    st = {}
    V = m.omega_step(Z, stats=st)
    ncon += st.get("conic", 0)

    def phi(x):
        W = np.array([[x[0], x[3], x[4]], [x[3], x[1], x[5]], [x[4], x[5], x[2]]])
        return (float(np.sum((W - Z) ** 2)) / (2 * m.BETA)
                + m.LAM * sum(float(m.q(W[e])) for e in m.PAIRS)), W

    def cons(x):
        w = np.linalg.eigvalsh(phi(x)[1])
        return np.r_[w - m.LO, m.HI - w]
    best = np.inf
    for s in range(6):
        x0 = np.r_[np.clip(np.diag(Z), 1, 5), 0, 0, 0] + (rng.normal(scale=0.05, size=6) if s else 0)
        r = minimize(lambda x: phi(x)[0], x0, method="SLSQP", constraints=[dict(type="ineq", fun=cons)],
                     options=dict(ftol=1e-15, maxiter=2000))
        if np.min(cons(r.x)) > -1e-9:
            best = min(best, r.fun)
    x = np.r_[np.diag(V), V[0, 1], V[0, 2], V[1, 2]]
    worst = max(worst, phi(x)[0] - best)
out.append(f"Omega-step: max excess over SLSQP multistart = {worst:.2e} (constrained cases: {ncon} of 300)")

# 4. gradients vs finite differences
worst = 0.0
for _ in range(50):
    S = np.cov(rng.normal(size=(3, 50)))
    B = np.zeros((3, 3))
    B[0, 1], B[0, 2], B[1, 2] = rng.normal(size=3)
    A = rng.normal(size=(3, 3))
    O = A @ A.T + np.eye(3)
    gB, gO = m.grads(B, O, S)
    h = 1e-6
    for (i, j) in m.DIRECTED:
        E = np.zeros((3, 3)); E[i, j] = h
        fd = (m.lik(B + E, O, S) - m.lik(B - E, O, S)) / (2 * h)
        worst = max(worst, abs(fd - gB[i, j]))
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3)); E[i, j] = E[j, i] = h
            fd = (m.lik(B, O + E, S) - m.lik(B, O - E, S)) / (2 * h)
            an = gO[i, j] * (1 if i == j else 2)
            worst = max(worst, abs(fd - an))
out.append(f"gradients: max |analytic - finite difference| = {worst:.2e}")
print("\n".join(out))
OUT.mkdir(exist_ok=True)
(OUT / "test_cycle.txt").write_text("\n".join(out) + "\n")

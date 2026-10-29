"""Constrained Gaussian likelihood fits on a mixed support (D, U) within K, p = 3.

Joint SLSQP over (B_D, diag Omega, Omega_U) with |B| <= 2 and eigenvalue constraints
spec(Omega) in [0.05, 10]; analytic gradients; multistart.  Optional: extra smooth terms
(penalty on selected coordinates, fixed coordinates) for path/profile computations.
"""
import warnings

import numpy as np
from scipy.optimize import minimize

import cycle as m

P = 3


class Param:
    def __init__(self, D, U, fixed=None):
        """fixed: dict coord -> value, coord = ('B', i, j) or ('O', i, j); held fixed."""
        self.D, self.U = list(D), list(U)
        self.fixed = dict(fixed or {})
        self.freeD = [e for e in self.D if ("B",) + e not in self.fixed]
        self.freeU = [e for e in self.U if ("O",) + e not in self.fixed]
        self.nx = len(self.freeD) + 3 + len(self.freeU)

    def unpack(self, x):
        B = np.zeros((P, P))
        O = np.zeros((P, P))
        k = 0
        for e in self.freeD:
            B[e] = x[k]; k += 1
        for i in range(P):
            O[i, i] = x[k]; k += 1
        for (i, j) in self.freeU:
            O[i, j] = O[j, i] = x[k]; k += 1
        for c, v in self.fixed.items():
            if c[0] == "B":
                B[c[1], c[2]] = v
            else:
                O[c[1], c[2]] = O[c[2], c[1]] = v
        return B, O

    def pack(self, B, O):
        return np.r_[[B[e] for e in self.freeD], np.diag(O), [O[e] for e in self.freeU]]

    def grad_vec(self, gB, gO):
        return np.r_[[gB[e] for e in self.freeD], np.diag(gO), [2 * gO[e] for e in self.freeU]]

    def bounds(self):
        return [(-m.MB, m.MB)] * len(self.freeD) + [(m.LO, m.HI)] * 3 + [(-m.HI, m.HI)] * len(self.freeU)


def _feasible_start(par, S, B):
    A = np.eye(P) - B
    R = A.T @ S @ A
    O = np.diag(np.clip(np.diag(R), m.LO + 1e-3, m.HI - 1e-3))
    for sh in (1.0, 0.5, 0.25, 0.0):
        O2 = O.copy()
        for (i, j) in par.freeU:
            O2[i, j] = O2[j, i] = sh * R[i, j]
        w = np.linalg.eigvalsh(O2)
        if w[0] > m.LO and w[-1] < m.HI:
            return par.pack(B, O2)
    return par.pack(B, O)


def fit(S, D, U, rng=None, warm=(), n_random=4, fixed=None, pen_coords=(), lam=m.LAM, delta=m.DELTA,
        include_default=True):
    """Minimize l_S (+ lam * sum q_delta over pen_coords) over the support within K.
    pen_coords: list of ('B', i, j) / ('O', i, j) (free coords) whose quasi-MCP enters the objective."""
    par = Param(D, U, fixed)
    rng = rng or np.random.default_rng(0)

    def fun(x):
        B, O = par.unpack(x)
        w = np.linalg.eigvalsh(O)
        if w[0] <= 1e-9:
            return 1e10, np.zeros_like(x)
        v = m.lik(B, O, S)
        gB, gO = m.grads(B, O, S)
        g = par.grad_vec(gB, gO)
        if pen_coords:
            for c in pen_coords:
                t = B[c[1], c[2]] if c[0] == "B" else O[c[1], c[2]]
                a = abs(t) / delta
                if a < 1:
                    v += lam * (2 * a - a * a)
                    dq = lam * (2 / delta - 2 * abs(t) / delta ** 2) * np.sign(t)
                else:
                    v += lam
                    dq = 0.0
                idx = (par.freeD.index((c[1], c[2])) if c[0] == "B"
                       else len(par.freeD) + 3 + par.freeU.index((c[1], c[2])))
                g[idx] += dq
        return v, g

    def cons(x):
        w = np.linalg.eigvalsh(par.unpack(x)[1])
        return np.r_[w - m.LO, m.HI - w]

    starts = [np.asarray(w, float) for w in warm]
    if include_default:
        starts.append(_feasible_start(par, S, np.zeros((P, P))))
    for _ in range(n_random):
        B = np.zeros((P, P))
        for e in par.freeD:
            B[e] = rng.uniform(-1.5, 1.5)
        for c, v in par.fixed.items():
            if c[0] == "B":
                B[c[1], c[2]] = v
        starts.append(_feasible_start(par, S, B))
    best = (np.inf, None)
    for x0 in starts:
        x0 = np.clip(x0, [b[0] for b in par.bounds()], [b[1] for b in par.bounds()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(fun, x0, jac=True, method="SLSQP", bounds=par.bounds(),
                         constraints=[dict(type="ineq", fun=cons)], options=dict(ftol=1e-15, maxiter=3000))
        if np.min(cons(r.x)) < -1e-10:
            continue
        v = fun(r.x)[0]
        if v < best[0]:
            best = (v, r.x)
    if best[1] is None:
        return np.inf, None, None, par
    B, O = par.unpack(best[1])
    return best[0], B, O, par

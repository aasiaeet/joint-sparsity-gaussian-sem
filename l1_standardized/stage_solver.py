"""Solver S2: sequential stages over a vertex order with an exact last stage.

Shares no code with solver S1 in l1core.py.  Fix a vertex order pi compatible with D.
Processing the vertices in this order, the column w_v = e_v - B[:, v] of vertex v enters
Omega only through Omega_iv = w_i^T Sigma w_v with earlier i, and these are affine in the
coefficients beta_v of v given the earlier columns.  So, given the earlier columns:
  * the zero constraints Omega_iv = 0 (pairs {i, v} not in U) are linear equations
    A beta_v = c, whose rank rho can be smaller than the number of rows (the extra rows
    must then be consistent, which happens when Sigma satisfies the model's constraints);
  * the cost of vertex v, ||beta_v||_1 + sum_{i earlier, {i,v} in U} |Omega_iv|, is a sum of
    absolute values of affine functions of beta_v.
Outer variables: for each of the vertices 2..p-1 of the order we solve the equations for a
pivot subset of rho coefficients and keep the other coefficients free.  Every point of the
stage's affine solution set corresponds to exactly one value of the free coefficients when
the pivot minor is nonsingular, and we take the union over all pivot subsets, so the free
coefficients scanned on a grid over [-R, R]^d cover every representation with R_1 <= R
(|B_ij| <= R_1).  The best grid points are refined by bounded scalar search (d = 1) or
Nelder-Mead (d > 1).  The last vertex is solved exactly: on the affine solution set its
cost is convex piecewise linear, so its minimum is attained at a vertex of the hyperplane
arrangement, and we enumerate all vertices.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _pk(i, j):
    return (min(i, j), max(i, j))


def _num_rank(A, rel=1e-9):
    """Numerical rank of each matrix in a stack (N, r, m)."""
    if A.shape[1] == 0 or A.shape[2] == 0:
        return np.zeros(A.shape[0], int)
    s = np.linalg.svd(A, compute_uv=False)
    scale = np.maximum(1.0, np.abs(A).max(axis=(1, 2)))
    return (s > rel * scale[:, None]).sum(axis=1)


def _lsq(A, c, rho=None, rel=1e-9):
    """Least-norm least-squares solutions of a stack of systems, with a truncated SVD.

    Singular values at or below rel * max(1, max|A|) are treated as zero (structural zeros of
    Sigma are only zero up to rounding, and inverting them would manufacture huge, spurious
    solutions); if rho is given, at most rho singular values are kept."""
    N, r, m = A.shape
    if m == 0 or r == 0:
        return np.zeros((N, m))
    U_, s, Vt = np.linalg.svd(A, full_matrices=False)
    scale = np.maximum(1.0, np.abs(A).max(axis=(1, 2)))
    keep = s > rel * scale[:, None]
    if rho is not None:
        keep[:, rho:] = False
    sinv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    return np.einsum("nkm,nk,nrk,nr->nm", Vt, sinv, U_, c)


class StagePlan:
    def __init__(self, S, D, U, order, tol=1e-9, seed=0):
        self.S = np.asarray(S, float)
        self.p = len(S)
        self.order = tuple(order)
        self.tol = tol
        Uset = set(U)
        pos = {v: k for k, v in enumerate(order)}
        for (i, j) in D:
            if pos[i] >= pos[j]:
                raise ValueError("order not compatible with D")
        self.stages = []
        for k, v in enumerate(order):
            P = sorted([i for (i, j) in D if j == v], key=lambda x: pos[x])
            E = list(order[:k])
            Req = [i for i in E if _pk(i, v) not in Uset]
            Rpen = [i for i in E if _pk(i, v) in Uset]
            self.stages.append(dict(v=v, P=P, Req=Req, Rpen=Rpen))
        self.outer_idx = list(range(1, self.p - 1))
        self._discover_ranks(seed)
        opts = []
        for k in self.outer_idx:
            st = self.stages[k]
            rho, m = st["rho"], len(st["P"])
            opts.append(list(itertools.combinations(range(m), rho)))
        self.branches = []
        for combo in itertools.product(*opts):
            d = sum(len(self.stages[k]["P"]) - self.stages[k]["rho"] for k in self.outer_idx)
            self.branches.append((combo, d))

    def _stage_mats(self, W, st):
        SW = np.einsum("ab,nbc->nac", self.S, W)
        N = W.shape[0]
        P, Req, Rpen, v = st["P"], st["Req"], st["Rpen"], st["v"]
        m = len(P)
        A_eq = SW[:, P][:, :, Req].transpose(0, 2, 1) if m else np.zeros((N, len(Req), 0))
        c_eq = SW[:, v, Req] if Req else np.zeros((N, 0))
        A_pen = SW[:, P][:, :, Rpen].transpose(0, 2, 1) if m else np.zeros((N, len(Rpen), 0))
        c_pen = SW[:, v, Rpen] if Rpen else np.zeros((N, 0))
        return A_eq, c_eq, A_pen, c_pen

    def _discover_ranks(self, seed, K=8):
        """Generic rank of every stage's equation system, from random consistent points."""
        rng = np.random.default_rng(seed)
        W = np.tile(np.eye(self.p), (K, 1, 1))
        self.param_level = False
        alive = np.ones(K, bool)
        for k, st in enumerate(self.stages):
            A_eq, c_eq, _, _ = self._stage_mats(W, st)
            m = len(st["P"])
            rho = int(_num_rank(A_eq[alive]).max()) if alive.any() and A_eq.shape[1] and m else 0
            st["rho"] = rho
            y0 = _lsq(A_eq, c_eq)
            res = np.abs(np.einsum("nrm,nm->nr", A_eq, y0) - c_eq).max(axis=1) if A_eq.shape[1] else np.zeros(K)
            cons = res < self.tol
            if not cons[alive].all():
                self.param_level = True  # consistency depends on earlier free coefficients
            alive &= cons
            if m:
                # random point of the solution set
                _, s, Vt = np.linalg.svd(A_eq if A_eq.shape[1] else np.zeros((K, 1, m)), full_matrices=True)
                Nb = Vt[:, rho:, :].transpose(0, 2, 1)
                z = rng.uniform(-1, 1, (K, m - rho))
                beta = y0 + np.einsum("nmd,nd->nm", Nb, z)
                W[:, st["P"], st["v"]] -= beta
        self.alive_fraction = float(alive.mean())

    # ------------------------------------------------------------------
    def _solve_stage(self, A_eq, c_eq, piv, zF, rho):
        """Coefficients of an outer stage from its free coordinates (pivot parametrization)."""
        N, r, m = A_eq.shape
        beta = np.zeros((N, m))
        ok = np.ones(N, bool)
        Fi = [x for x in range(m) if x not in piv]
        if Fi:
            beta[:, Fi] = zF
        if rho == 0:
            if r:
                ok &= np.abs(c_eq).max(axis=1) < self.tol
            return beta, ok
        rhs_full = c_eq - (np.einsum("nrf,nf->nr", A_eq[:, :, Fi], zF) if Fi else 0.0)
        Ap_full = A_eq[:, :, list(piv)]  # (N, r, rho)
        best_det = np.zeros(N)
        sol = np.zeros((N, rho))
        for Rs in itertools.combinations(range(r), rho):
            Ap = Ap_full[:, list(Rs), :]
            det = np.abs(np.linalg.det(Ap))
            upd = det > best_det
            if not upd.any():
                continue
            Ap2 = Ap.copy(); Ap2[~upd] = np.eye(rho)
            s = np.linalg.solve(Ap2, rhs_full[:, list(Rs)][..., None])[..., 0]
            sol[upd] = s[upd]
            best_det = np.where(upd, det, best_det)
        beta[:, list(piv)] = sol
        ok &= best_det > 1e-10
        res = np.abs(np.einsum("nrm,nm->nr", A_eq, beta) - c_eq).max(axis=1)
        ok &= res < self.tol * np.maximum(1.0, np.abs(beta).max(axis=1))
        return beta, ok

    def evaluate(self, Z, combo, return_B=False):
        """Batched objective over outer points Z (N x d) for one pivot branch."""
        p = self.p
        Z = np.atleast_2d(Z)
        N = Z.shape[0]
        W = np.tile(np.eye(p), (N, 1, 1))
        cost = np.zeros(N)
        ok = np.ones(N, bool)
        col = 0
        for k, piv in zip(self.outer_idx, combo):
            st = self.stages[k]
            A_eq, c_eq, A_pen, c_pen = self._stage_mats(W, st)
            m, rho = len(st["P"]), st["rho"]
            nf = m - rho
            zF = Z[:, col:col + nf]; col += nf
            beta, okk = self._solve_stage(A_eq, c_eq, piv, zF, rho)
            ok &= okk
            pen = np.abs(c_pen - np.einsum("nrm,nm->nr", A_pen, beta)).sum(axis=1) if st["Rpen"] else 0.0
            cost = cost + np.abs(beta).sum(axis=1) + pen
            if m:
                W[:, st["P"], st["v"]] -= beta
        inner, y, okl = self._last(W, return_B)
        val = np.where(ok & okl, cost + inner, np.inf)
        if return_B:
            st = self.stages[-1]
            if st["P"]:
                W[:, st["P"], st["v"]] -= y
            return val, np.eye(p)[None] - W
        return val

    def _last(self, W, return_y=False):
        st = self.stages[-1]
        A_eq, c_eq, A_pen, c_pen = self._stage_mats(W, st)
        N, r, m = A_eq.shape
        rho = st["rho"]
        y0 = _lsq(A_eq, c_eq, rho=rho)
        ok = np.ones(N, bool)
        if r:
            res = np.abs(np.einsum("nrm,nm->nr", A_eq, y0) - c_eq).max(axis=1)
            ok &= res < self.tol * np.maximum(1.0, np.abs(y0).max(axis=1) if m else 1.0)
        d = m - rho
        if m == 0:
            val = np.abs(c_pen).sum(axis=1) if st["Rpen"] else np.zeros(N)
            return val, np.zeros((N, 0)), ok
        if rho:
            _, s, Vt = np.linalg.svd(A_eq, full_matrices=True)
            Nb = Vt[:, rho:, :].transpose(0, 2, 1)  # (N, m, d)
            ok &= _num_rank(A_eq) == rho
        else:
            Nb = np.tile(np.eye(m), (N, 1, 1))
        # terms |G z + h|: coefficients y = y0 + Nb z, penalties c_pen - A_pen y
        G = np.concatenate([Nb, -np.einsum("nrm,nmd->nrd", A_pen, Nb)], axis=1)
        h = np.concatenate([y0, c_pen - np.einsum("nrm,nm->nr", A_pen, y0)], axis=1)
        nt = G.shape[1]
        if d == 0:
            return np.abs(h).sum(axis=1), y0, ok
        best = np.full(N, np.inf)
        bestz = np.zeros((N, d))
        for sub in itertools.combinations(range(nt), d):
            Gs = G[:, list(sub), :]
            dt = np.abs(np.linalg.det(Gs))
            sing = dt < 1e-12
            Gs = Gs.copy(); Gs[sing] = np.eye(d)
            z = np.linalg.solve(Gs, -h[:, list(sub)][..., None])[..., 0]
            val = np.abs(np.einsum("ntd,nd->nt", G, z) + h).sum(axis=1)
            val = np.where(sing, np.inf, val)
            upd = val < best
            if upd.any():
                best = np.where(upd, val, best)
                bestz[upd] = z[upd]
        y = y0 + np.einsum("nmd,nd->nm", Nb, bestz)
        return best, y, ok

    # ------------------------------------------------------------------
    def solve(self, R, n_grid, n_refine=8, chunk=100000):
        """Grid over [-R, R]^d for every pivot branch, then local refinement."""
        best_val, best_B = np.inf, None
        diag = dict(branches=len(self.branches), grid_points=0, feasible_points=0,
                    param_level=self.param_level)
        for combo, d in self.branches:
            if d == 0:
                val, B = self.evaluate(np.zeros((1, 0)), combo, return_B=True)
                diag["grid_points"] += 1
                diag["feasible_points"] += int(np.isfinite(val[0]))
                if val[0] < best_val:
                    best_val, best_B = float(val[0]), B[0]
                continue
            ng = n_grid.get(d, min(n_grid.values())) if isinstance(n_grid, dict) else n_grid
            g = np.linspace(-R, R, ng)
            mesh = np.array(list(itertools.product(g, repeat=d))) if d > 1 else g[:, None]
            vals = np.concatenate([self.evaluate(mesh[s:s + chunk], combo) for s in range(0, len(mesh), chunk)])
            diag["grid_points"] += len(mesh)
            fin = np.isfinite(vals)
            diag["feasible_points"] += int(fin.sum())
            if not fin.any():
                continue
            h = 2 * R / (ng - 1)
            order = np.argsort(vals)
            picked = []
            for idx in order:
                if not np.isfinite(vals[idx]):
                    break
                x = mesh[idx]
                if any(np.max(np.abs(x - q)) < 2.5 * h for q in picked):
                    continue
                picked.append(x)
                if len(picked) >= n_refine:
                    break
            for x0 in picked:
                f = lambda z: float(self.evaluate(np.atleast_2d(z), combo)[0])
                if d == 1:
                    res = minimize_scalar(f, bounds=(x0[0] - h, x0[0] + h), method="bounded",
                                          options=dict(xatol=1e-14, maxiter=1000))
                    xs = np.array([res.x])
                    for w in (1e-3 * h, 1e-6 * h):
                        r2 = minimize_scalar(f, bounds=(xs[0] - w, xs[0] + w), method="bounded",
                                             options=dict(xatol=1e-16, maxiter=1000))
                        if r2.fun <= f(xs):
                            xs = np.array([r2.x])
                else:
                    simplex = np.vstack([x0] + [x0 + 0.5 * h * e for e in np.eye(d)])
                    res = minimize(f, x0, method="Nelder-Mead",
                                   options=dict(initial_simplex=simplex, xatol=1e-12, fatol=1e-15,
                                                maxiter=4000 * d, maxfev=8000 * d))
                    xs = res.x
                    for _ in range(4):  # restarts with smaller simplices
                        simplex = np.vstack([xs] + [xs + 1e-3 * h * e for e in np.eye(d)])
                        r2 = minimize(f, xs, method="Nelder-Mead",
                                      options=dict(initial_simplex=simplex, xatol=1e-14, fatol=1e-16,
                                                   maxiter=4000 * d, maxfev=8000 * d))
                        if r2.fun < f(xs) - 1e-16:
                            xs = r2.x
                        else:
                            break
                val, B = self.evaluate(np.atleast_2d(xs), combo, return_B=True)
                if val[0] < best_val:
                    best_val, best_B = float(val[0]), B[0]
            val, B = self.evaluate(np.atleast_2d(mesh[order[0]]), combo, return_B=True)
            if val[0] < best_val:
                best_val, best_B = float(val[0]), B[0]
        return best_val, best_B, diag


def compatible_orders(D, p):
    out = []
    for perm in itertools.permutations(range(p)):
        pos = {v: k for k, v in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in D):
            out.append(perm)
    return out


def s2_support(S, D, U, R, n_grid, n_refine=8, max_orders=None):
    """Best value over the orders compatible with D (each order is a separate parametrization)."""
    p = len(S)
    best = (np.inf, None, None)
    diags = []
    orders = compatible_orders(D, p)
    if max_orders is not None:
        orders = orders[:max_orders]
    for order in orders:
        plan = StagePlan(S, D, U, order)
        v, B, dg = plan.solve(R, n_grid, n_refine)
        dg["order"] = order
        diags.append(dg)
        if v < best[0]:
            best = (v, B, order)
    return best[0], best[1], best[2], diags

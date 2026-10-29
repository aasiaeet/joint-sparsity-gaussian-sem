"""Population fits l_min(H; Sigma*) for all 34,752 acyclic mixed supports on four vertices.

Usage
    python3 fit_supports.py timing [--workers W]  # 200 random supports, no warm starts, run-time projection
    python3 fit_supports.py full [--workers W]    # all supports, level by level (warm starts from sub-supports)

Domain: B acyclic on D with |B_ij| <= M_B = 1000, Omega positive definite with
zeros off U (no bound on Omega).  The bound separates attained exact fits from
limits at infinity.  The minimum-edge exact fits have |B| < 1; exact fits of bowed
supersets whose fibers are unbounded can sit anywhere up to the bound.

Per support H = (D, U), in this order:

1. Residual fit (exactness).  Phi(B, Omega) = Sigma* for some Omega supported on U
   iff the error correlations rho_ij(B) = R_ij / sqrt(R_ii R_jj),
   R = (I-B)^T Sigma* (I-B), vanish for every pair {i,j} not in U (Omega = R is then
   positive definite automatically).  Bounded nonlinear least squares (scipy
   least_squares, trf, analytic Jacobian) over B_D from B = 0, the regression fit,
   30 draws in [-2, 2] and 10 draws in [-5, 5]; early stop at max|rho| < 1e-12
   (supports with at most 5 edges run all 42 starts and the distinct exact
   solutions are counted).  Exact: min over starts of max|rho| < 1e-9.
   When the best point has |B| at the bound and max|rho| < 1e-2, the fit is
   repeated with bounds 1e4 and 1e5 (diagnostic): a residual that keeps falling
   as the bound grows means the error correlations vanish only as |B| -> infinity.
   This does not make the support near-exact: the correlation matrix of R
   degenerates along such paths and the KL gap stays bounded away from 0.

2. KL fit.  gap(H) = l_min(H; Sigma*) - l_sat, l_sat = 0.5 log det Sigma* + 2, so
   gap = min KL(N(0, Sigma*) || N(0, Phi(theta))) >= 0.
   - U empty: exact regression MLE (closed form).
   - D empty, bidirected components complete: closed form (blocks of Sigma*).
   - bidirected components complete: Omega profiled in closed form
     (Omega = block-diagonal part of R); L-BFGS-B over B_D with the envelope gradient.
   - otherwise (a bidirected component that is not complete): joint L-BFGS-B over
     (B_D, diag Omega, Omega_U) with a positive-definiteness barrier.
   Starts: optima of all immediate sub-supports (full mode), B = 0, the
   regression fit on D, 10 uniform draws in [-2, 2]^|D|.  The best value over
   these is gap_independent (it never sees the residual fit).  One more start
   from the residual fit's best B gives the reported gap.  Stop early once a
   start reaches gap < 1e-12 (the global minimum is 0).  Exact fit: gap < 1e-10.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import multiprocessing as mpc
import time
import warnings

import numpy as np
from scipy.optimize import least_squares, minimize

from common import (DIRECTED, GSTAR, MASTER_SEED, P, PAIRS, RESULTS, SUPPORTS, STUDY_ID,
                    covariance, immediate_subsupports, loss, saturated_value)

CONFIG = dict(
    study="population fits for the Verma graph, all acyclic mixed supports on 4 vertices",
    master_seed=MASTER_SEED,
    parameter_seed_rule="default_rng(SeedSequence([20261022, 6, 0, 1])), values rounded to 4 decimals",
    kl_seed_rule="default_rng(SeedSequence([20261022, 6, support_index, 1]))",
    resid_seed_rule="default_rng(SeedSequence([20261022, 6, support_index, 2]))",
    class_seed_rule="default_rng(SeedSequence([20261022, 6, support_index, 3])) over GF(2^61-1)",
    timing_selection_seed_rule="default_rng(SeedSequence([20261022, 6, 0, 4])), 200 supports without replacement",
    domain="|B_ij| <= 1000 on the support, Omega positive definite (unbounded); the domain K of the "
           "global-selection study (|B| <= 2, spec Omega in [0.05, 10]) is reported, not imposed",
    coefficient_bound=1000.0,
    rho_at_infinity_diagnostic=dict(trigger="best residual point at the bound and max|rho| < 1e-2",
                          bounds=[1e4, 1e5], starts=10, limit_if_ratio_below=0.1),
    kl=dict(random_starts=10, start_box=[-2.0, 2.0], lbfgs_ftol=1e-15, lbfgs_gtol=1e-11,
            lbfgs_maxiter=3000, maxcor=20, exact_stop=1e-12, exact_tol=1e-10),
    resid=dict(random_starts_narrow=30, narrow_box=[-2.0, 2.0], random_starts_wide=10, wide_box=[-5.0, 5.0],
               xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000, exact_stop=1e-12, exact_tol=1e-9,
               all_starts_up_to_edges=5, solution_cluster_tol=1e-6),
    projection_limit_hours=12.0,
)


# ----------------------------------------------------------------------------
# Generating parameter
# ----------------------------------------------------------------------------

def generating_parameter():
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, 0, 1]))
    sgn = lambda: float(rng.choice([-1.0, 1.0]))
    b12 = round(sgn() * rng.uniform(0.5, 1.0), 4)
    b23 = round(sgn() * rng.uniform(0.5, 1.0), 4)
    b34 = round(sgn() * rng.uniform(0.5, 1.0), 4)
    w = [round(float(x), 4) for x in rng.uniform(0.7, 1.3, P)]
    rho = sgn() * rng.uniform(0.3, 0.6)
    w24 = round(rho * np.sqrt(w[1] * w[3]), 4)
    B = np.zeros((P, P))
    B[0, 1], B[1, 2], B[2, 3] = b12, b23, b34
    O = np.diag(w)
    O[1, 3] = O[3, 1] = w24
    return B, O


B_STAR, O_STAR = generating_parameter()
SIGMA = covariance(B_STAR, O_STAR)
L_SAT = saturated_value(SIGMA)


# ----------------------------------------------------------------------------
# KL fit
# ----------------------------------------------------------------------------

def _embed_B(x, D):
    B = np.zeros((P, P))
    for v, e in zip(x, D):
        B[e] = v
    return B


def _block_mask(s):
    M = np.zeros((P, P), bool)
    for c in s.components():
        for i in c:
            for j in c:
                M[i, j] = True
    return M


def _profile_vg(x, D, mask):
    B = _embed_B(x, D)
    A = np.eye(P) - B
    SA = SIGMA @ A
    R = A.T @ SA
    O = np.where(mask, R, 0.0)
    sign, ld = np.linalg.slogdet(O)
    if sign <= 0:
        return 1e10, np.zeros(len(x))
    K = np.linalg.inv(O)
    val = 0.5 * (ld + np.trace(K @ R))
    gB = -SA @ K
    return float(val), np.array([gB[e] for e in D])


def _unpack_joint(z, D, U):
    nd = len(D)
    B = _embed_B(z[:nd], D)
    O = np.diag(z[nd:nd + P]).astype(float)
    for v, (i, j) in zip(z[nd + P:], U):
        O[i, j] = O[j, i] = v
    return B, O


def _joint_vg(z, D, U):
    B, O = _unpack_joint(z, D, U)
    try:
        L = np.linalg.cholesky(O)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(z)
    ld = 2.0 * np.sum(np.log(np.diag(L)))
    K = np.linalg.inv(O)
    A = np.eye(P) - B
    SA = SIGMA @ A
    R = A.T @ SA
    KR = K @ R
    val = 0.5 * (ld + np.trace(KR))
    gB = -SA @ K
    gO = 0.5 * (K - KR @ K)
    gO = (gO + gO.T) / 2
    g = np.r_[[gB[e] for e in D], np.diag(gO), [2 * gO[i, j] for i, j in U]]
    return float(val), g


def _omega_start(B, U):
    A = np.eye(P) - B
    R = A.T @ SIGMA @ A
    d = np.diag(R).copy()
    off = np.array([R[i, j] for i, j in U])
    for f in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
        O = np.diag(d)
        for v, (i, j) in zip(f * off, U):
            O[i, j] = O[j, i] = v
        if np.linalg.eigvalsh(O)[0] > 1e-3 * d.min():
            return O
    return np.diag(d)


def regression_fit(D):
    B = np.zeros((P, P))
    om = np.zeros(P)
    for j in range(P):
        pa = [i for (i, jj) in D if jj == j]
        if pa:
            beta = np.linalg.solve(SIGMA[np.ix_(pa, pa)], SIGMA[pa, j])
            B[pa, j] = beta
            om[j] = SIGMA[j, j] - SIGMA[j, pa] @ beta
        else:
            om[j] = SIGMA[j, j]
    return B, np.diag(om)


def fit_kl(s, warm=(), rng=None, resid_B=None):
    """Return dict with gap, gap_independent, B, O, method, n_starts, n_iter, seconds."""
    t0 = time.time()
    cfg = CONFIG["kl"]
    if rng is None:
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, s.index, 1]))
    D, U = s.D, s.U
    nd = len(D)
    out = dict(n_starts=0, n_iter=0, n_fail=0)
    gap_indep = None
    if not U:
        B, O = regression_fit(D)
        out.update(method="regression_mle", B=B, O=O, n_starts=1)
    elif not D and s.complete_components:
        mask = _block_mask(s)
        out.update(method="closed_form_noB", B=np.zeros((P, P)), O=np.where(mask, SIGMA, 0.0), n_starts=1)
    else:
        prof = s.complete_components
        mask = _block_mask(s) if prof else None
        Bstarts = [Bw for Bw, _ in warm]
        Ostarts = [Ow for _, Ow in warm]
        B_reg, _ = regression_fit(D)
        Bstarts += [np.zeros((P, P)), B_reg]
        Ostarts += [None, None]
        if nd:
            for _ in range(cfg["random_starts"]):
                Bstarts.append(_embed_B(rng.uniform(*cfg["start_box"], size=nd), D))
                Ostarts.append(None)
        else:  # Omega only, non-complete pattern: random rescalings of the start
            for _ in range(cfg["random_starts"]):
                Bstarts.append(np.zeros((P, P)))
                Ostarts.append("random")
        M = CONFIG["coefficient_bound"]
        Bstarts = [np.clip(Bx, -M, M) for Bx in Bstarts]
        n_indep = len(Bstarts)
        if resid_B is not None:
            Bstarts.append(resid_B)
            Ostarts.append(None)
        best = None
        gap_indep = None
        for k_start, (Bs, Os) in enumerate(zip(Bstarts, Ostarts)):
            if k_start == n_indep:
                gap_indep = best[0] - L_SAT if best is not None else np.inf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if prof:
                    x0 = np.array([Bs[e] for e in D])
                    r = minimize(_profile_vg, x0, args=(D, mask), jac=True, method="L-BFGS-B",
                                 bounds=[(-M, M)] * nd, options=dict(ftol=cfg["lbfgs_ftol"], gtol=cfg["lbfgs_gtol"],
                                              maxiter=cfg["lbfgs_maxiter"], maxcor=cfg["maxcor"]))
                    B = _embed_B(r.x, D)
                    A = np.eye(P) - B
                    O = np.where(mask, A.T @ SIGMA @ A, 0.0)
                else:
                    if isinstance(Os, str):
                        O0 = _omega_start(Bs, U)
                        f = rng.uniform(-1, 1, size=len(U))
                        for k, (i, j) in enumerate(U):
                            O0[i, j] = O0[j, i] = f[k] * O0[i, j]
                        if np.linalg.eigvalsh(O0)[0] <= 0:
                            O0 = np.diag(np.diag(O0))
                    elif Os is None:
                        O0 = _omega_start(Bs, U)
                    else:
                        O0 = Os
                    z0 = np.r_[[Bs[e] for e in D], np.diag(O0), [O0[i, j] for i, j in U]]
                    r = minimize(_joint_vg, z0, args=(D, U), jac=True, method="L-BFGS-B",
                                 bounds=[(-M, M)] * nd + [(None, None)] * (P + len(U)), options=dict(ftol=cfg["lbfgs_ftol"], gtol=cfg["lbfgs_gtol"],
                                              maxiter=cfg["lbfgs_maxiter"], maxcor=cfg["maxcor"]))
                    B, O = _unpack_joint(r.x, D, U)
            out["n_starts"] += 1
            out["n_iter"] += int(r.nit)
            v = loss(B, O, SIGMA)
            if not np.isfinite(v):
                out["n_fail"] += 1
                continue
            if best is None or v < best[0]:
                best = (v, B, O)
            if v - L_SAT < cfg["exact_stop"]:
                break
        if gap_indep is None:
            gap_indep = best[0] - L_SAT
        out.update(method="profile_lbfgsb" if prof else "joint_lbfgsb", B=best[1], O=best[2])
    out["gap"] = loss(out["B"], out["O"], SIGMA) - L_SAT
    out["gap_independent"] = out["gap"] if gap_indep is None else float(gap_indep)
    out["exact"] = bool(out["gap"] < cfg["exact_tol"])
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------
# Residual (exactness) fit
# ----------------------------------------------------------------------------

def _resid_fun(x, D, req):
    B = _embed_B(x, D)
    A = np.eye(P) - B
    R = A.T @ SIGMA @ A
    d = np.sqrt(np.diag(R))
    return np.array([R[i, j] / (d[i] * d[j]) for i, j in req])


def _resid_jac(x, D, req):
    B = _embed_B(x, D)
    A = np.eye(P) - B
    SA = SIGMA @ A
    R = A.T @ SA
    dg = np.diag(R)
    J = np.zeros((len(req), len(D)))
    for r_, (i, j) in enumerate(req):
        rho = R[i, j] / np.sqrt(dg[i] * dg[j])
        for c_, (k, l) in enumerate(D):
            dRij = -((SA[k, j] if i == l else 0.0) + (SA[k, i] if j == l else 0.0))
            dRii = -2 * SA[k, i] if i == l else 0.0
            dRjj = -2 * SA[k, j] if j == l else 0.0
            J[r_, c_] = dRij / np.sqrt(dg[i] * dg[j]) - 0.5 * rho * (dRii / dg[i] + dRjj / dg[j])
    return J


def _ls_multistart(D, req, starts, M, all_starts, cfg):
    best = None
    sols = []
    n_run = 0
    for x0 in starts:
        n_run += 1
        x0 = np.clip(x0, -M * (1 - 1e-12), M * (1 - 1e-12))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = least_squares(_resid_fun, x0, jac=_resid_jac, args=(D, req), method="trf",
                              bounds=(-M, M), xtol=cfg["xtol"], ftol=cfg["ftol"], gtol=cfg["gtol"],
                              max_nfev=cfg["max_nfev"])
        m = float(np.max(np.abs(r.fun)))
        if best is None or m < best[0]:
            best = (m, r.x.copy())
        if m < cfg["exact_tol"]:
            if not any(np.max(np.abs(r.x - y)) < cfg["solution_cluster_tol"] for y in sols):
                sols.append(r.x.copy())
        if m < cfg["exact_stop"] and not all_starts:
            break
    return best, sols, n_run


def fit_resid(s, rng=None, all_starts=None, diagnostic=True):
    t0 = time.time()
    cfg = CONFIG["resid"]
    M = CONFIG["coefficient_bound"]
    if rng is None:
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, s.index, 2]))
    if all_starts is None:
        all_starts = s.count <= cfg["all_starts_up_to_edges"]
    D, U = s.D, s.U
    req = [pr for pr in PAIRS if pr not in set(U)]
    nd = len(D)
    out = dict(n_starts=0, solutions=[], at_bound=False, resid_1e4=None, resid_1e5=None, limit=False)
    if not req:
        out.update(resid=0.0, B=np.zeros((P, P)), method="no_constraint")
    elif not nd:
        out.update(resid=float(np.max(np.abs(_resid_fun(np.zeros(0), D, req)))), B=np.zeros((P, P)),
                   method="no_free_coefficient")
    else:
        B_reg, _ = regression_fit(D)
        starts = [np.zeros(nd), np.array([B_reg[e] for e in D])]
        starts += [rng.uniform(*cfg["narrow_box"], size=nd) for _ in range(cfg["random_starts_narrow"])]
        starts += [rng.uniform(*cfg["wide_box"], size=nd) for _ in range(cfg["random_starts_wide"])]
        best, sols, out["n_starts"] = _ls_multistart(D, req, starts, M, all_starts, cfg)
        out.update(resid=best[0], B=_embed_B(best[1], D), method="least_squares_trf",
                   solutions=[_embed_B(y, D) for y in sols])
        out["at_bound"] = bool(np.max(np.abs(best[1])) > 0.999 * M)
        ld = CONFIG["rho_at_infinity_diagnostic"]
        if diagnostic and out["at_bound"] and best[0] < 1e-2 and best[0] >= cfg["exact_tol"]:
            for Mx, key in zip(ld["bounds"], ["resid_1e4", "resid_1e5"]):
                st = [best[1] * (Mx / M)] + [rng.uniform(-5, 5, size=nd) for _ in range(ld["starts"] - 1)]
                bx, _, _ = _ls_multistart(D, req, st, Mx, False, cfg)
                out[key] = bx[0]
            out["limit"] = bool(out["resid_1e5"] < ld["limit_if_ratio_below"] * best[0])
    out["exact"] = bool(out["resid"] < cfg["exact_tol"])
    out["n_solutions"] = len(out["solutions"]) if out["method"] == "least_squares_trf" else None
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------

def _row(s, kl, rs):
    O = kl["O"]
    ev = np.linalg.eigvalsh(O)
    Ar = np.eye(P) - rs["B"]
    Or = Ar.T @ SIGMA @ Ar
    return dict(index=s.index, name=s.name, edges=s.count, n_directed=len(s.D), n_bidirected=len(s.U),
                bowed=int(s.bowed), complete_components=int(s.complete_components),
                kl_method=kl["method"], gap=kl["gap"], exact_kl=int(kl["exact"]),
                gap_independent=kl["gap_independent"], exact_kl_independent=int(kl["gap_independent"] < CONFIG["kl"]["exact_tol"]),
                kl_starts=kl["n_starts"],
                kl_iters=kl["n_iter"], kl_fail=kl.get("n_fail", 0), kl_maxabsB=float(np.max(np.abs(kl["B"]))),
                kl_omega_eigmin=float(ev[0]), kl_omega_eigmax=float(ev[-1]), kl_seconds=kl["seconds"],
                resid_method=rs["method"], resid=rs["resid"], exact_resid=int(rs["exact"]),
                resid_starts=rs["n_starts"], resid_maxabsB=float(np.max(np.abs(rs["B"]))),
                resid_omega_eigmin=float(np.linalg.eigvalsh(Or)[0]),
                resid_omega_eigmax=float(np.linalg.eigvalsh(Or)[-1]),
                resid_at_bound=int(rs["at_bound"]), resid_1e4=rs["resid_1e4"], resid_1e5=rs["resid_1e5"],
                rho_vanishes_at_infinity=int(rs["limit"]),
                n_exact_solutions=rs["n_solutions"], resid_seconds=rs["seconds"])


def _fit_full(args):
    idx, warm = args
    s = SUPPORTS[idx]
    rs = fit_resid(s)
    kl = fit_kl(s, warm=warm, resid_B=rs["B"] if rs["method"] == "least_squares_trf" else None)
    sols = [B for B in rs["solutions"]] if s.count <= 5 else []
    return _row(s, kl, rs), (kl["B"], kl["O"]), sols


def _fit_timing(idx):
    s = SUPPORTS[idx]
    rs = fit_resid(s)
    kl = fit_kl(s, warm=(), resid_B=rs["B"] if rs["method"] == "least_squares_trf" else None)
    return _row(s, kl, rs)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def param_record():
    return dict(B={"1->2": B_STAR[0, 1], "2->3": B_STAR[1, 2], "3->4": B_STAR[2, 3]},
                Omega_diag=[float(x) for x in np.diag(O_STAR)], Omega_offdiag={"2<->4": O_STAR[1, 3]},
                Sigma_star=SIGMA.tolist(), l_sat=L_SAT,
                Omega_eigenvalues=np.linalg.eigvalsh(O_STAR).tolist())


def main_timing(workers):
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, 0, 4]))
    idx = sorted(rng.choice(len(SUPPORTS), size=200, replace=False).tolist())
    t0 = time.time()
    with mpc.Pool(workers) as pool:
        rows = pool.map(_fit_timing, idx, chunksize=1)
    wall = time.time() - t0
    write_csv(RESULTS / "timing_200.csv", rows)
    cpu = sum(r["kl_seconds"] + r["resid_seconds"] for r in rows)
    per = cpu / len(rows)
    proj_cpu_h = per * len(SUPPORTS) / 3600
    proj_wall_h = proj_cpu_h / workers
    rec = dict(n=len(rows), wall_seconds=wall, cpu_seconds=cpu, cpu_seconds_per_support=per,
               kl_seconds_per_support=sum(r["kl_seconds"] for r in rows) / len(rows),
               resid_seconds_per_support=sum(r["resid_seconds"] for r in rows) / len(rows),
               max_seconds_one_support=max(r["kl_seconds"] + r["resid_seconds"] for r in rows),
               projected_cpu_hours_all=proj_cpu_h, workers=workers, projected_wall_hours=proj_wall_h,
               limit_hours=CONFIG["projection_limit_hours"], proceed=bool(proj_wall_h <= CONFIG["projection_limit_hours"]),
               exact_kl=sum(r["exact_kl"] for r in rows), exact_kl_independent=sum(r["exact_kl_independent"] for r in rows),
               exact_resid=sum(r["exact_resid"] for r in rows), rho_vanishes_at_infinity=sum(r["rho_vanishes_at_infinity"] for r in rows),
               disagreements_final=[r["name"] for r in rows if r["exact_kl"] != r["exact_resid"]],
               disagreements_independent=[r["name"] for r in rows if r["exact_kl_independent"] != r["exact_resid"]])
    (RESULTS / "timing_200.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def main_full(workers):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "config.json").write_text(json.dumps(dict(CONFIG, generating=param_record()), indent=1))
    t_all = time.time()
    best = {}
    rows = {}
    sols = {}
    levels = {}
    for s in SUPPORTS:
        levels.setdefault(s.count, []).append(s.index)
    level_times = {}
    with mpc.Pool(workers) as pool:
        for k in sorted(levels):
            t0 = time.time()
            jobs = []
            for idx in levels[k]:
                warm = [best[j] for j in immediate_subsupports(SUPPORTS[idx])]
                jobs.append((idx, warm))
            res = pool.map(_fit_full, jobs, chunksize=4)
            for (idx, _), (row, bo, sl) in zip(jobs, res):
                rows[idx] = row
                best[idx] = bo
                if sl:
                    sols[idx] = sl
            level_times[k] = time.time() - t0
            ne = sum(rows[i]["exact_kl"] for i in levels[k])
            print(f"level {k}: {len(levels[k])} supports, {ne} exact (KL), {time.time() - t0:.1f} s", flush=True)
    rows = [rows[i] for i in range(len(SUPPORTS))]
    write_csv(RESULTS / "fits.csv", rows)
    np.savez_compressed(RESULTS / "fits_params.npz",
                        B=np.array([best[i][0] for i in range(len(SUPPORTS))]),
                        O=np.array([best[i][1] for i in range(len(SUPPORTS))]))
    with open(RESULTS / "exact_solutions_le5.json", "w") as f:
        json.dump({str(i): [B.tolist() for B in v] for i, v in sols.items()}, f)
    rec = dict(wall_seconds=time.time() - t_all, level_seconds=level_times,
               cpu_seconds=sum(r["kl_seconds"] + r["resid_seconds"] for r in rows))
    (RESULTS / "fit_runtime.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="timing", choices=["timing", "full"])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    if args.mode == "timing":
        main_timing(args.workers)
    else:
        main_full(args.workers)

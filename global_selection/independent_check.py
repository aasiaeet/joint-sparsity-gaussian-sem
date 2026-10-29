"""Independent check of the global-selection study on three vertices.

Section S18 of the Supplementary Material ("Relation to the quasi-MCP
criterion" and "Independent reimplementation").  No solver code is shared with
global_selection.py:

* Supports are enumerated by nilpotency of the adjacency matrix (A^3 = 0).
* The constrained likelihood l_min(H; S) over K is computed by a joint direct
  parametrization z = (B_D, diag Omega, Omega_U), without profiling Omega,
  with a logarithmic barrier for the spectral box 0.05 <= eig(Omega) <= 10
  (barrier path mu = 1e-3, ..., 1e-13) and L-BFGS-B for |B| <= 2, from many
  random starts.  Supports without bidirected edges use the regression MLE
  when it lies in K.
* Since supp H' within supp H implies l_min(H) <= l_min(H'), the reported
  value is the minimum over sub-supports (counted as closure changes).
* Model classes are recomputed from a finite-difference Jacobian at fresh
  random points and, for the generating classes, by mutual-containment fits.
* Regional lower bounds for the criterion l_S + lambda P_delta: for every
  vertex order and every set L of coordinates at least delta in magnitude,
  q_delta(t) >= |t|/delta on |t| <= delta gives the lower bound
      min  l_S(theta) + lambda |L| + (lambda/delta) sum_{t notin L} |theta_t|,
      |theta_t| <= delta for t notin L,
  and these regions cover K.  The minima are computed by multistart local
  optimization, so the bounds are numerical, not certificates.

Stages (python3 independent_check.py STAGE [--workers W] [--limit N] [--out DIR]):
  population  population fibers and classes for all 200 supports
              -> independent_check_population.json (+ _population_raw.json)
  samples     350 regenerated samples: all whose selection is not a
              minimum-edge support or whose tie set is incomplete, and five
              others per cell; compared with runs.csv.gz and
              lmin_all_datasets.npz -> independent_check_samples.csv
              (+ independent_check_samples_raw.json, about 3.7 MB, not shipped)
  certify     regional lower bounds on 86 of these samples
              -> independent_check_certify_subset.csv; needs
              independent_check_samples_raw.json from the samples stage
  tables      recomputes the selection proportions and Wilson intervals of
              summary.csv -> independent_check_tables.csv
  all         all four in order

Inputs are read from results/; outputs go to --out (default results/).
--limit N runs only the first N samples of the samples stage (a fast check).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES
P = 3
MB, OLO, OHI = 2.0, 0.05, 10.0
MASTER = 20260925
DESIGNS = {  # (design index, B edges, Omega off-diagonal, e*)
    "G1": (0, {(0, 1): 0.8, (1, 2): -0.7}, {}, 2),
    "G2": (1, {(0, 1): 0.7}, {(1, 2): 0.5}, 2),
    "G3": (2, {(0, 1): 0.7, (1, 2): 0.6}, {(0, 2): 0.4}, 3),
}
NS = [100, 200, 400, 800, 1600, 3200, 6400]
VSEED = 5150001  # seed for all randomness in this check
BIG = 1e20
MUS = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]
N_RANDOM_STARTS = 12
WORKERS = min(8, os.cpu_count() or 1)
LIMIT = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- supports
ORD = [(i, j) for i in range(P) for j in range(P) if i != j]
PR = [(0, 1), (0, 2), (1, 2)]


def is_acyclic(D):
    A = np.zeros((P, P), int)
    for i, j in D:
        A[i, j] = 1
    return not np.any(np.linalg.matrix_power(A, P))


class Sup:
    def __init__(self, D, U):
        self.D = tuple(sorted(D))
        self.U = tuple(sorted(U))
        self.count = len(D) + len(U)
        self.key = frozenset([f"{i+1}->{j+1}" for i, j in D] + [f"{i+1}<->{j+1}" for i, j in U])
        und = {tuple(sorted(e)) for e in D}
        self.bowed = any(u in und for u in U)

    def orders(self):
        return [pm for pm in itertools.permutations(range(P))
                if all(pm.index(i) < pm.index(j) for i, j in self.D)]


def all_supports():
    out = []
    for r in range(len(ORD) + 1):
        for D in itertools.combinations(ORD, r):
            if not is_acyclic(D):
                continue
            for s in range(4):
                for U in itertools.combinations(PR, s):
                    out.append(Sup(D, U))
    return out


SUPS = all_supports()
KEY2I = {s.key: k for k, s in enumerate(SUPS)}
SUBSETS = [[k2 for k2, t in enumerate(SUPS) if t.key <= s.key and k2 != k] for k, s in enumerate(SUPS)]


def key_of(name):
    return frozenset() if name == "empty" else frozenset(name.split(","))


def name_of(key):
    return ",".join(sorted(key, key=lambda e: ("<->" in e, e))) or "empty"


# ---------------------------------------------------------------- model
def sigma_of(B, O):
    Ai = np.linalg.inv(np.eye(P) - B)
    return Ai.T @ O @ Ai


def design_theta(d):
    _, Bd, Od, _ = DESIGNS[d]
    B = np.zeros((P, P))
    for e, v in Bd.items():
        B[e] = v
    O = np.eye(P)
    for (i, j), v in Od.items():
        O[i, j] = O[j, i] = v
    return B, O


def lik(B, O, S):
    ev = np.linalg.eigvalsh(O)
    if ev[0] <= 0:
        return np.inf
    A = np.eye(P) - B
    return 0.5 * (np.sum(np.log(ev)) + np.trace(np.linalg.solve(O, A.T @ S @ A)))


def unpack(z, D, U):
    B = np.zeros((P, P))
    nd = len(D)
    for v, e in zip(z[:nd], D):
        B[e] = v
    O = np.diag(z[nd:nd + P])
    for v, (i, j) in zip(z[nd + P:], U):
        O[i, j] = O[j, i] = v
    return B, O


def fg(z, S, D, U, mu, extra=None):
    """l_S plus barrier mu * (-sum log(ev-lo) - sum log(hi-ev)); optional extra(z) -> (val, grad)."""
    B, O = unpack(z, D, U)
    ev, V = np.linalg.eigh(O)
    if ev[0] <= OLO or ev[-1] >= OHI:
        return BIG, np.zeros_like(z)
    A = np.eye(P) - B
    R = A.T @ S @ A
    Oi = (V / ev) @ V.T
    val = 0.5 * (np.sum(np.log(ev)) + np.sum(Oi * R))
    gB = -S @ A @ Oi
    gO = 0.5 * (Oi - Oi @ R @ Oi)
    if mu > 0:
        val -= mu * (np.sum(np.log(ev - OLO)) + np.sum(np.log(OHI - ev)))
        gO = gO - mu * (V * (1 / (ev - OLO) - 1 / (OHI - ev))) @ V.T
    g = np.r_[[gB[e] for e in D], np.diag(gO), [2 * gO[i, j] for i, j in U]]
    if extra is not None:
        v2, g2 = extra(z)
        val += v2
        g = g + g2
    return float(val), g


def feasible_start(B0, S, D, U, rng=None):
    A = np.eye(P) - B0
    R = A.T @ S @ A
    d = np.clip(np.diag(R), OLO + 0.02, OHI - 0.5)
    off = np.array([R[i, j] for i, j in U])
    if rng is not None and len(U):
        off = off * rng.uniform(0.0, 1.0, len(U))
    for s in [1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05, 0.0]:
        z = np.r_[[B0[e] for e in D], d, s * off]
        ev = np.linalg.eigvalsh(unpack(z, D, U)[1])
        if ev[0] > OLO + 1e-3 and ev[-1] < OHI - 1e-3:
            return z
    raise RuntimeError("no feasible start")


def barrier_path(z0, S, D, U, bounds, extra=None):
    z = z0.copy()
    for mu in MUS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(fg, z, args=(S, D, U, mu, extra), jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(ftol=1e-16, gtol=1e-12, maxiter=3000, maxcor=20))
        if r.fun < BIG / 2:
            z = r.x
    return z


def ols(S, D):
    B = np.zeros((P, P))
    om = np.zeros(P)
    for j in range(P):
        pa = [i for i, jj in D if jj == j]
        if pa:
            b = np.linalg.solve(S[np.ix_(pa, pa)], S[pa, j])
            B[pa, j] = b
            om[j] = S[j, j] - S[j, pa] @ b
        else:
            om[j] = S[j, j]
    return B, np.diag(om)


def lmin(S, s, rng, nrand=N_RANDOM_STARTS):
    """Returns (value, B, Omega, method)."""
    D, U = s.D, s.U
    Bo, Oo = ols(S, D)
    if not U:
        if np.all(np.abs(Bo) <= MB) and np.all((np.diag(Oo) >= OLO) & (np.diag(Oo) <= OHI)):
            return lik(Bo, Oo, S), Bo, Oo, "regression"
    bounds = [(-MB, MB)] * len(D) + [(OLO, OHI)] * P + [(-OHI, OHI)] * len(U)
    starts = [np.zeros((P, P)), np.clip(Bo, -MB + 1e-3, MB - 1e-3)]
    nr = nrand if D else 2
    for _ in range(nr):
        B0 = np.zeros((P, P))
        for e in D:
            B0[e] = rng.uniform(-MB + 1e-3, MB - 1e-3)
        starts.append(B0)
    best = None
    for k, B0 in enumerate(starts):
        z0 = feasible_start(B0, S, D, U, rng if k >= 2 else None)
        z = barrier_path(z0, S, D, U, bounds)
        B, O = unpack(z, D, U)
        v = lik(B, O, S)
        if best is None or v < best[0]:
            best = (v, B, O, "barrier")
    return best


def lmin_all(S, rng):
    fits = [lmin(S, s, rng) for s in SUPS]
    raw = np.array([f[0] for f in fits])
    val = raw.copy()
    # monotone closure over sub-supports (valid: feasible sets are nested)
    for k in np.argsort([s.count for s in SUPS], kind="stable"):
        for k2 in SUBSETS[k]:
            if val[k2] < val[k]:
                val[k] = val[k2]
    return raw, val, fits


def sample_cov(d, n, rep):
    di = DESIGNS[d][0]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, di, n, rep]))
    Sig = sigma_of(*design_theta(d))
    L = np.linalg.cholesky(Sig)
    X = rng.standard_normal((n, P)) @ L.T
    X = X - X.mean(axis=0)
    return X.T @ X / n


# ---------------------------------------------------------------- classes
def fd_jacobian(s, B, O):
    nd = len(s.D)
    z = np.r_[[B[e] for e in s.D], np.diag(O), [O[i, j] for i, j in s.U]]
    iu = np.triu_indices(P)

    def sig(zz):
        return sigma_of(*unpack(zz, s.D, s.U))[iu]
    h = 1e-6
    return np.column_stack([(sig(z + h * e) - sig(z - h * e)) / (2 * h) for e in np.eye(len(z))])


def cvals(Sig):
    d = np.sqrt(np.diag(Sig))
    C = Sig / np.outer(d, d)
    K = np.linalg.inv(Sig)
    k = np.sqrt(np.diag(K))
    PC = -K / np.outer(k, k)
    return np.array([C[0, 1], C[0, 2], C[1, 2], PC[0, 1], PC[0, 2], PC[1, 2]])


CNAMES = ["rho12", "rho13", "rho23", "rho12.3", "rho13.2", "rho23.1"]


def rand_theta(s, rng):
    while True:
        B = np.zeros((P, P))
        for e in s.D:
            B[e] = rng.uniform(0.25, 1.1) * rng.choice([-1, 1])
        O = np.diag(rng.uniform(0.7, 1.5, P))
        for i, j in s.U:
            O[i, j] = O[j, i] = rng.uniform(0.1, 0.45) * rng.choice([-1, 1])
        if np.linalg.eigvalsh(O)[0] > 0.2:
            return B, O


def classes():
    rng = np.random.default_rng(VSEED + 1)
    out = []
    for s in SUPS:
        rk = 0
        vm = np.zeros(6)
        for _ in range(5):
            B, O = rand_theta(s, rng)
            sv = np.linalg.svd(fd_jacobian(s, B, O), compute_uv=False)
            rk = max(rk, int(np.sum(sv > 1e-6 * sv[0])))
            vm = np.maximum(vm, np.abs(cvals(sigma_of(B, O))))
        van = tuple(c for c, v in zip(CNAMES, vm) if v < 1e-9)
        out.append((rk, van))
    return out


# ---------------------------------------------------------------- population
def pop_design(d):
    rng = np.random.default_rng(np.random.SeedSequence([VSEED, DESIGNS[d][0], 0]))
    Sig = sigma_of(*design_theta(d))
    lstar = 0.5 * np.linalg.slogdet(Sig)[1] + P / 2
    raw, val, fits = lmin_all(Sig, rng)
    return dict(lstar=lstar, raw=raw.tolist(), val=val.tolist(),
                fits=[(f[1].tolist(), f[2].tolist(), f[3]) for f in fits])


def mutual_fit(sa, sb, rng, reps=3):
    """Does sb fit Sigma(theta) exactly for random theta on sa (relaxed to |B|<=2 and the barrier box of this check)?"""
    worst = 0.0
    for _ in range(reps):
        B, O = rand_theta(sa, rng)
        Sig = sigma_of(B, O)
        ls = 0.5 * np.linalg.slogdet(Sig)[1] + P / 2
        worst = max(worst, lmin(Sig, sb, rng)[0] - ls)
    return worst


def stage_population():
    import multiprocessing as mp
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(min(3, WORKERS)) as pool:
        res = dict(zip(DESIGNS, pool.map(pop_design, list(DESIGNS))))
    cls = classes()
    stored = json.load(open(RES / "population.json"))
    main_cls = {key_of(x["support"]): (x["dim"], tuple(x["vanishing"])) for x in stored["supports"]}
    cls_agree = sum(main_cls[s.key] == cls[k] for k, s in enumerate(SUPS))
    report = dict(n_supports=len(SUPS), class_agreement=f"{cls_agree}/{len(SUPS)}",
                  n_distinct_classes=len(set(cls)), designs={})
    rng = np.random.default_rng(VSEED + 7)
    for d, r in res.items():
        e_star = DESIGNS[d][3]
        gaps = np.array(r["val"]) - r["lstar"]
        rawgaps = np.array(r["raw"]) - r["lstar"]
        fit = [k for k in range(len(SUPS)) if gaps[k] <= 1e-10]
        amb = [(name_of(SUPS[k].key), float(gaps[k])) for k in range(len(SUPS)) if 1e-10 < gaps[k] < 1e-5]
        cmin = min(SUPS[k].count for k in fit)
        emin = sorted(name_of(SUPS[k].key) for k in fit if SUPS[k].count == cmin)
        T = stored["designs"][d]
        main_emin = sorted(name_of(key_of(x)) for x in T["E_min_names"])
        main_gaps = {key_of(k): v for k, v in T["gaps"].items()}
        gdiff = [(name_of(s.key), float(gaps[k]), main_gaps[s.key]) for k, s in enumerate(SUPS)
                 if abs(gaps[k] - main_gaps[s.key]) > 1e-8]
        gsup = KEY2I[frozenset([f"{i+1}->{j+1}" for i, j in DESIGNS[d][1]] +
                               [f"{i+1}<->{j+1}" for i, j in DESIGNS[d][2]])]
        gcls = cls[gsup]
        emin_idx = [k for k in fit if SUPS[k].count == cmin]
        # mutual containment for same-dimension classes (skip saturated: generating model is open)
        mutual = {}
        if gcls[0] < 6:
            for k in emin_idx:
                mutual[name_of(SUPS[k].key)] = (mutual_fit(SUPS[gsup], SUPS[k], rng),
                                                mutual_fit(SUPS[k], SUPS[gsup], rng))
        # per-edge gap to fewer-edge supports (population crossover)
        per_edge = min(gaps[k] / (e_star - SUPS[k].count) for k in range(len(SUPS)) if SUPS[k].count < e_star)
        # crossover n: largest n with log(n)/n >= per_edge (n >= 3)
        ncross = max(n for n in range(3, 100000) if math.log(n) / n >= per_edge)
        nonfit_le = [k for k in range(len(SUPS)) if gaps[k] > 1e-10 and SUPS[k].count <= e_star]
        kmin = min(nonfit_le, key=lambda k: gaps[k])
        emin_min_coord = min(min([abs(np.array(r["fits"][k][0])[e]) for e in SUPS[k].D] +
                                 [abs(np.array(r["fits"][k][1])[e]) for e in SUPS[k].U]) for k in emin_idx)
        emin_boundary = [name_of(SUPS[k].key) for k in emin_idx
                         if np.max(np.abs(np.array(r["fits"][k][0]))) > MB - 1e-4 or
                         np.linalg.eigvalsh(np.array(r["fits"][k][1]))[0] < OLO + 1e-4 or
                         np.linalg.eigvalsh(np.array(r["fits"][k][1]))[-1] > OHI - 1e-4]
        report["designs"][d] = dict(
            l_star=r["lstar"], n_exact_fit=len(fit), main_n_exact_fit=T["n_exactly_fitting"],
            minimal_count=cmin, main_minimal_count=T["minimal_fitting_count"],
            E_min=emin, main_E_min=main_emin, E_min_equal=emin == main_emin,
            generating_class=list(gcls), E_min_all_in_generating_class=all(cls[k] == gcls for k in emin_idx),
            n_class_members_with_e_star=sum(1 for k, s in enumerate(SUPS) if s.count == e_star and cls[k] == gcls),
            n_E_min_bowed=sum(SUPS[k].bowed for k in emin_idx),
            ambiguous_gaps=amb, gap_disagreements_gt_1e8=gdiff[:40], n_gap_disagreements=len(gdiff),
            monotone_closure_changes=int(np.sum(rawgaps - gaps > 1e-10)),
            nearest_nonfit_le_estar=(name_of(SUPS[kmin].key), float(gaps[kmin]), SUPS[kmin].count),
            main_nearest=(T["smallest_gap_support"], T["smallest_gap_nonfitting_count_le_e_star"]),
            per_edge_gap_fewer=float(per_edge), crossover_n=ncross,
            E_min_min_abs_coordinate=float(emin_min_coord), E_min_touching_bound=emin_boundary,
            mutual_containment_max_gaps=mutual,
        )
    report["runtime_s"] = time.monotonic() - t0
    (OUT / "independent_check_population.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    json.dump(dict(res=res), open(OUT / "independent_check_population_raw.json", "w"))
    log(f"population done in {report['runtime_s']:.0f}s")
    for d, v in report["designs"].items():
        log(f"{d}: Emin equal={v['E_min_equal']} |Emin|={len(v['E_min'])} fit={v['n_exact_fit']}/{v['main_n_exact_fit']} "
            f"gapdiff={v['n_gap_disagreements']} amb={len(v['ambiguous_gaps'])} cross={v['crossover_n']} per_edge={v['per_edge_gap_fewer']:.5f}")
    return report


# ---------------------------------------------------------------- samples
def one_dataset(job):
    d, n, rep = job
    t0 = time.monotonic()
    S = sample_cov(d, n, rep)
    rng = np.random.default_rng(np.random.SeedSequence([VSEED, DESIGNS[d][0], n, rep]))
    raw, val, fits = lmin_all(S, rng)
    lam = delta = math.log(n) / n
    cnt = np.array([s.count for s in SUPS])
    score = val + lam * cnt
    best = float(score.min())
    ties = [k for k in range(len(SUPS)) if score[k] <= best + 1e-8]
    ties12 = [k for k in range(len(SUPS)) if score[k] <= best + 1e-11]
    sel = min(ties, key=lambda k: (SUPS[k].count, k))
    order = np.argsort(score)
    runner = next(int(k) for k in order if score[k] > best + 1e-8)

    def mincoord(k):
        s = SUPS[k]
        B, O = fits[k][1], fits[k][2]
        c = [abs(B[e]) for e in s.D] + [abs(O[e]) for e in s.U]
        return min(c) / delta if c else np.inf
    return dict(design=d, n=n, rep=rep, S=S.tolist(), min_eig=float(np.linalg.eigvalsh(S)[0]),
                raw=raw.tolist(), val=val.tolist(), best=best,
                ties=[name_of(SUPS[k].key) for k in ties], ties_tight=[name_of(SUPS[k].key) for k in ties12],
                selected=name_of(SUPS[sel].key), sel_min_coord_over_delta=float(mincoord(sel)),
                ties_min_coord_over_delta=float(min(mincoord(k) for k in ties)),
                runner=name_of(SUPS[runner].key), runner_gap=float(score[runner] - best),
                fits={name_of(SUPS[k].key): (fits[k][1].tolist(), fits[k][2].tolist()) for k in ties},
                closure_changes=int(np.sum(raw - val > 1e-10)), runtime=time.monotonic() - t0)


def choose_datasets():
    import pandas as pd
    r = pd.read_csv(RES / "runs.csv.gz")
    pick = set()
    bad = r[(~r.thresholded_is_emin) | (~r.selected_is_emin) | (r.n_ties_1e12 == 35)]
    pick |= set(zip(bad.design, bad.n, bad.replication))
    rng = np.random.default_rng(VSEED + 3)
    for (d, n), g in r.groupby(["design", "n"]):
        good = g[g.thresholded_is_emin & (g.n_ties_1e12 != 35)]
        reps = rng.choice(good.replication.values, size=min(5, len(good)), replace=False)
        pick |= {(d, int(n), int(x)) for x in reps}
    return sorted((str(d), int(n), int(x)) for d, n, x in pick)


def stage_samples():
    import multiprocessing as mp
    import pandas as pd
    jobs = choose_datasets()
    if LIMIT is not None:
        jobs = jobs[:LIMIT]
    log(f"samples: {len(jobs)} datasets")
    t0 = time.monotonic()
    out = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for k, res in enumerate(pool.imap_unordered(one_dataset, jobs, chunksize=1)):
            out.append(res)
            if k % 40 == 0:
                log(f"  {k+1}/{len(jobs)} {time.monotonic()-t0:.0f}s")
    out.sort(key=lambda x: (x["design"], x["n"], x["rep"]))
    json.dump(out, open(OUT / "independent_check_samples_raw.json", "w"))
    # comparison
    r = pd.read_csv(RES / "runs.csv.gz").set_index(["design", "n", "replication"])
    npz = np.load(RES / "lmin_all_datasets.npz")
    main_names = [key_of(x) for x in npz["support_names"]]
    perm = [KEY2I[k] for k in main_names]  # column c of the stored table -> index perm[c] here
    idx = {(str(d), int(n), int(x)): i for i, (d, n, x) in enumerate(zip(npz["design"], npz["n"], npz["replication"]))}
    pop = json.load(open(RES / "population.json"))
    emin = {d: {key_of(x) for x in pop["designs"][d]["E_min_names"]} for d in DESIGNS}
    rows = []
    for o in out:
        d, n, rep = o["design"], o["n"], o["rep"]
        R = r.loc[(d, n, rep)]
        th = np.empty(len(SUPS))
        th[perm] = npz["lmin"][idx[(d, n, rep)]]
        check = np.array(o["val"])
        diff = check - th
        lam = math.log(n) / n
        cnt = np.array([s.count for s in SUPS])
        main_score = th + lam * cnt
        main_ties = {SUPS[k].key for k in range(len(SUPS)) if main_score[k] <= main_score.min() + 1e-8}
        check_ties = {key_of(x) for x in o["ties"]}
        sel_key = key_of(o["selected"])
        rows.append(dict(
            design=d, n=n, rep=rep,
            min_eig_check=o["min_eig"], min_eig_main=R.sample_min_eig,
            S_match=abs(o["min_eig"] - R.sample_min_eig) < 1e-12,
            max_lmin_diff=float(np.max(np.abs(diff))),
            n_check_lower_1e8=int(np.sum(diff < -1e-8)), n_main_lower_1e8=int(np.sum(diff > 1e-8)),
            worst_check_lower=float(diff.min()), worst_main_lower=float(diff.max()),
            where_check_lower=";".join(name_of(SUPS[k].key) for k in np.where(diff < -1e-8)[0]),
            where_main_lower=";".join(name_of(SUPS[k].key) for k in np.where(diff > 1e-8)[0]),
            best_check=o["best"], best_main=R.l0_min_score, best_diff=o["best"] - R.l0_min_score,
            selected_check=o["selected"], selected_main=R.selected_support,
            main_selected_in_check_ties=key_of(R.selected_support) in check_ties,
            tie_sets_equal=check_ties == main_ties, n_ties_check=len(check_ties), n_ties_main_csv=int(R.n_ties_1e12),
            thresholded_main=R.thresholded_support,
            check_selected_is_emin=sel_key in emin[d], main_thresholded_is_emin=bool(R.thresholded_is_emin),
            check_all_ties_same_emin_status=len({t in emin[d] for t in check_ties}) == 1,
            check_count=len(sel_key), main_count=int(R.thresholded_count),
            sel_min_coord_over_delta=o["sel_min_coord_over_delta"], main_min_coord=R.refit_min_coord_over_delta,
            ties_min_coord_over_delta=o["ties_min_coord_over_delta"],
            runner_check=o["runner"], runner_gap_check=o["runner_gap"],
            runner_main=R.runner_up_support, runner_gap_main=R.runner_up_gap,
            closure_changes=o["closure_changes"], runtime=o["runtime"]))
    pd.DataFrame(rows).to_csv(OUT / "independent_check_samples.csv", index=False)
    log(f"samples done {time.monotonic()-t0:.0f}s")


# ---------------------------------------------------------------- certify
def lb_problem(S, order, L, lam, delta, starts, rng, nrand=3):
    """min l_S + lam|L| + (lam/delta) sum_{t notin L}|theta_t| over forward B of `order`, full Omega,
    |theta_t| <= delta off L, theta in K.  Split theta_t = u - v (u, v in [0, delta])."""
    fwd = [(order[a], order[b]) for a in range(P) for b in range(a + 1, P)]
    coords = [("B", e) for e in fwd] + [("O", e) for e in PR]
    T = [c for c in range(6) if c not in L]
    D, U = fwd, PR
    nT = len(T)
    # variable x = (z (3 B + 3 diag + 3 off), v_T); theta_t = z_t - v_t for t in T with z_t in [0, delta]
    zpos = {0: 0, 1: 1, 2: 2, 3: 6, 4: 7, 5: 8}
    ratio = lam / delta

    def f(x):
        z = x[:9].copy()
        vv = x[9:]
        for k, t in enumerate(T):
            z[zpos[t]] = x[zpos[t]] - vv[k]
        val, g = fg(z, S, D, U, mu_holder[0])
        if val >= BIG / 2:
            return BIG, np.zeros_like(x)
        gx = np.r_[g, np.zeros(nT)]
        for k, t in enumerate(T):
            gx[9 + k] = -g[zpos[t]]
        pen = ratio * (np.sum([x[zpos[t]] for t in T]) + np.sum(vv))
        gx[[zpos[t] for t in T]] += ratio
        gx[9:] += ratio
        return val + pen, gx

    bounds = [(-MB, MB)] * 3 + [(OLO, OHI)] * 3 + [(-OHI, OHI)] * 3 + [(0, delta)] * nT
    for t in T:
        bounds[zpos[t]] = (0, delta)
    mu_holder = [0.0]
    best = np.inf
    xs = []
    for B0, O0 in starts:
        z = np.r_[[B0[e] for e in fwd], np.diag(O0), [O0[e] for e in PR]]
        xs.append(z)
    for _ in range(nrand):
        B0 = np.zeros((P, P))
        for e in fwd:
            B0[e] = rng.uniform(-1.5, 1.5)
        xs.append(feasible_start(B0, S, fwd, PR, rng))
    for z in xs:
        z = z.copy()
        vv = np.zeros(nT)
        for k, t in enumerate(T):
            th = float(np.clip(z[zpos[t]], -delta, delta))
            z[zpos[t]] = max(th, 0.0)
            vv[k] = max(-th, 0.0)
        # make Omega strictly feasible
        Bs, Os = unpack(z, fwd, PR)
        ev = np.linalg.eigvalsh(Os)
        if ev[0] <= OLO + 1e-6 or ev[-1] >= OHI - 1e-6:
            for s_ in [0.95, 0.8, 0.6, 0.4, 0.2, 0.0]:
                zz = z.copy()
                zz[6:9] *= s_
                ev = np.linalg.eigvalsh(unpack(zz, fwd, PR)[1])
                if ev[0] > OLO + 1e-6 and ev[-1] < OHI - 1e-6:
                    z = zz
                    break
        x = np.r_[z, vv]
        for t in T:  # re-clip after shrink
            x[zpos[t]] = min(max(x[zpos[t]], 0.0), delta)
        for mu in MUS:
            mu_holder[0] = mu
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = minimize(f, x, jac=True, method="L-BFGS-B", bounds=bounds,
                             options=dict(ftol=1e-16, gtol=1e-12, maxiter=3000))
            if r.fun < BIG / 2:
                x = r.x
        mu_holder[0] = 0.0
        v, _ = f(x)
        best = min(best, v)
    return best + lam * len(L)


def certify_dataset(job):
    d, n, rep, val, best, fits_sel = job
    t0 = time.monotonic()
    S = sample_cov(d, n, rep)
    lam = delta = math.log(n) / n
    rng = np.random.default_rng(np.random.SeedSequence([VSEED, DESIGNS[d][0], n, rep, 5]))
    val = np.array(val)
    # region lower bounds; trivial screen: l >= l_min(full order support) on the region
    worst = np.inf
    worst_at = None
    cache = {}
    solved = 0
    screened = 0
    for order in itertools.permutations(range(P)):
        fwd = [(order[a], order[b]) for a in range(P) for b in range(a + 1, P)]
        full = KEY2I[frozenset([f"{i+1}->{j+1}" for i, j in fwd] + ["1<->2", "1<->3", "2<->3"])]
        coords_keys = [f"{i+1}->{j+1}" for i, j in fwd] + ["1<->2", "1<->3", "2<->3"]
        for r_ in range(7):
            for L in itertools.combinations(range(6), r_):
                if val[full] + lam * len(L) >= best + 1e-9:
                    screened += 1
                    continue
                Lkey = frozenset(coords_keys[c] for c in L)
                # starts: fits of every sub-support of L (from the selection stage refits) + the selected refit
                starts = []
                cand = sorted([k for k, s in enumerate(SUPS) if s.key <= Lkey], key=lambda k: val[k])[:3]
                for k in cand:
                    if k not in cache:
                        cache[k] = lmin(S, SUPS[k], rng, nrand=4)[1:3]
                    Bk, Ok = cache[k]
                    Bf = np.zeros((P, P))
                    for e in fwd:
                        Bf[e] = Bk[e]
                    if np.allclose(Bf, Bk):
                        starts.append((Bk, Ok))
                for nm, (Bs, Os) in fits_sel.items():
                    Bs, Os = np.array(Bs), np.array(Os)
                    Bf = np.zeros((P, P))
                    for e in fwd:
                        Bf[e] = Bs[e]
                    if np.allclose(Bf, Bs):
                        starts.append((Bs, Os))
                lb = lb_problem(S, order, L, lam, delta, starts, rng)
                solved += 1
                if lb < worst:
                    worst, worst_at = lb, (order, [coords_keys[c] for c in L])
    return dict(design=d, n=n, rep=rep, l0_best=best, min_region_lb=float(worst),
                lb_minus_l0=float(worst - best), worst_region=str(worst_at), solved=solved, screened=screened,
                runtime=time.monotonic() - t0)


def stage_certify():
    import multiprocessing as mp
    import pandas as pd
    raw = json.load(open(OUT / "independent_check_samples_raw.json"))
    rng = np.random.default_rng(VSEED + 11)
    # all non-E_min datasets at G1/G2, a sample of G3 small-n ones, and a spread of the rest
    pop = json.load(open(RES / "population.json"))
    emin = {d: {key_of(x) for x in pop["designs"][d]["E_min_names"]} for d in DESIGNS}
    bad = [o for o in raw if key_of(o["selected"]) not in emin[o["design"]]]
    good = [o for o in raw if key_of(o["selected"]) in emin[o["design"]]]
    pick = [o for o in bad if o["design"] != "G3"]
    g3bad = [o for o in bad if o["design"] == "G3"]
    pick += [g3bad[i] for i in rng.choice(len(g3bad), size=min(20, len(g3bad)), replace=False)]
    pick += [good[i] for i in rng.choice(len(good), size=min(50, len(good)), replace=False)]
    jobs = [(o["design"], o["n"], o["rep"], o["val"], o["best"], o["fits"]) for o in pick]
    log(f"certify: {len(jobs)} datasets")
    t0 = time.monotonic()
    out = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for k, res in enumerate(pool.imap_unordered(certify_dataset, jobs, chunksize=1)):
            out.append(res)
            if k % 20 == 0:
                log(f"  {k+1}/{len(jobs)} {time.monotonic()-t0:.0f}s")
    pd.DataFrame(out).sort_values(["design", "n", "rep"]).to_csv(OUT / "independent_check_certify_subset.csv", index=False)
    log(f"certify done {time.monotonic()-t0:.0f}s")


# ---------------------------------------------------------------- tables
def wilson(k, m, z=1.959963984540054):
    p = k / m
    c = (p + z * z / (2 * m)) / (1 + z * z / m)
    h = z / (1 + z * z / m) * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return c - h, c + h


def stage_tables():
    import pandas as pd
    r = pd.read_csv(RES / "runs.csv.gz")
    s = pd.read_csv(RES / "summary.csv").set_index(["design", "n"])
    out = []
    for (d, n), g in r.groupby(["design", "n"]):
        m = len(g)
        k = int(g.thresholded_is_emin.sum())
        lo, hi = wilson(k, m)
        row = dict(design=d, n=n, m=m, k=k, prop=k / m, lo=lo, hi=hi,
                   cls=g.thresholded_in_generating_class.mean(),
                   fewer=(g.thresholded_count < DESIGNS[d][3]).mean(),
                   more=(g.thresholded_count > DESIGNS[d][3]).mean())
        S = s.loc[(d, n)]
        row["matches_summary"] = bool(abs(S.prop_emin - row["prop"]) < 1e-12 and abs(S.emin_wilson_lo - lo) < 1e-9 and
                                      abs(S.emin_wilson_hi - hi) < 1e-9 and abs(S.prop_generating_class - row["cls"]) < 1e-12 and
                                      abs(S.prop_fewer_than_e_star - row["fewer"]) < 1e-12 and
                                      abs(S.prop_more_than_e_star - row["more"]) < 1e-12 and m == 200)
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / "independent_check_tables.csv", index=False)
    log("tables: all cells match summary: " + str(all(o["matches_summary"] for o in out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["population", "samples", "certify", "tables", "all"])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=None, help="samples stage: first N samples only")
    ap.add_argument("--out", default=str(RES), help="output directory (default: results/)")
    args = ap.parse_args()
    WORKERS, LIMIT, OUT = args.workers, args.limit, Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    st = args.stage
    log(f"independent check stage={st} seed={VSEED} numpy={np.__version__}")
    if st in ("population", "all"):
        stage_population()
    if st in ("samples", "all"):
        stage_samples()
    if st in ("certify", "all"):
        stage_certify()
    if st in ("tables", "all"):
        stage_tables()

"""Shared pieces of the standardized l1 search (Supplement S13).

Algebraic invariant tables of all supports, generating parameters, and the evaluation of
one standardized covariance: the minimum of R_1 over all exact representations and over
the representations whose graph lies in the algebraic class of the generating graph.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import gzip
import itertools
import json
import time
from pathlib import Path

import numpy as np

from l1core import (all_supports, complete_supports, support_name, parse_support, is_bowfree,
                    invariant, constraint_names, constraint_label, constraint_values, phi, omega_of,
                    standardize, actual_support, solve_support, r1_value)
from stage_solver import s2_support

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MASTER_SEED = 20261022


def open_table(name):
    """Open results/<name> for reading; fall back to the gzipped copy results/<name>.gz."""
    path = RESULTS / name
    if path.exists():
        return open(path, newline="")
    gz = RESULTS / (name + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", newline="")
    raise FileNotFoundError(path)


# ----------------------------------------------------------------------------
# Invariant table
# ----------------------------------------------------------------------------

def invariant_table(p):
    """(rank, vanishing-constraint indices) for every support, cached in results/invariants_p<p>.csv."""
    path = RESULTS / f"invariants_p{p}.csv"
    sups = all_supports(p)
    names = constraint_names(p)
    if path.exists():
        tab = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                van = tuple(int(x) for x in row["vanishing_idx"].split(";") if x != "")
                tab[parse_support(row["support"])] = (int(row["rank"]), van)
        if len(tab) == len(sups):
            return tab
    tab = {}
    rows = []
    for k, (D, U) in enumerate(sups):
        r, van, smallest = invariant(D, U, p, seed=[MASTER_SEED, p, k], draws=4)
        tab[(D, U)] = (r, van)
        rows.append(dict(support=support_name(D, U), n_edges=len(D) + len(U), bowfree=int(is_bowfree(D, U)),
                         rank=r, vanishing_idx=";".join(map(str, van)),
                         vanishing=" ".join(constraint_label(names[i]) for i in van),
                         smallest_nonvanishing=smallest))
    RESULTS.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return tab


def canon(D, U, p):
    best = None
    for perm in itertools.permutations(range(p)):
        D2 = tuple(sorted((perm[i], perm[j]) for i, j in D))
        U2 = tuple(sorted(tuple(sorted((perm[i], perm[j]))) for i, j in U))
        if best is None or (D2, U2) < best:
            best = (D2, U2)
    return best


def bowfree_graphs_up_to_iso(p):
    return sorted({canon(D, U, p) for (D, U) in all_supports(p) if is_bowfree(D, U)},
                  key=lambda s: (len(s[0]) + len(s[1]), s))


# ----------------------------------------------------------------------------
# Generating parameters
# ----------------------------------------------------------------------------

REGIMES = dict(
    moderate=dict(coef=(0.3, 1.2), diag=(0.6, 1.6), rho=(0.15, 0.6)),
    wide=dict(coef=(0.1, 3.0), diag=(0.05, 1.5), rho=(0.1, 0.9)),
)


def draw_parameter(D, U, p, rng, regime):
    cfg = REGIMES[regime]
    for _ in range(10000):
        B = np.zeros((p, p))
        for e in D:
            B[e] = rng.choice([-1, 1]) * rng.uniform(*cfg["coef"])
        om = rng.uniform(*cfg["diag"], p)
        O = np.diag(om)
        for i, j in U:
            O[i, j] = O[j, i] = rng.choice([-1, 1]) * rng.uniform(*cfg["rho"]) * np.sqrt(om[i] * om[j])
        if np.linalg.eigvalsh(O)[0] > 0.02 * om.min():
            return B, O
    raise RuntimeError("no PD draw")


def grid_parameters(D, U, p, mags, max_signs, rng):
    """Equal-magnitude coefficients with sign patterns; unit error variances; bidirected rho = +-0.5."""
    edges = list(D) + list(U)
    pats = list(itertools.product([1, -1], repeat=len(edges)))
    if len(pats) > max_signs:
        idx = rng.choice(len(pats), max_signs, replace=False)
        pats = [pats[0]] + [pats[i] for i in sorted(idx) if i != 0][:max_signs - 1]
    out = []
    for c in mags:
        for sg in pats:
            B = np.zeros((p, p)); O = np.eye(p)
            for e, s in zip(edges, sg):
                if e in D:
                    B[e] = s * c
                else:
                    O[e] = O[e[::-1]] = s * 0.5
            if np.linalg.eigvalsh(O)[0] > 1e-3:
                out.append((c, sg, B, O))
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def extra_vanishing(S, van_star, p, tol=1e-8):
    vals = np.abs(constraint_values(S))
    return [i for i in np.flatnonzero(vals < tol) if i not in set(van_star)]


def evaluate_sigma_p3(S, key_star, inv_tab, seed, s1_random=20, s2_grid=1001):
    """p = 3: S1 on complete and class supports, S2 on all 200 supports."""
    t0 = time.time()
    p = 3
    rng = np.random.default_rng(seed)
    sups = all_supports(p)
    cls = [s for s in sups if inv_tab[s] == key_star]
    out = {}
    # S1, complete supports
    g1 = []
    for D, U, perm in complete_supports(p):
        sol = solve_support(S, D, U, rng, n_random=s1_random, grid_levels=(-1, 0, 1))
        g1.append(sol)
    g1v = min(s.value for s in g1)
    # S1, class supports
    i1 = []
    for D, U in cls:
        sol = solve_support(S, D, U, rng, n_random=6, grid_levels=(-1, 0, 1), grid_max_dim=2) if len(cls) <= 20 else None
        i1.append(sol)
    i1v = min([s.value for s in i1 if s is not None], default=np.inf)
    # S2, all supports
    R = max(2.5, 1.5 * min(g1v, i1v if np.isfinite(i1v) else g1v) + 0.5)
    v2 = {}
    sup2 = {}
    for D, U in sups:
        v, B, order, _ = s2_support(S, D, U, R=R, n_grid=s2_grid, n_refine=6)
        v2[(D, U)] = v
        if np.isfinite(v):
            sup2[(D, U)] = actual_support(B, omega_of(B, S))
    g2v = min(v2.values())
    i2v = min([v2[s] for s in cls], default=np.inf)
    gmin = min(g1v, g2v)
    imin = min(i1v, i2v)
    # support of a global minimizer (prefer one in the class when tied within 1e-10)
    cands = [(s.value, s.support) for s in g1] + [(v2[h], sup2[h]) for h in sup2]
    best_in = min([v for v, a in cands if inv_tab.get(a) == key_star], default=np.inf)
    best_out = min([v for v, a in cands if inv_tab.get(a) != key_star], default=np.inf)
    arg = min(cands, key=lambda x: x[0])
    out.update(global_s1=g1v, global_s2=g2v, global_min=gmin, class_s1=i1v, class_s2=i2v, class_min=imin,
               margin=imin - gmin, global_support=support_name(*arg[1]), global_support_rank=inv_tab[arg[1]][0],
               global_support_in_class=int(inv_tab[arg[1]] == key_star),
               best_actual_in_class=best_in, best_actual_outside=best_out,
               n_class_supports=len(cls), n_class_feasible_s2=int(sum(np.isfinite(v2[s]) for s in cls)),
               n_supports_feasible_s2=int(sum(np.isfinite(v) for v in v2.values())),
               s1_s2_global_diff=g1v - g2v, s1_s2_class_diff=(i1v - i2v) if np.isfinite(i1v) else np.nan,
               R_grid=R, seconds=time.time() - t0)
    return out, np.array([v2[s] for s in sups])


def maximal_supports(supports):
    sets = [(frozenset(D), frozenset(U)) for D, U in supports]
    out = []
    for (D, U), a in zip(supports, sets):
        if not any(a != b and a[0] <= b[0] and a[1] <= b[1] for b in sets):
            out.append((D, U))
    return out


def evaluate_sigma_p4(S, key_star, inv_tab, class_supports, seed, s1_random=20, class_random=6,
                      maximal_random=30, class_trigger=1e-3, maximal=None, all_class_max=10**9):
    """p = 4: S1 on the 24 complete supports (all local solutions kept).

    The class minimum is at most the best local solution whose actual support is in the class
    (best_local_in_class).  Only when that bound exceeds the global minimum by more than
    class_trigger (or no in-class local solution was found) do we solve every support of the
    class (class_random starts) and, again with maximal_random starts, its maximal supports
    (for generic Sigma the class minimum is attained on the closure of a maximal class support).
    Otherwise the reported margin is the upper bound best_local_in_class - global_min < class_trigger."""
    t0 = time.time()
    p = 4
    rng = np.random.default_rng(seed)
    locs = []
    for D, U, perm in complete_supports(p):
        sol = solve_support(S, D, U, rng, n_random=s1_random, keep_locals=True)
        for v, b in sol.local_values:
            B = np.zeros((p, p))
            for e, x in zip(D, b):
                B[e] = x
            a = actual_support(B, omega_of(B, S))
            locs.append((v, a, perm, B))
    gmin = min(v for v, *_ in locs)
    arg = min(locs, key=lambda x: x[0])
    in_loc = [x for x in locs if inv_tab.get(x[1]) == key_star]
    out_loc = [x for x in locs if inv_tab.get(x[1]) != key_star]
    best_in_loc = min([x[0] for x in in_loc], default=np.inf)
    best_out_loc = min([x[0] for x in out_loc], default=np.inf)
    class_vals = {}
    ran_class = False
    if best_in_loc > gmin + class_trigger:
        ran_class = True
        if len(class_supports) <= all_class_max:
            for D, U in class_supports:
                sol = solve_support(S, D, U, rng, n_random=class_random)
                class_vals[support_name(D, U)] = sol.value if sol is not None else np.inf
        for D, U in (maximal if maximal is not None else maximal_supports(class_supports)):
            sol = solve_support(S, D, U, rng, n_random=maximal_random)
            if sol is not None:
                nm = support_name(D, U)
                class_vals[nm] = min(class_vals.get(nm, np.inf), sol.value)
    class_min = min([best_in_loc] + list(class_vals.values()))
    gkey = inv_tab.get(arg[1])
    out = dict(global_min=gmin, global_support=support_name(*arg[1]), global_order="".join(str(v + 1) for v in arg[2]),
               global_support_rank=gkey[0] if gkey else -1, global_support_in_class=int(gkey == key_star),
               best_local_in_class=best_in_loc, best_local_outside=best_out_loc, class_min=class_min,
               margin=class_min - gmin, margin_is_upper_bound=int(not ran_class), ran_class=int(ran_class),
               class_scope=("all+maximal" if len(class_supports) <= all_class_max else "maximal") if ran_class else "",
               n_class_supports=len(class_supports),
               n_class_feasible=int(sum(np.isfinite(v) for v in class_vals.values())) if ran_class else -1,
               n_locals=len(locs), seconds=time.time() - t0)
    return out, arg[3], class_vals

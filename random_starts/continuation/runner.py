"""Shared pieces for rerunning and continuing runs of the random-start study.

Data, starts, tuning, steps and the cycle are those of ../random_starts.py.
run_alternating_snap repeats random_starts.run_alternating operation by operation and
adds (a) a configurable cycle cap, (b) a snapshot of the state at a given cycle
(default 20,000, the cap of the study) and (c) an optional read-only hook called after
every Omega-step.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent          # random_starts/
sys.path.insert(0, str(STUDY))
import random_starts as rs  # noqa: E402

gs = rs.gs
STUDY_RESULTS = rs.RESULTS
P, OFFDIAG, IU = rs.P, rs.OFFDIAG, rs.IU
penalty, arrow_prox, loss_grad, NoiseProx = rs.penalty, rs.arrow_prox, rs.loss_grad, rs.NoiseProx


def load_cfg(max_cycles=None):
    """Tuning and steps of the random-start study (results/config.json)."""
    cfg = json.loads((STUDY_RESULTS / "config.json").read_text())
    cfg.pop("admissibility", None)
    cfg.pop("steps", None)
    if max_cycles is not None:
        cfg["max_cycles"] = int(max_cycles)
    return cfg


def load_population():
    return json.loads(rs.POPULATION.read_text())["designs"]


def sample_S(cfg, d, n, rep):
    """Sample covariance; seed SeedSequence([master_seed, design_index, n, dataset])."""
    di = cfg["designs"].index(d)
    B, O = gs.design_parameters(d)
    Sig = gs.covariance(B, O)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep]))
    return gs.sample_covariance(Sig, n, rng)


def random_start(cfg, S, d, n, rep, k):
    """Random start k; seed SeedSequence([master_seed, design_index, n, dataset, 2, k])."""
    di = cfg["designs"].index(d)
    srng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep, 2, k]))
    return rs.random_start(S, srng, cfg)


def refit_starts(cfg, S, d, n, rep):
    """Refit starts: constrained fits of all supports with solver seed
    SeedSequence([master_seed, design_index, n, dataset, 1]), then the 50 supports with the
    smallest fixed-tuning criterion at the refit (stable sort).
    Returns {rank (1-based): (B0, O0, start_support, start_criterion)}."""
    di = cfg["designs"].index(d)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["master_seed"], di, n, rep, 1]))
    fits, _ = gs.lmin_all(S, rng)
    crit = np.array([rs.criterion(f.B, f.O, S, cfg["lam"], cfg["delta"]) for f in fits])
    top = np.argsort(crit, kind="stable")[:cfg["refit_starts"]]
    return {rank + 1: (fits[i].B.copy(), fits[i].O.copy(), gs.SUPPORTS[i].name, float(crit[i]))
            for rank, i in enumerate(top)}


def run_alternating_snap(s, b0, o0, cfg, prox=None, snap_at=20000, hook=None):
    """random_starts.run_alternating with a snapshot of the state at cycle `snap_at` and an
    optional hook(t, b_old, o_old, gb_old, go_mid, bn, on, z_omega, prox, order) after each
    Omega-step.  The hook must not modify its arguments."""
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
    curv_b = curv_o = 0.0
    min_eig = np.linalg.eigvalsh(o)[0]
    status = "cycle_cap"
    step = np.nan
    t = 0
    orders = set()
    snap = None
    n_conic0 = sum(v for k, v in prox.statuses.items() if k != "analytic_interior")
    for t in range(1, cfg["max_cycles"] + 1):
        bn, _, order = arrow_prox(b - alpha * gb, alpha, lam, delta, M)
        val_mid, _, go_mid = loss_grad(bn, o, s)
        db = bn - b
        nb2 = float(np.sum(db * db))
        if nb2 > 1e-10:
            curv_b = max(curv_b, alpha * 2 * (val_mid - val - float(np.sum(gb * db))) / nb2)
        z_omega = o - beta * go_mid
        on = prox(z_omega)
        if hook is not None:
            hook(t, b, o, gb, go_mid, bn, on, z_omega, prox, order)
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
        if t == snap_at:
            # the state the study recorded for a run stopped at its 20,000-cycle cap
            snap = dict(B=b.copy(), O=o.copy(), objective=float(f), final_step=step,
                        max_objective_increase=float(max_increase),
                        max_curvature_ratio_B=float(curv_b), max_curvature_ratio_Omega=float(curv_o),
                        min_omega_eig=float(min(min_eig, np.linalg.eigvalsh(o)[0])),
                        noise_conic_calls=int(sum(v for k, v in prox.statuses.items()
                                                  if k != "analytic_interior") - n_conic0),
                        seconds=time.monotonic() - started)
    min_eig = min(min_eig, np.linalg.eigvalsh(o)[0])
    n_conic = sum(v for k, v in prox.statuses.items() if k != "analytic_interior") - n_conic0
    info = dict(objective=float(f), start_objective=float(f_start), cycles=t, status=status,
                final_step=step, max_objective_increase=float(max_increase),
                max_curvature_ratio_B=float(curv_b), max_curvature_ratio_Omega=float(curv_o),
                min_omega_eig=float(min_eig), noise_conic_calls=int(n_conic),
                runtime_seconds=time.monotonic() - started)
    return b, o, info, snap


def describe(b, o, cfg, emin, e_star):
    return rs.describe(b, o, cfg, emin, e_star)

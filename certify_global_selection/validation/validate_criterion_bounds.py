"""Validation of the certified criterion bound (criterion_bounds.py, criterion_bounds_bspace.py).

A. Negative controls: with the target raised to s* + 1e-6 and s* + 1e-3 (both above the
   criterion value F(refit) = s* attained at the refit of the selected support), the
   certifier must not certify the sample.  Run on 3 samples.
B. Point-sampling soundness of every bounding function: random boxes in random regions
   (half of them centered at the selected refit), 400 random points per box drawn from the
   region (B in the box, small coordinates in [-delta', delta'], Omega positive definite,
   concentrated near the profile Omega), compared with the box bound:
     B-space boxes: F >= min(s*, lb) - 1e-12 for every point; F >= s* - 1e-12 if pruned;
     9-D boxes:     F >= lb - 1e-12 for every point of the box in K; no point of an
                    "infeasible" box is in K; F >= s* - 1e-12 for points of "KL" boxes.
C. Two-sided check: F at the selected refit, in interval arithmetic, minus s*.
Writes results/validation/criterion_bounds.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "global_selection"))
warnings.filterwarnings("ignore")
import certify_criterion as cc  # noqa: E402
import criterion_bounds as cb  # noqa: E402
import criterion_bounds_bspace as cbb  # noqa: E402
from criterion_bounds import Ctx, PAIRS, regions_for, certify_region  # noqa: E402
from criterion_bounds_bspace import certify_region_B  # noqa: E402
import global_selection as gs  # noqa: E402

rng = np.random.default_rng(271828)
z, a = cc.load()
out = {}
t0 = time.monotonic()


def make_ctx(k, s_override=None):
    s_star, sel = cc.s_star_of(k)
    aa = a[a["sample"] == k]
    lbp = {(gs.SUPPORTS[int(r.support_index)].D, gs.SUPPORTS[int(r.support_index)].U): float(r.lb)
           for _, r in aa.iterrows()}
    return Ctx(z["S"][k], int(z["n"][k]), s_star if s_override is None else s_override, lbp), s_star, sel


def F_point(S, B, O, lam, delta):
    return gs.criterion(B, O, S, lam, delta)


# ------------------------------------------------------------------ A. negative controls
neg = []
for k in [0, 20, 50]:
    for bump in [1e-6, 1e-3]:
        ctx, s_star, sel = make_ctx(k)
        ctx2 = Ctx(z["S"][k], int(z["n"][k]), s_star + bump, ctx.lb_prof)
        scr, pending = regions_for(ctx2)
        certified_all = True
        failed_region = None
        for reg in pending:
            r = certify_region_B(ctx2, reg, max_boxes=150_000)
            if r is None or not r["certified"]:
                r = certify_region(ctx2, reg, max_boxes=150_000)
            if not r["certified"]:
                certified_all = False
                failed_region = (r["order"], r["L"], r["H_D"], r["H_U"])
                break
        neg.append(dict(sample=k, bump=bump, wrongly_certified=certified_all, first_failed_region=failed_region))
out["negative_controls"] = neg
out["negative_controls_ok"] = not any(x["wrongly_certified"] for x in neg)


# ------------------------------------------------------------------ B. point sampling
def sample_points(ctx, reg, lo, hi, npts, B9=False):
    """Points of region reg with B (and for 9-D boxes Omega) inside the box."""
    S = ctx.S
    dp = ctx.delta_hi
    pts = []
    for _ in range(npts * 3):
        if len(pts) >= npts:
            break
        x = lo[:3] + rng.uniform(size=3) * (hi[:3] - lo[:3])
        B = np.zeros((3, 3))
        for c, (i, j) in enumerate(reg.fwd):
            B[i, j] = x[c]
        if B9:
            xo = lo[3:] + rng.uniform(size=6) * (hi[3:] - lo[3:])
            O = np.diag(xo[3:])
            for c, (i, j) in enumerate(PAIRS):
                O[i, j] = O[j, i] = xo[c]
        else:
            A = np.eye(3) - B
            R = A.T @ S @ A
            O = cbb.profile_omega(R[None], reg.U)[0]
            sc = 10 ** rng.uniform(-4, -0.5)
            E = rng.standard_normal((3, 3)) * sc
            O = O + (E + E.T) / 2
            for c in range(3, 6):
                if c not in reg.L:
                    p, q = PAIRS[c - 3]
                    O[p, q] = O[q, p] = rng.uniform(-dp, dp) * rng.choice([0, 0.1, 1])
        if np.linalg.eigvalsh(O)[0] <= 1e-6:
            continue
        pts.append((B, O))
    return pts


viol = dict(bspace_bound=0, bspace_prune=0, nine_bound=0, nine_infeasible=0, nine_kl=0)
counts = dict(bspace_boxes=0, bspace_points=0, nine_boxes=0, nine_points=0)
worst = dict(bspace=math.inf, nine=math.inf)
for k in [0, 20, 36, 50, 85]:
    ctx, s_star, sel = make_ctx(k)
    lam = math.log(ctx.n) / ctx.n
    scr, pending = regions_for(ctx)
    selB = z["B"][k][sel]
    for trial in range(40):
        reg = pending[rng.integers(len(pending))]
        tau = ctx.s_star - float((ctx.lam * float(len(reg.L))).lo) - float(ctx.fsat.lo) - 1e-12
        # B-space box
        if tau < 0.09:
            mu_lo, mu_hi = ctx.mu_bounds(tau)
            mus = []
            for mu in sorted({mu_hi, min(1.9, mu_hi * 1.05), min(1.9, max(mu_hi, 1.3)), 1.6, 1.9}):
                if mu >= mu_hi:
                    MH = cb.IA.point(np.array(mu))
                    mR = ((cb.IA.point(np.array(1.0)) / MH - 0.5) / MH.sqr()).lo
                    if mR > 0:
                        mus.append((mR, mu))
            cen = np.array([selB[e] if (c in reg.L) else 0.0 for c, e in enumerate(reg.fwd)])
            if trial % 2:
                cen = np.array([rng.uniform(-1.5, 1.5) if c in reg.L else 0.0 for c in range(3)])
            w = 10 ** rng.uniform(-3, -0.5)
            lo = np.array([max(-2, cen[c] - w) if c in reg.L else -ctx.delta_hi for c in range(3)])
            hi = np.array([min(2, cen[c] + w) if c in reg.L else ctx.delta_hi for c in range(3)])
            pr = cbb.prune_B(ctx, reg, lo[None], hi[None], mu_lo, mu_hi)[0]
            lbP = cbb.bound_P(ctx, reg, lo[None], hi[None], mus)[0]
            lbB = cbb.bound_B_multi(ctx, reg, lo[None], hi[None], mus)[0][0]
            lb = max(lbP, lbB)
            counts["bspace_boxes"] += 1
            for B, O in sample_points(ctx, reg, np.r_[lo, np.zeros(6)], np.r_[hi, np.zeros(6)], 400):
                f = F_point(ctx.S, B, O, lam, lam)
                counts["bspace_points"] += 1
                if pr and f < s_star - 1e-12:
                    viol["bspace_prune"] += 1
                if f < min(s_star, lb) - 1e-12:
                    viol["bspace_bound"] += 1
                worst["bspace"] = min(worst["bspace"], f - min(s_star, lb))
        # 9-D box around a random point of the region
        lo0, hi0 = reg.initial_box()
        c9 = lo0 + rng.uniform(size=9) * (hi0 - lo0)
        if trial % 2:
            A = np.eye(3) - selB
            Osel = z["O"][k][sel]
            for c, e in enumerate(reg.fwd):
                c9[c] = selB[e] if c in reg.L else 0.0
            for c, (i, j) in enumerate(PAIRS):
                c9[3 + c] = Osel[i, j] if (3 + c) in reg.L else 0.0
            c9[6:] = np.diag(Osel)
        w9 = 10 ** rng.uniform(-3, -1)
        lo9 = np.maximum(lo0, c9 - w9 * (hi0 - lo0))
        hi9 = np.minimum(hi0, c9 + w9 * (hi0 - lo0))
        nL = len(reg.L)
        tau9 = ctx.s_star - float((ctx.lam * float(nL)).lo) - float(ctx.fsat.lo) - 1e-12
        mu9 = ctx.mu_bounds(max(tau9, 0.0))
        lb9, st9, _ = cb.bound_boxes(ctx, reg, lo9[None], hi9[None], mu9)
        counts["nine_boxes"] += 1
        for B, O in sample_points(ctx, reg, lo9, hi9, 400, B9=True):
            ev = np.linalg.eigvalsh(O)
            inK = ev[0] >= 0.05 and ev[-1] <= 10
            f = F_point(ctx.S, B, O, lam, lam)
            counts["nine_points"] += 1
            if st9[0] == 1 and inK:
                viol["nine_infeasible"] += 1
            if st9[0] == 2 and f < s_star - 1e-12:
                viol["nine_kl"] += 1
            if st9[0] == 0 and inK and f < lb9[0] - 1e-12:
                viol["nine_bound"] += 1
            if st9[0] == 0 and inK:
                worst["nine"] = min(worst["nine"], f - lb9[0])
out["sampling_counts"] = counts
out["sampling_violations"] = viol
out["sampling_min_F_minus_bound"] = worst
out["sampling_ok"] = all(v == 0 for v in viol.values())

# ------------------------------------------------------------------ C. two-sided
two = []
for k in range(len(z["S"])):
    s_star, sel = cc.s_star_of(k)
    n = int(z["n"][k])
    lam = math.log(n) / n
    B, O = z["B"][k][sel], z["O"][k][sel]
    f = F_point(z["S"][k], B, O, lam, lam)
    two.append(f - s_star)
out["refit_criterion_minus_s_star_max_abs"] = float(np.max(np.abs(two)))
out["runtime_s"] = time.monotonic() - t0
print(json.dumps(out, indent=1, default=str))
(ROOT / "results" / "validation" / "criterion_bounds.json").write_text(json.dumps(out, indent=1, default=str) + "\n")

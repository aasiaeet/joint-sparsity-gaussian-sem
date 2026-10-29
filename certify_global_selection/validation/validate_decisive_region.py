"""Validation of the criterion bounds at the decisive region of a sample.

For a sample, the decisive region is (pi, L) with H(pi, L) = the selected support H^ and pi a
vertex order compatible with it; the refit theta^ lies in it and F(theta^) = s*.
A. Negative control: with target s* + 1e-6 and s* + 1e-3 the decisive region must not be
   certified (budgets: 400,000 B-space boxes, then 1,000,000 9-D boxes); with the true
   target it must be certified.
B. Soundness near the optimum: boxes centered at theta^ with relative widths 1e-4..1e-1,
   and 300 points per box drawn near theta^ (B in the box; Omega = Omega^ plus perturbations
   of size 1e-5..1e-2; small coordinates uniform in a sub-interval of [-delta', delta']):
     B-space boxes: F >= min(s*, lb) - 1e-12;   9-D boxes: F >= lb - 1e-12 for points in K.
   The minimum slack F - bound is reported (it should be small but nonnegative).
Samples: 0 (G1 n=100), 20 (G2 n=100, non-minimum-edge selection), 50 (G3 n=100,
underfit), 36 (G2 n=400), 85 (G3 n=6400).
Usage: python3 validate_decisive_region.py [k1,k2,... [suffix]]
Writes results/validation/decisive_region{suffix}.json.
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
from criterion_bounds import Ctx, PAIRS, certify_region  # noqa: E402
from criterion_bounds_bspace import certify_region_B  # noqa: E402
from ia import IA  # noqa: E402
import global_selection as gs  # noqa: E402

rng = np.random.default_rng(161803)
z, a = cc.load()
out = dict(samples={})
t0 = time.monotonic()


def ctx_for(k, target_shift=0.0):
    s_star, sel = cc.s_star_of(k)
    aa = a[a["sample"] == k]
    lbp = {(gs.SUPPORTS[int(r.support_index)].D, gs.SUPPORTS[int(r.support_index)].U): float(r.lb)
           for _, r in aa.iterrows()}
    return Ctx(z["S"][k], int(z["n"][k]), s_star + target_shift, lbp), s_star, sel


def decisive_region(ctx, sup):
    order = sup.orders[0]
    fwd = [(order[0], order[1]), (order[0], order[2]), (order[1], order[2])]
    L = tuple(sorted([c for c, e in enumerate(fwd) if e in sup.D] + [3 + PAIRS.index(u) for u in sup.U]))
    return cb.Region(ctx, order, L)


def mus_for(ctx, reg):
    tau = ctx.s_star - float((ctx.lam * float(len(reg.L))).lo) - float(ctx.fsat.lo) - 1e-12
    if tau <= 1e-12:
        return tau, None, None, []
    mu_lo, mu_hi = ctx.mu_bounds(tau)
    mus = []
    for mu in sorted({mu_hi, min(1.9, mu_hi * 1.05), min(1.9, max(mu_hi, 1.3)), 1.6, 1.9}):
        if mu >= mu_hi:
            MH = IA.point(np.array(mu))
            mR = ((IA.point(np.array(1.0)) / MH - 0.5) / MH.sqr()).lo
            if mR > 0:
                mus.append((mR, mu))
    return tau, mu_lo, mu_hi, mus


SAMPLES = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0, 20, 50, 36, 85]
TAG = sys.argv[2] if len(sys.argv) > 2 else ''
for k in SAMPLES:
    res = {}
    ctx, s_star, sel = ctx_for(k)
    sup = gs.SUPPORTS[sel]
    reg = decisive_region(ctx, sup)
    res["selected"] = sup.name
    res["region"] = ("".join(str(v + 1) for v in reg.order), reg.L)
    # ---------------- A. negative controls on the decisive region
    for shift in [0.0, 1e-6, 1e-3]:
        c2, _, _ = ctx_for(k, shift)
        r2 = cb.Region(c2, reg.order, reg.L)
        rB = certify_region_B(c2, r2, max_boxes=400_000)
        cert = rB is not None and rB["certified"]
        if not cert:
            r9 = certify_region(c2, r2, max_boxes=1_000_000)
            cert = r9["certified"]
        res[f"certified_with_target_shift_{shift:g}"] = bool(cert)
    # ---------------- B. soundness near the optimum
    lam = math.log(ctx.n) / ctx.n
    Bh, Oh = z["B"][k][sel], z["O"][k][sel]
    xh = np.r_[[Bh[e] for e in reg.fwd], [Oh[p] for p in PAIRS], np.diag(Oh)]
    tau, mu_lo, mu_hi, mus = mus_for(ctx, reg)
    slack_b, slack_9, viol_b, viol_9, npts_b, npts_9, nin_9 = math.inf, math.inf, 0, 0, 0, 0, 0
    dp = ctx.delta_hi
    for trial in range(30):
        w = 10 ** rng.uniform(-4, -1)
        tw = dp * 10 ** rng.uniform(-3, 0)
        # B-space box
        lo = np.array([max(-2, xh[c] - w) if c in reg.L else -tw for c in range(3)])
        hi = np.array([min(2, xh[c] + w) if c in reg.L else tw for c in range(3)])
        lbs = []
        if 1e-12 < tau < 0.0966 and mus:
            pr = cbb.prune_B(ctx, reg, lo[None], hi[None], mu_lo, mu_hi)[0]
            lbs.append(cbb.bound_P(ctx, reg, lo[None], hi[None], mus)[0])
            lbs.append(cbb.bound_B_multi(ctx, reg, lo[None], hi[None], mus)[0][0])
            lbs.append(cbb.graded_bound_B(ctx, reg, lo[None], hi[None], mu_hi)[0])
            lbB = max(lbs)
            for _ in range(300):
                x = lo + rng.uniform(size=3) * (hi - lo)
                B = np.zeros((3, 3))
                for c, (i, j) in enumerate(reg.fwd):
                    B[i, j] = x[c]
                E = rng.standard_normal((3, 3)) * 10 ** rng.uniform(-5, -2)
                O = Oh + (E + E.T) / 2
                for c in range(3, 6):
                    if c not in reg.L:
                        p, q = PAIRS[c - 3]
                        O[p, q] = O[q, p] = rng.uniform(-tw, tw)
                if np.linalg.eigvalsh(O)[0] <= 0:
                    continue
                f = gs.criterion(B, O, ctx.S, lam, lam)
                npts_b += 1
                bound = s_star if pr else min(s_star, lbB)
                slack_b = min(slack_b, f - bound)
                viol_b += f < bound - 1e-12
        # 9-D box around the refit
        lo0, hi0 = reg.initial_box()
        lo9 = np.maximum(lo0, xh - w * np.maximum(np.abs(xh), 0.1))
        hi9 = np.minimum(hi0, xh + w * np.maximum(np.abs(xh), 0.1))
        for c in reg.tdims:
            lo9[c], hi9[c] = -tw, tw
        tau9 = ctx.s_star - float((ctx.lam * float(len(reg.L))).lo) - float(ctx.fsat.lo) - 1e-12
        mu9 = ctx.mu_bounds(max(tau9, 0.0))
        lb9, st9, _ = cb.bound_boxes(ctx, reg, lo9[None], hi9[None], mu9)
        for _ in range(300):
            x = lo9 + rng.uniform(size=9) * (hi9 - lo9)
            B = np.zeros((3, 3))
            for c, (i, j) in enumerate(reg.fwd):
                B[i, j] = x[c]
            O = np.diag(x[6:])
            for c, (i, j) in enumerate(PAIRS):
                O[i, j] = O[j, i] = x[3 + c]
            ev = np.linalg.eigvalsh(O)
            npts_9 += 1
            if ev[0] < 0.05 or ev[-1] > 10:
                continue
            nin_9 += 1
            f = gs.criterion(B, O, ctx.S, lam, lam)
            if st9[0] == 0:
                slack_9 = min(slack_9, f - lb9[0])
                viol_9 += f < lb9[0] - 1e-12
            elif st9[0] == 2:
                viol_9 += f < s_star - 1e-12
            elif st9[0] == 1:
                viol_9 += 1
    res.update(bspace_points=npts_b, bspace_violations=int(viol_b), bspace_min_slack=slack_b,
               nineD_points=npts_9, nineD_points_in_K=nin_9, nineD_violations=int(viol_9), nineD_min_slack=slack_9)
    out["samples"][str(k)] = res
    print(k, json.dumps(res, default=str), flush=True)
out["negative_controls_ok"] = all((v["certified_with_target_shift_0"] and not v["certified_with_target_shift_1e-06"]
                                   and not v["certified_with_target_shift_0.001"]) for v in out["samples"].values())
out["soundness_ok"] = all(v["bspace_violations"] == 0 and v["nineD_violations"] == 0 for v in out["samples"].values())
out["runtime_s"] = time.monotonic() - t0
print(json.dumps(out, indent=1, default=str))
(ROOT / "results" / "validation" / f"decisive_region{TAG}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")

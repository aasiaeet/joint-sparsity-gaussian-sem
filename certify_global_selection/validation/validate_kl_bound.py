"""Soundness check of the joint KL bound (quad_kl_bound) and the graded pair bound in the
regions where they decide: samples 2, 3 (G1, n=100) and 21 (G2, n=100), bowed supports,
B near the flat end of the bowed coordinate.  For random B-boxes (widths 1e-4..1e-1) and 400
random region points per box (B in the box, Omega = R(B)-based random positive definite
matrices with the small pairs uniform in [-delta', delta']), checks F >= min(s*, bound) - 1e-12.
Writes results/validation/kl_bound.json."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
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
from criterion_bounds import Ctx, PAIRS  # noqa: E402
import global_selection as gs  # noqa: E402

rng = np.random.default_rng(141421)
z, a = cc.load()
cases = [(2, (0, 2, 1), (2, 5)), (2, (2, 0, 1), (1, 5)), (3, (0, 2, 1), (2, 5)), (21, (0, 1, 2), (0, 3))]
out = {}
for k, order, L in cases:
    s_star, sel = cc.s_star_of(k)
    aa = a[a["sample"] == k]
    lbp = {(gs.SUPPORTS[int(r.support_index)].D, gs.SUPPORTS[int(r.support_index)].U): float(r.lb) for _, r in aa.iterrows()}
    ctx = Ctx(z["S"][k], int(z["n"][k]), s_star, lbp)
    reg = cb.Region(ctx, order, L)
    tau = ctx.s_star - float((ctx.lam * float(len(L))).lo) - float(ctx.fsat.lo) - 1e-12
    mu_lo, mu_hi = ctx.mu_bounds(tau)
    lam = math.log(ctx.n) / ctx.n
    dp = ctx.delta_hi
    viol = npts = 0
    slack = math.inf
    nbind = 0
    for trial in range(60):
        cen = np.array([rng.choice([-1.99, -1.9, -1.5, 0.0, 1.0, 1.9]) + rng.uniform(-0.05, 0.05) if c in L else 0.0
                        for c in range(3)])
        w = 10 ** rng.uniform(-4, -1)
        lo = np.array([max(-2, cen[c] - w) if c in L else -dp * rng.uniform(0.01, 1) for c in range(3)])
        hi = np.array([min(2, cen[c] + w) if c in L else -lo[c] for c in range(3)])
        g = cbb.graded_bound_B(ctx, reg, lo[None], hi[None], mu_hi)[0]
        bound = min(s_star, g)
        nbind += g >= ctx.target
        for _ in range(400):
            x = lo + rng.uniform(size=3) * (hi - lo)
            B = np.zeros((3, 3))
            for c, (i, j) in enumerate(reg.fwd):
                B[i, j] = x[c]
            A = np.eye(3) - B
            R = A.T @ ctx.S @ A
            E = rng.standard_normal((3, 3)) * 10 ** rng.uniform(-3, -0.3)
            O = R + (E + E.T) / 2
            for c in range(3, 6):
                if c not in L:
                    p, q = PAIRS[c - 3]
                    O[p, q] = O[q, p] = rng.uniform(-dp, dp)
            if np.linalg.eigvalsh(O)[0] <= 1e-8:
                continue
            f = gs.criterion(B, O, ctx.S, lam, lam)
            npts += 1
            slack = min(slack, f - bound)
            viol += f < bound - 1e-12
    out[f"sample{k}_order{''.join(str(v + 1) for v in order)}_L{L}"] = dict(points=npts, violations=int(viol),
                                                                          min_slack=slack, boxes_certified_by_bound=int(nbind))
    print(k, order, L, out[f"sample{k}_order{''.join(str(v + 1) for v in order)}_L{L}"], flush=True)
out["ok"] = all(v["violations"] == 0 for v in out.values() if isinstance(v, dict))
(ROOT / "results" / "validation" / "kl_bound.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(json.dumps(out, indent=1, default=str))

"""Tightness and soundness of the joint KL bound against numerical inner minima: for fixed B
(point boxes), minimize F(B, Omega) over Omega (small pairs in [-delta', delta'], L-BFGS-B,
8 starts) and compare with the bound min(s*, graded_bound_B).  Writes
results/validation/kl_bound_inner.json."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json, math, sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "global_selection"))
warnings.filterwarnings("ignore")
import certify_criterion as cc
import criterion_bounds as cb
import criterion_bounds_bspace as cbb
from criterion_bounds import Ctx, PAIRS
import global_selection as gs
rng = np.random.default_rng(1732)
z, a = cc.load()
cases = [(2, (0, 2, 1), (2, 5)), (3, (0, 2, 1), (2, 5)), (21, (0, 1, 2), (0, 3)), (50, (0, 1, 2), (0, 4))]
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
    worst = math.inf; viol = 0; rows = []
    for b in np.linspace(-2, 2, 21):
        x = np.array([b if c in L else 0.0 for c in range(3)])
        B = np.zeros((3, 3))
        for c, (i, j) in enumerate(reg.fwd):
            B[i, j] = x[c]
        g = cbb.graded_bound_B(ctx, reg, x[None], x[None], mu_hi)[0]
        A = np.eye(3) - B; R = A.T @ ctx.S @ A
        def F(v):
            O = np.diag(v[:3]).astype(float)
            for c, (i, j) in enumerate(PAIRS):
                O[i, j] = O[j, i] = v[3 + c]
            if np.linalg.eigvalsh(O)[0] <= 1e-9:
                return 1e10
            return gs.criterion(B, O, ctx.S, lam, lam)
        bnds = [(1e-3, 50)] * 3 + [((-dp, dp) if (3 + c) not in L else (-50, 50)) for c in range(3)]
        best = math.inf
        for s in range(8):
            v0 = np.r_[np.diag(R) * rng.uniform(0.7, 1.3, 3), [np.clip(R[i, j] * rng.uniform(0, 1), *bnds[3 + c]) for c, (i, j) in enumerate(PAIRS)]]
            r = minimize(F, v0, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=3000))
            best = min(best, r.fun)
        bound = min(s_star, g)
        worst = min(worst, best - bound)
        viol += best < bound - 1e-9
        rows.append((float(b), float(best - s_star), float(g - s_star)))
    key = f"sample{k}_order{''.join(str(v + 1) for v in order)}_L{L}"
    out[key] = dict(min_inner_minus_bound=worst, violations=int(viol),
                    rows_b_innerminus_s_bound_minus_s=rows)
    print(key, worst, viol, flush=True)
out["ok"] = all(v["violations"] == 0 for v in out.values() if isinstance(v, dict))
(ROOT / "results" / "validation" / "kl_bound_inner.json").write_text(json.dumps(out, indent=1, default=str) + "\n")

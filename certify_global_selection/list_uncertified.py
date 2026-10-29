"""Regions of the criterion bound that were not certified (samples 49 and 57, Section S18).

For each, report the profile margin of the support H(pi, L) and a numerical minimum of the
criterion over the region (floating point, multistart L-BFGS-B from the refits of every
support compatible with the vertex order; not certified).
Reads support_refits.npz, criterion_regions.csv.gz and criterion_samples.csv.
Writes results/uncertified_regions.csv.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "global_selection"))
warnings.filterwarnings("ignore")
import global_selection as gs  # noqa: E402

PAIRS = [(0, 1), (0, 2), (1, 2)]
RESULTS = HERE / "results"
z = np.load(RESULTS / "support_refits.npz")
regs = pd.read_csv(RESULTS / "criterion_regions.csv.gz", float_precision="round_trip", low_memory=False,
                   dtype={"order": str, "L": str}, keep_default_na=False)
s_star = pd.read_csv(RESULTS / "criterion_samples.csv", float_precision="round_trip").set_index("sample")["s_star"]
bad = [r for _, r in regs.iterrows() if str(r["certified"]) != "True"]
rows = []
rng = np.random.default_rng(7)
for r in bad:
    k = int(r["sample"])
    order = tuple(int(c) - 1 for c in r["order"])
    L = tuple(int(c) for c in r["L"].split(",")) if r["L"] else ()
    S = z["S"][k]; n = int(z["n"][k]); lam = math.log(n) / n
    fwd = [(order[0], order[1]), (order[0], order[2]), (order[1], order[2])]

    def unpack(x):
        B = np.zeros((3, 3))
        for c, e in enumerate(fwd):
            B[e] = x[c]
        O = np.diag(x[6:9]).astype(float)
        for c, (i, j) in enumerate(PAIRS):
            O[i, j] = O[j, i] = x[3 + c]
        return B, O

    def F(x):
        B, O = unpack(x)
        ev = np.linalg.eigvalsh(O)
        if ev[0] < 0.05 or ev[-1] > 10:
            return 1e6
        return gs.criterion(B, O, S, lam, lam)
    bnds = [((-2, 2) if c in L else (-lam, lam)) for c in range(3)] + \
           [((-10, 10) if 3 + c in L else (-lam, lam)) for c in range(3)] + [(0.05, 10)] * 3
    lo_b = np.array([b[0] for b in bnds]); hi_b = np.array([b[1] for b in bnds])
    best = math.inf
    for s in gs.SUPPORTS:
        if order not in s.orders:
            continue
        Bk, Ok = z["B"][k][s.index], z["O"][k][s.index]
        x0 = np.clip(np.r_[[Bk[e] for e in fwd], [Ok[p] for p in PAIRS], np.diag(Ok)], lo_b, hi_b)
        for rep in range(3):
            xs = np.clip(x0 + (rng.normal(0, 0.2, 9) if rep else 0), lo_b, hi_b)
            res = minimize(F, xs, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=4000, ftol=1e-15, gtol=1e-12))
            best = min(best, res.fun)
    H = gs.SUPPORTS[[s.index for s in gs.SUPPORTS if s.D == tuple(sorted(fwd[c] for c in L if c < 3))
                     and s.U == tuple(sorted(PAIRS[c - 3] for c in L if c >= 3))][0]].name
    rows.append(dict(sample=k, design=r["design"], n=int(r["n"]), rep=int(r["rep"]), order=r["order"], L=r["L"],
                     H=H, H_profile_margin=float(r["H_margin"]), numerical_region_min_minus_s_star=best - s_star[k],
                     boxes_all_stages=int(r["boxes_all_stages"]), wall_s=float(r["wall_s"]), stages=r["tried"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "uncertified_regions.csv", index=False)

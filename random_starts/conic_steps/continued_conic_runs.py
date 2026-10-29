"""The two continued runs that used the conic Omega-step after cycle 20,000
(G3, n = 1000, sample 24, refit start 48; G3, n = 250, sample 5, refit start 49).

Each run is continued as in ../continuation/continue_capped.py (fresh NoiseProx, cap
400,000) (a) with the conic Omega-step, recording solver statuses and spectral repairs, and
(b) with ExactProx from conic_steps.py (Dykstra projection accepted only with a KKT residual
<= 1e-9, conic solve otherwise).  In both runs the entrywise solution has an off-diagonal
entry inside (0, delta), where the penalty is not locally constant, so the closed-form exact
step does not apply and (b) coincides with (a).

Starts are read from ../continuation/results/starts_capped.npz when present and rebuilt
otherwise.  Each run takes several minutes.

Usage: python3 continued_conic_runs.py [--workers W] [--out DIR]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "continuation"))
import runner as ec  # noqa: E402
import continue_capped as cc  # noqa: E402
from conic_steps import ExactProx  # noqa: E402

KEYS = ["G3|1000|24|refit|48", "G3|250|5|refit|49"]
_STARTS = {}


def one(job):
    key, mode = job
    B0, O0, S = _STARTS[key]
    d = key.split("|")[0]
    cfg = ec.load_cfg(max_cycles=400000)
    pop = ec.load_population()[d]
    lo, hi = cfg["omega_spectrum"]
    cls = ec.NoiseProx if mode == "conic" else ExactProx
    prox = cls(ec.P, cfg["beta"], cfg["lam"], cfg["delta"], lo, hi)
    t0 = time.monotonic()
    b, o, info, snap = ec.run_alternating_snap(S, B0, O0, cfg, prox=prox)
    desc = ec.describe(b, o, cfg, set(pop["E_min_indices"]), pop["e_star"])
    return dict(key=key, mode=mode, cycles=info["cycles"], status=info["status"], objective=info["objective"],
                support_thr=desc["support_thr"], support_raw=desc["support_raw"], is_emin=desc["is_emin"],
                max_objective_increase=info["max_objective_increase"], min_omega_eig=info["min_omega_eig"],
                final_max_eig=float(np.linalg.eigvalsh(o)[-1]),
                prox_statuses=json.dumps(prox.statuses), max_kkt_exact=getattr(prox, "max_kkt", np.nan),
                runtime_seconds=time.monotonic() - t0)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, capped = cc.capped_runs()
    sel = capped[[cc.key_of(r) in KEYS for _, r in capped.iterrows()]]
    _STARTS.update(cc.load_starts(sel, args.workers, cc.OUT))
    jobs = [(k, m) for k in KEYS for m in ("conic", "exact")]
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(out / "continued_conic_runs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(df.T.to_string())


if __name__ == "__main__":
    main()

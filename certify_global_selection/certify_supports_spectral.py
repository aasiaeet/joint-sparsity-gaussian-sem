"""Second pass over the support fits not certified by the unconstrained profile bound.

These are fits at which the spectral constraint Omega <= 10 I is active.  For |U| <= 1 the
spectrally constrained profile (support_bounds_spectral.py) is the exact inner minimum; it
is minimized by branch and bound and the result replaces the lower bound.  For |U| = 2
there is no closed form under the spectral constraint; these fits stay uncertified, with
the certified interval [lb, ub] reported (Section S18).

Usage: python3 certify_supports_spectral.py [--dir DIR]
Reads and rewrites support_bounds.csv.gz (adds lb_final, certified_final and related columns).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "global_selection"))
warnings.filterwarnings("ignore")
import global_selection as gs  # noqa: E402
from support_bounds import TOL_CERT  # noqa: E402
from support_bounds_spectral import spectral_lower  # noqa: E402

RESULTS = HERE / "results"


def main():
    ap = argparse.ArgumentParser(description="Spectrally constrained pass over uncertified support fits.")
    ap.add_argument("--dir", type=str, default=str(RESULTS),
                    help="directory with support_bounds.csv.gz and support_refits.npz")
    a = ap.parse_args()
    out = Path(a.dir)
    d = pd.read_csv(out / "support_bounds.csv.gz", float_precision="round_trip")
    z = np.load(out / "support_refits.npz")
    d["lb_spectral"] = np.nan
    d["spectral_boxes"] = np.nan
    d["spectral_runtime_s"] = np.nan
    d["lb_final"] = d["lb"]
    d["certified_final"] = d["certified"]
    d["uncertified_reason"] = ""
    for idx in d.index[~d.certified]:
        r = d.loc[idx]
        sup = gs.SUPPORTS[int(r.support_index)]
        if len(sup.U) <= 1 and len(sup.D) >= 1:
            t0 = time.monotonic()
            res = spectral_lower(z["S"][int(r["sample"])], sup, ub0=float(r.ub))
            d.loc[idx, "lb_spectral"] = res["lb"]
            d.loc[idx, "spectral_boxes"] = res["boxes"]
            d.loc[idx, "spectral_runtime_s"] = time.monotonic() - t0
            lbf = max(float(r.lb), float(res["lb"]))
            d.loc[idx, "lb_final"] = lbf
            ok = (r.found - lbf <= TOL_CERT) and (r.ub - r.found <= TOL_CERT)
            d.loc[idx, "certified_final"] = ok
            if not ok:
                d.loc[idx, "uncertified_reason"] = "spectral constraint active; constrained-profile B&B did not reach 1e-8"
        else:
            d.loc[idx, "uncertified_reason"] = ("spectral constraint lambda_max(Omega) <= 10 active at the fit; "
                                                "|U| = 2 has no closed-form spectrally constrained profile; "
                                                "only the unconstrained-profile lower bound is certified")
    d["found_minus_lb_final"] = d["found"] - d["lb_final"]
    d.to_csv(out / "support_bounds.csv.gz", index=False)
    bad = d[~d.certified_final]
    print("certified", int(d.certified_final.sum()), "of", len(d))
    print(d[~d.certified][["sample", "design", "n", "rep", "support", "found", "lb", "lb_spectral", "lb_final", "ub",
                           "certified_final", "uncertified_reason"]].to_string())


if __name__ == "__main__":
    main()

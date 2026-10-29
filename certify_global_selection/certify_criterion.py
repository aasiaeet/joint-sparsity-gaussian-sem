"""Certified lower bound of the quasi-MCP criterion over K for each subset sample.

For a sample with selected l0 score s* = min_H l_min(H; S) + lambda_n |H| (Section 7.1,
lambda_n = delta_n = log(n)/n as in Theorem 9), certify that
    inf_K  l_S + lambda_n P_{delta_n}  >=  s* - 1e-8.
K is split into 6 x 64 regions by vertex order and by the set L of coordinates larger
than delta_n in magnitude (criterion_bounds.py); regions whose saturated-likelihood bound
already exceeds s* are screened out, and each of the remaining 132 regions is bounded by
three stages of branch and bound: the 9-dimensional bound with 60,000 boxes, the B-space
bound with an analytic inner bound over Omega (criterion_bounds_bspace.py), and the
9-dimensional bound with the full budget.

Usage: python3 certify_criterion.py [--workers W] [--only k1,...] [--skip k1,...] [--out DIR]
Reads support_refits.npz and support_bounds.csv.gz (profile lower bounds, column lb).
Writes criterion_regions.csv.gz (one row per region) and criterion_samples.csv (one row
per sample); rows of samples already present in these files are replaced.
Samples 49 and 57 are run region by region with certify_criterion_regions.py.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "global_selection"))
import global_selection as gs  # noqa: E402
from criterion_bounds import Ctx, certify_region, regions_for  # noqa: E402
from criterion_bounds_bspace import certify_region_B  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

RESULTS = HERE / "results"
INPUT = RESULTS
_Z = None
_A = None


def load():
    """Refits (support_refits.npz) and certified profile lower bounds (support_bounds.csv.gz)."""
    global _Z, _A
    if _Z is None:
        _Z = dict(np.load(INPUT / "support_refits.npz"))
        _A = pd.read_csv(INPUT / "support_bounds.csv.gz", float_precision="round_trip")
    return _Z, _A


def merge_rows(path, new, samples):
    """Replace the rows of `samples` in the table at `path` by `new` (a DataFrame)."""
    if path.exists():
        old = pd.read_csv(path, float_precision="round_trip", low_memory=False)
        new = pd.concat([old[~old["sample"].isin(samples)], new], ignore_index=True)
    new = new.sort_values("sample", kind="stable")
    new.to_csv(path, index=False)
    return new


def s_star_of(k):
    z, _ = load()
    n = int(z["n"][k])
    lam = math.log(n) / n
    vals = z["npz_value"][k]
    counts = np.array([s.count for s in gs.SUPPORTS])
    score = vals + lam * counts
    return float(score.min()), int(np.argmin(score))


def certify_sample(k, max_boxes=2_000_000, max_boxes_B=400_000):
    z, a = load()
    t0 = time.monotonic()
    s_star, sel = s_star_of(k)
    aa = a[a["sample"] == k]
    lbp = {}
    for _, r in aa.iterrows():
        sup = gs.SUPPORTS[int(r.support_index)]
        lbp[(sup.D, sup.U)] = float(r.lb)
    ctx = Ctx(z["S"][k], int(z["n"][k]), s_star, lbp)
    screened, pending = regions_for(ctx)
    rows = []
    for reg in pending:
        margin = ctx.lb_prof.get((reg.D, reg.U), -math.inf) + len(reg.L) * float(ctx.lam.lo) - s_star
        # stage 1: 9-D branch and bound with a small budget
        res = certify_region(ctx, reg, max_boxes=60_000)
        res["method"] = "9-D"
        tried = "9-D(60k)"
        boxes_used = res["boxes"]
        if not res["certified"]:
            # stage 2: B-space branch and bound
            rB = certify_region_B(ctx, reg, max_boxes=max_boxes_B)
            if rB is not None:
                tried += "+B-space"
                boxes_used += rB["boxes"]
                if rB["certified"]:
                    res = rB
        if not res["certified"]:
            # stage 3: 9-D with the full budget
            res = certify_region(ctx, reg, max_boxes=max_boxes)
            res["method"] = "9-D"
            tried += "+9-D(full)"
            boxes_used += res["boxes"]
        res["boxes_all_stages"] = boxes_used
        res.update(sample=k, design=str(z["design"][k]), n=int(z["n"][k]), rep=int(z["rep"][k]),
                   tried=tried, H_margin=margin)
        rows.append(res)
    screen_min = min(x[2] for x in screened) if screened else math.inf
    out = dict(sample=k, design=str(z["design"][k]), n=int(z["n"][k]), rep=int(z["rep"][k]),
               s_star=s_star, selected=gs.SUPPORTS[sel].name, regions_total=len(screened) + len(pending),
               regions_screened=len(screened), regions_bnb=len(pending),
               regions_certified=sum(r["certified"] for r in rows) + len(screened),
               all_certified=all(r["certified"] for r in rows),
               min_region_lb_minus_s_star=min([r["min_lb"] for r in rows if r["certified"]] + [screen_min]) - s_star,
               boxes=sum(r["boxes_all_stages"] for r in rows), runtime_s=time.monotonic() - t0)
    return rows, out


def main():
    global INPUT
    ap = argparse.ArgumentParser(description="Certify the criterion lower bound on the subset samples.")
    ap.add_argument("--only", type=str, default=None, help="comma-separated sample indices")
    ap.add_argument("--skip", type=str, default=None, help="comma-separated sample indices to leave out")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--max-boxes", type=int, default=2_000_000)
    ap.add_argument("--input", type=str, default=str(RESULTS),
                    help="directory with support_refits.npz and support_bounds.csv.gz")
    ap.add_argument("--out", type=str, default=str(RESULTS))
    a = ap.parse_args()
    INPUT = Path(a.input)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z, _ = load()
    ks = list(range(len(z["S"]))) if not a.only else [int(x) for x in a.only.split(",")]
    if a.skip:
        ks = [k for k in ks if k not in {int(x) for x in a.skip.split(",")}]
    # longest first: front-door and small-n samples are the slowest
    ks.sort(key=lambda k: (str(z["design"][k]) != "G3", int(z["n"][k]), k))
    import multiprocessing as mp
    t0 = time.monotonic()
    from functools import partial
    f = partial(certify_sample, max_boxes=a.max_boxes)
    if a.workers > 1 and len(ks) > 1:
        res = []
        with mp.get_context("fork").Pool(min(a.workers, len(ks))) as pool:
            for r in pool.imap_unordered(f, ks, chunksize=1):
                res.append(r)
                print(f"[{time.monotonic()-t0:7.0f}s] sample {r[1]['sample']} {r[1]['design']} n={r[1]['n']} "
                      f"rep={r[1]['rep']} certified={r[1]['all_certified']} boxes={r[1]['boxes']} "
                      f"t={r[1]['runtime_s']:.0f}s", flush=True)
    else:
        res = [f(k) for k in ks]
    rows = [r for rr, _ in res for r in rr]
    samples = sorted([o for _, o in res], key=lambda o: o["sample"])
    merge_rows(out / "criterion_regions.csv.gz", pd.DataFrame(rows), ks)
    sd = merge_rows(out / "criterion_samples.csv", pd.DataFrame(samples), ks)
    print(sd[sd["sample"].isin(ks)].to_string())
    print("wall", time.monotonic() - t0)


if __name__ == "__main__":
    main()

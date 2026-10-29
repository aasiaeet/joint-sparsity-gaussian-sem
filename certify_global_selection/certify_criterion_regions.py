"""Region-parallel version of certify_criterion.py for the slowest samples.

Same three stages per region (9-D with 60,000 boxes, B-space, 9-D with the full budget),
but one job per (sample, region), so that the regions of one sample run in parallel and
the stage budgets can be raised.  Used for samples 49 and 57 (front-door, n = 100), whose
selected support is saturated with three edges, so that tau = lambda for |L| = 2 and
2 lambda for |L| = 1; on 12 and 16 of their regions the bounds did not close (Section S18).

Usage: python3 certify_criterion_regions.py --only 49,57 [--workers W] [--maxB 3000000] [--max9 4000000]
Rows of these samples in criterion_regions.csv.gz and criterion_samples.csv are replaced.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
import certify_criterion as cc  # noqa: E402
from criterion_bounds import Ctx, certify_region, regions_for  # noqa: E402
from criterion_bounds_bspace import certify_region_B  # noqa: E402
import global_selection as gs  # noqa: E402

ARGS = None


def make_ctx(k):
    z, a = cc.load()
    s_star, sel = cc.s_star_of(k)
    aa = a[a["sample"] == k]
    lbp = {(gs.SUPPORTS[int(r.support_index)].D, gs.SUPPORTS[int(r.support_index)].U): float(r.lb)
           for _, r in aa.iterrows()}
    return Ctx(z["S"][k], int(z["n"][k]), s_star, lbp), s_star, sel


def one_region(t):
    k, idx = t
    z, _ = cc.load()
    ctx, s_star, sel = make_ctx(k)
    scr, pending = regions_for(ctx)
    reg = pending[idx]
    t0 = time.monotonic()
    margin = ctx.lb_prof.get((reg.D, reg.U), -math.inf) + len(reg.L) * float(ctx.lam.lo) - s_star
    res = certify_region(ctx, reg, max_boxes=60_000)
    res["method"] = "9-D"
    tried, used = "9-D(60k)", res["boxes"]
    if not res["certified"]:
        rB = certify_region_B(ctx, reg, max_boxes=ARGS.maxB)
        if rB is not None:
            tried += "+B-space"
            used += rB["boxes"]
            if rB["certified"]:
                res = rB
    if not res["certified"]:
        res = certify_region(ctx, reg, max_boxes=ARGS.max9)
        res["method"] = "9-D"
        tried += "+9-D(full)"
        used += res["boxes"]
    res.update(sample=k, design=str(z["design"][k]), n=int(z["n"][k]), rep=int(z["rep"][k]), tried=tried,
               H_margin=margin, boxes_all_stages=used, wall_s=time.monotonic() - t0,
               n_screened=len(scr), n_pending=len(pending), s_star=s_star, selected=gs.SUPPORTS[sel].name)
    return res


def sample_row(rr):
    """Per-sample summary of the region rows of one sample (as in certify_criterion.py)."""
    n_regions = rr[0]["n_pending"]
    cert = [r for r in rr if r["certified"]]
    return dict(sample=rr[0]["sample"], design=rr[0]["design"], n=rr[0]["n"], rep=rr[0]["rep"],
                s_star=rr[0]["s_star"], selected=rr[0]["selected"],
                regions_total=rr[0]["n_screened"] + n_regions,
                regions_screened=rr[0]["n_screened"], regions_bnb=n_regions,
                regions_certified=rr[0]["n_screened"] + len(cert),
                all_certified=bool(len(rr) == n_regions and len(cert) == n_regions),
                min_region_lb_minus_s_star=min([r["min_lb"] for r in cert] + [math.inf]) - rr[0]["s_star"],
                boxes=sum(r["boxes_all_stages"] for r in rr),
                runtime_s=sum(r["wall_s"] for r in rr),
                regions_finished=len(rr))


def main():
    global ARGS
    ap = argparse.ArgumentParser(description="Region-parallel certification of the criterion bound.")
    ap.add_argument("--only", required=True, help="comma-separated sample indices")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--maxB", type=int, default=3_000_000)
    ap.add_argument("--max9", type=int, default=4_000_000)
    ap.add_argument("--out", type=str, default=str(cc.RESULTS))
    ARGS = ap.parse_args()
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    ks = [int(x) for x in ARGS.only.split(",")]
    jobs = []
    for k in ks:
        ctx, _, _ = make_ctx(k)
        scr, pending = regions_for(ctx)
        jobs += [(k, i) for i in range(len(pending))]
    t0 = time.monotonic()
    rows = []
    with mp.get_context("fork").Pool(ARGS.workers) as pool:
        for r in pool.imap_unordered(one_region, jobs, chunksize=1):
            rows.append(r)
            if not r["certified"] or r["wall_s"] > 60:
                print(f"[{time.monotonic()-t0:6.0f}s] sample {r['sample']} region {r['order']} L={r['L']} "
                      f"certified={r['certified']} boxes={r['boxes_all_stages']} t={r['wall_s']:.0f}s {r['tried']}",
                      flush=True)
    samples = [sample_row([r for r in rows if r["sample"] == k]) for k in ks]
    cc.merge_rows(out / "criterion_regions.csv.gz", pd.DataFrame(rows), ks)
    cc.merge_rows(out / "criterion_samples.csv", pd.DataFrame(samples), ks)
    print(pd.DataFrame(samples).to_string())
    print("wall", time.monotonic() - t0, "regions", len(jobs), flush=True)


if __name__ == "__main__":
    main()

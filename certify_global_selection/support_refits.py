"""Constrained fits of all 200 supports on the 86 samples to be certified.

The subset is listed in ../global_selection/results/independent_check_certify_subset.csv:
the 16 chain and confounded-pair samples whose selected support is not a minimum-edge
support, 20 front-door samples that underfit, and 50 others (Section S18).  For each sample
the solver of ../global_selection/global_selection.py is rerun with the seeds of the
global-selection study: data SeedSequence([20260925, design_index, n, rep]) and solver
SeedSequence([20260925, design_index, n, rep, 1]).  The fitted values reproduce
../global_selection/results/lmin_all_datasets.npz exactly; the fit points (B, Omega) give
the certified upper bounds.

Usage: python3 support_refits.py [--workers W] [--only k1,k2,...] [--out DIR]
Writes support_refits.npz (not stored; rerun takes under a minute) and subset.csv.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "global_selection"))
import global_selection as gs  # noqa: E402

GS_RESULTS = HERE.parent / "global_selection" / "results"
RESULTS = HERE / "results"
DI = {"G1": 0, "G2": 1, "G3": 2}


def subset():
    c = pd.read_csv(GS_RESULTS / "independent_check_certify_subset.csv")
    r = pd.read_csv(GS_RESULTS / "runs.csv.gz", float_precision="round_trip")
    m = c[["design", "n", "rep"]].merge(r, left_on=["design", "n", "rep"],
                                         right_on=["design", "n", "replication"])
    m = m.sort_values(["design", "n", "rep"]).reset_index(drop=True)
    return m


def sample_S(design, n, rep):
    di = DI[design]
    rng = np.random.default_rng(np.random.SeedSequence([gs.CONFIG["master_seed"], di, n, rep]))
    Bt, Ot = gs.design_parameters(design)
    return gs.sample_covariance(gs.covariance(Bt, Ot), n, rng)


def one(job):
    design, n, rep = job
    t0 = time.monotonic()
    di = DI[design]
    S = sample_S(design, n, rep)
    rng = np.random.default_rng(np.random.SeedSequence([gs.CONFIG["master_seed"], di, n, rep, 1]))
    fits, repairs = gs.lmin_all(S, rng)
    return dict(design=design, n=n, rep=rep, S=S,
                value=np.array([f.value for f in fits]),
                B=np.array([f.B for f in fits]), O=np.array([f.O for f in fits]),
                method=[f.method for f in fits], repairs=len(repairs),
                runtime=time.monotonic() - t0)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description="Refit all supports on the certification subset.")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only", type=str, default=None, help="comma-separated sample indices")
    ap.add_argument("--out", type=str, default=str(RESULTS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = subset()
    if a.only:
        m = m.iloc[[int(x) for x in a.only.split(",")]]
    jobs = [(d, int(n), int(r)) for d, n, r in zip(m.design, m.n, m.rep)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=1)
    npz = np.load(GS_RESULTS / "lmin_all_datasets.npz")
    names = [str(x) for x in npz["support_names"]]
    assert names == [s.name for s in gs.SUPPORTS]
    idx = {(str(d), int(n), int(x)): i for i, (d, n, x) in
           enumerate(zip(npz["design"], npz["n"], npz["replication"]))}
    rows = []
    for o, (_, row) in zip(res, m.iterrows()):
        ref = npz["lmin"][idx[(o["design"], o["n"], o["rep"])]]
        diff = float(np.max(np.abs(o["value"] - ref)))
        ev = np.linalg.eigvalsh(o["S"])[0]
        rows.append(dict(design=o["design"], n=o["n"], rep=o["rep"],
                         lambda_n=row.lambda_n, l0_min_score=row.l0_min_score,
                         selected_support=row.selected_support, selected_is_emin=bool(row.selected_is_emin),
                         thresholded_is_emin=bool(row.thresholded_is_emin),
                         S_min_eig=ev, S_min_eig_match=bool(ev == row.sample_min_eig),
                         max_abs_value_diff_vs_stored=diff, repairs=o["repairs"],
                         n_joint_slsqp=sum(mm.startswith("joint") for mm in o["method"]),
                         refit_runtime_s=o["runtime"]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "subset.csv", index=False)
    np.savez_compressed(out / "support_refits.npz",
                        design=np.array([o["design"] for o in res]), n=np.array([o["n"] for o in res]),
                        rep=np.array([o["rep"] for o in res]), S=np.array([o["S"] for o in res]),
                        value=np.array([o["value"] for o in res]), B=np.array([o["B"] for o in res]),
                        O=np.array([o["O"] for o in res]),
                        method=np.array([o["method"] for o in res]),
                        npz_value=np.array([npz["lmin"][idx[(o["design"], o["n"], o["rep"])]] for o in res]),
                        support_names=np.array(names))
    print("all S match:", df.S_min_eig_match.all(), "max diff:", df.max_abs_value_diff_vs_stored.max())


if __name__ == "__main__":
    main()

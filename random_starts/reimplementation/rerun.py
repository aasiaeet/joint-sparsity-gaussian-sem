"""Rerun the random and deterministic starts of the random-start study with cycle.py.

Usage: python3 rerun.py TAG DESIGNS NS A:B [--small] [--workers W]
  e.g. python3 rerun.py full G1,G2,G3 250,1000,4000 0:100
       python3 rerun.py small G1,G2,G3 250,1000,4000 0:30 --small
Runs the deterministic start and random starts 1-20 of samples A, ..., B-1 of each cell
(--small: starts ten times smaller).  Writes results/rerun_TAG.csv and
results/rerun_TAG_endpoints.npz (final B and Omega).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import cycle as m

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def run_one(t):
    d, n, ds, k, small = t
    S = m.sample_S(d, n, ds)
    if k == 0:
        B0, O0, ex = np.zeros((3, 3)), np.diag(np.diag(S)), dict(start_order="", start_omega_scale=np.nan)
        kind = "deterministic"
    else:
        sd = 0.05 if small else 0.5
        B0, O0, ex = m.random_start(S, d, n, ds, k, B_sd=sd, G_sd=sd)
        kind = "random"
    t0 = time.monotonic()
    B, O, info = m.run(S, B0, O0)
    row = dict(design=d, n=n, dataset=ds, start_type=kind, start=k, **ex, **info,
               support_raw=m.support_name(B, O, 0.0), support_thr=m.support_name(B, O, m.DELTA),
               support_half=m.support_name(B, O, m.DELTA / 2), runtime=time.monotonic() - t0,
               B0=B0.ravel().tolist(), O0=O0.ravel().tolist())
    return row, B, O


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("designs")
    ap.add_argument("ns")
    ap.add_argument("datasets", help="A:B")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    designs = args.designs.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    a, b = (int(x) for x in args.datasets.split(":"))
    jobs = [(d, n, ds, k, args.small) for d in designs for n in ns for ds in range(a, b) for k in range(21)]
    OUT.mkdir(exist_ok=True)
    t0 = time.monotonic()
    rows, Bs, Os = [], [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (row, B, O) in enumerate(pool.imap(run_one, jobs, chunksize=2)):
            rows.append(row)
            Bs.append(B)
            Os.append(O)
            if i % 500 == 0:
                print(f"{i+1}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["B0", "O0"]).to_csv(OUT / f"rerun_{args.tag}.csv", index=False)
    np.savez_compressed(OUT / f"rerun_{args.tag}_endpoints.npz", B=np.array(Bs), O=np.array(Os),
                        B0=np.array([r["B0"] for r in rows]), O0=np.array([r["O0"] for r in rows]),
                        key=np.array([f"{r['design']}|{r['n']}|{r['dataset']}|{r['start']}" for r in rows]))
    print("done", len(rows), f"{time.monotonic()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

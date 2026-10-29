"""Continue the runs of the random-start study that stopped at the 20,000-cycle cap.

Supplementary Section S18 (Anomalies).  Each of the 1,805 capped runs is rerun from its
start (same data, start, tuning and steps; the trajectory is deterministic) with a cap of
400,000 cycles and the stopping rule of the study (Frobenius length of a full cycle below
1e-8).  The state at cycle 20,000 is recorded and compared with the cap-time row of
../results/runs.csv.gz.

Usage:
  python3 continue_capped.py --starts [--workers W]   rebuild every start and compare it
                                                      with the recorded start
  python3 continue_capped.py --check [--workers W]    10 capped runs to cycle 20,000 only
  python3 continue_capped.py --run [--workers W]      all capped runs, cap 400,000
Options --out DIR (default results/) and --limit N (first N capped runs, for a fast check).

Seeds: data SeedSequence([20260925, design_index, n, dataset]); random starts
SeedSequence([20260925, design_index, n, dataset, 2, start]); refit solver
SeedSequence([20260925, design_index, n, dataset, 1]).  The 10 runs of --check are drawn
with numpy default_rng(20261022) (see choose_check).
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
import runner as ec  # noqa: E402

NEW_CAP = 400000
OLD_CAP = 20000
CHECK_SEED = 20261022
OUT = HERE / "results"
PARTIAL_NAME = "runs_continued.jsonl"


def capped_runs():
    r = pd.read_csv(ec.STUDY_RESULTS / "runs.csv.gz")
    c = r[r.status == "cycle_cap"].copy()
    c = c.sort_values(["design", "n", "dataset", "start_type", "start"]).reset_index(drop=True)
    return r, c


def key_of(row):
    return f"{row['design']}|{int(row['n'])}|{int(row['dataset'])}|{row['start_type']}|{int(row['start'])}"


# ----------------------------------------------------------------------------
# Starts
# ----------------------------------------------------------------------------

def _starts_for_dataset(job):
    d, n, rep, items = job
    cfg = ec.load_cfg()
    S = ec.sample_S(cfg, d, n, rep)
    out = []
    refits = None
    for st, k in items:
        if st == "random":
            B0, O0, ex = ec.random_start(cfg, S, d, n, rep, k)
            out.append(dict(key=f"{d}|{n}|{rep}|{st}|{k}", B0=B0, O0=O0, start_order=ex["start_order"],
                            start_omega_scale=ex["start_omega_scale"], start_B_clipped=ex["start_B_clipped"]))
        else:
            if refits is None:
                refits = ec.refit_starts(cfg, S, d, n, rep)
            B0, O0, sup, crit = refits[k]
            out.append(dict(key=f"{d}|{n}|{rep}|{st}|{k}", B0=B0, O0=O0, start_support=sup,
                            start_criterion=crit))
    for x in out:
        x["start_objective_recomputed"] = float(ec.rs.criterion(x["B0"], x["O0"], S, cfg["lam"], cfg["delta"]))
        x["S"] = S
    return out


def build_starts(c, workers):
    """Rebuild the starts of the runs in table c; returns {key: start record}."""
    import multiprocessing as mp
    t0 = time.monotonic()
    groups = {}
    for _, row in c.iterrows():
        groups.setdefault((row.design, int(row.n), int(row.dataset)), []).append((row.start_type, int(row.start)))
    jobs = [(d, n, rep, items) for (d, n, rep), items in groups.items()]
    # datasets that need the constrained fits of all supports first (slow)
    jobs.sort(key=lambda t: (not any(st == "refit" for st, _ in t[3]),))
    res = []
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for k, out in enumerate(pool.imap_unordered(_starts_for_dataset, jobs, chunksize=1)):
            res.extend(out)
            if k % 50 == 0:
                print(f"starts {k+1}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)
    return {x["key"]: x for x in res}


def phase_starts(c, workers, out):
    t0 = time.monotonic()
    by = build_starts(c, workers)
    rows = []
    for _, row in c.iterrows():
        x = by[key_of(row)]
        chk = dict(key=key_of(row), start_type=row.start_type,
                   start_objective_saved=row.start_objective,
                   start_objective_recomputed=x["start_objective_recomputed"],
                   start_objective_absdiff=abs(row.start_objective - x["start_objective_recomputed"]))
        if row.start_type == "random":
            chk.update(start_order_match=str(int(row.start_order)) == x["start_order"],
                       start_omega_scale_absdiff=abs(row.start_omega_scale - x["start_omega_scale"]),
                       start_B_clipped_match=int(row.start_B_clipped) == x["start_B_clipped"])
        else:
            chk.update(start_support_match=row.start_support == x["start_support"],
                       start_criterion_absdiff=abs(row.start_criterion - x["start_criterion"]))
        rows.append(chk)
    df = pd.DataFrame(rows)
    df.to_csv(out / "starts_capped_check.csv.gz", index=False)
    keys = [key_of(row) for _, row in c.iterrows()]
    np.savez_compressed(out / "starts_capped.npz", key=np.array(keys),
                        B0=np.array([by[k]["B0"] for k in keys]),
                        O0=np.array([by[k]["O0"] for k in keys]), S=np.array([by[k]["S"] for k in keys]))
    print("starts done", len(keys), f"{time.monotonic()-t0:.0f}s")
    print("max |start objective diff|", df.start_objective_absdiff.max())
    rnd = df[df.start_type == "random"]
    ref = df[df.start_type == "refit"]
    if len(rnd):
        print("random: order mismatches", int((~rnd.start_order_match.astype(bool)).sum()),
              "max scale diff", rnd.start_omega_scale_absdiff.max(),
              "clip mismatches", int((~rnd.start_B_clipped_match.astype(bool)).sum()))
    if len(ref):
        print("refit: support mismatches", int((~ref.start_support_match.astype(bool)).sum()),
              "max criterion diff", ref.start_criterion_absdiff.max())


def load_starts(c, workers, out):
    """Starts from starts_capped.npz when it holds every key of c, otherwise rebuilt."""
    keys = [key_of(r) for _, r in c.iterrows()]
    path = out / "starts_capped.npz"
    if path.exists():
        z = np.load(path)
        have = {k: (B0, O0, S) for k, B0, O0, S in zip(z["key"], z["B0"], z["O0"], z["S"])}
        if all(k in have for k in keys):
            return have
    by = build_starts(c, workers)
    return {k: (by[k]["B0"], by[k]["O0"], by[k]["S"]) for k in keys}


# ----------------------------------------------------------------------------
# One continued run
# ----------------------------------------------------------------------------

_POP = None


def _pop():
    global _POP
    if _POP is None:
        _POP = ec.load_population()
    return _POP


def continue_one(job):
    key, B0, O0, S, cap = job
    d, n, rep, st, k = key.split("|")
    cfg = ec.load_cfg(max_cycles=cap)
    pop = _pop()[d]
    emin, e_star = set(pop["E_min_indices"]), pop["e_star"]
    b, o, info, snap = ec.run_alternating_snap(S, B0, O0, cfg, snap_at=OLD_CAP)
    row = dict(design=d, n=int(n), dataset=int(rep), start_type=st, start=int(k), cap=cap)
    row.update({f"new_{a}": v for a, v in info.items()})
    row.update({f"new_{a}": v for a, v in ec.describe(b, o, cfg, emin, e_star).items()})
    if snap is not None:
        row["snap_found"] = True
        for a in ("objective", "final_step", "max_objective_increase", "max_curvature_ratio_B",
                  "max_curvature_ratio_Omega", "min_omega_eig", "noise_conic_calls", "seconds"):
            row[f"cap20k_{a}"] = snap[a]
        row.update({f"cap20k_{a}": v for a, v in ec.describe(snap["B"], snap["O"], cfg, emin, e_star).items()})
        row["cap20k_B"] = snap["B"].ravel().tolist()
        row["cap20k_O"] = snap["O"].ravel().tolist()
    else:
        row["snap_found"] = False
    row["final_B"] = b.ravel().tolist()
    row["final_O"] = o.ravel().tolist()
    return row


def compare_snapshot(row, old):
    """Compare the state of a rerun at cycle 20,000 with the recorded cap-time row."""
    out = dict(key=f"{row['design']}|{row['n']}|{row['dataset']}|{row['start_type']}|{row['start']}",
               snap_found=row["snap_found"])
    if not row["snap_found"]:
        out["reproduces"] = False
        return out
    out["objective_absdiff"] = abs(row["cap20k_objective"] - old["objective"])
    for a in ("support_thr", "support_raw", "support_half", "is_emin", "n_coords_in_(0,delta)"):
        out[f"{a}_match"] = str(row[f"cap20k_{a}"]) == str(old[a])
    for a in ("final_step", "max_objective_increase", "max_curvature_ratio_B", "max_curvature_ratio_Omega",
              "min_omega_eig", "min_nonzero_coord"):
        x, y = float(row[f"cap20k_{a}"]), float(old[a])
        out[f"{a}_reldiff"] = abs(x - y) / max(abs(y), 1e-300)
    out["start_objective_absdiff"] = abs(row["new_start_objective"] - old["start_objective"])
    out["reproduces"] = bool(out["objective_absdiff"] <= 1e-10 and out["support_thr_match"]
                             and out["support_raw_match"] and out["support_half_match"])
    return out


def choose_check(c):
    """One random-start run per (design, n) cell and one refit-start run drawn from all capped
    refit runs, with default_rng(20261022)."""
    rng = np.random.default_rng(CHECK_SEED)
    picks = []
    for d in ("G1", "G2", "G3"):
        for n in (250, 1000, 4000):
            sub = c[(c.design == d) & (c.n == n) & (c.start_type == "random")]
            picks.append(sub.index[rng.integers(len(sub))])
    sub = c[c.start_type == "refit"]
    picks.append(sub.index[rng.integers(len(sub))])
    return c.loc[picks]


def phase_check(c, workers, out, limit=None):
    import multiprocessing as mp
    t0 = time.monotonic()
    sel = choose_check(c)
    if limit is not None:
        sel = sel.iloc[:limit]
    starts = load_starts(sel, workers, out)
    jobs = [(key_of(r), *starts[key_of(r)], OLD_CAP) for _, r in sel.iterrows()]
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        rows = pool.map(continue_one, jobs, chunksize=1)
    res = []
    for row, (_, old) in zip(rows, sel.iterrows()):
        cmp_ = compare_snapshot(row, old)
        cmp_.update(new_status=row["new_status"], new_cycles=row["new_cycles"],
                    final_vs_saved_objective_absdiff=abs(row["new_objective"] - old["objective"]),
                    old_is_emin=bool(old["is_emin"]), runtime_s=row["new_runtime_seconds"])
        res.append(cmp_)
    df = pd.DataFrame(res)
    df.to_csv(out / "repro_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
    print("all reproduce:", bool(df.reproduces.all()), f"{time.monotonic()-t0:.0f}s")


# ----------------------------------------------------------------------------
# All capped runs to 400,000 cycles
# ----------------------------------------------------------------------------

def phase_run(c, workers, out):
    import multiprocessing as mp
    t0 = time.monotonic()
    starts = load_starts(c, workers, out)
    partial = out / PARTIAL_NAME
    done = set()
    if partial.exists():
        with open(partial) as f:
            for line in f:
                r = json.loads(line)
                done.add(f"{r['design']}|{r['n']}|{r['dataset']}|{r['start_type']}|{r['start']}")
    keys = [key_of(r) for _, r in c.iterrows()]
    pending = [k for k in keys if k not in done]
    # random runs first, interleaving cells
    pending.sort(key=lambda k: (k.split("|")[3] != "random", int(k.split("|")[2]), k))
    print(f"capped runs {len(keys)}, done {len(done)}, pending {len(pending)}, workers {workers}", flush=True)
    jobs = [(k, *starts[k], NEW_CAP) for k in pending]
    with mp.get_context("fork").Pool(workers) as pool, open(partial, "a") as fout:
        for i, row in enumerate(pool.imap_unordered(continue_one, jobs, chunksize=1)):
            fout.write(json.dumps(row, default=float) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(jobs) - 1:
                print(f"{len(done)+i+1}/{len(keys)} {time.monotonic()-t0:.0f}s", flush=True)
    print("continuation done", f"{time.monotonic()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--starts", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=None, help="first N capped runs only (fast check)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, c = capped_runs()
    if args.starts:
        phase_starts(c if args.limit is None else c.iloc[:args.limit], args.workers, out)
    if args.check:
        phase_check(c, args.workers, out, args.limit)
    if args.run:
        phase_run(c if args.limit is None else c.iloc[:args.limit], args.workers, out)


if __name__ == "__main__":
    main()

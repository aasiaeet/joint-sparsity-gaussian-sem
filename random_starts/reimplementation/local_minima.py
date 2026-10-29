"""Are the end points of failed random starts local minima of l_S + lambda P_delta?

Supplementary Section S18 (Independent reimplementation).  For every failed random-start end
point of a rerun by rerun.py (not a minimum-edge support at threshold delta):
  * edge count, bowedness, smallest nonzero |coordinate| (the penalty is flat iff all >= delta);
  * likelihood gap to the saturated value 0.5 log det S + 3/2 (exact fit of S);
  * first-order conditions: gradient of l on the free coordinates of the support; for
    addable zero coordinates (acyclic B additions and Omega pairs), |d l| / (2 lambda/delta);
  * restricted Hessian eigenvalues on the support (finite differences of the analytic gradient);
  * random perturbations (all support coordinates, Omega diagonal, and addable zero
    coordinates along a random compatible order), radius below the saturation margin;
  * deletion and refit: for each edge, the constrained MLE on the support without it,
    criterion evaluated with the full quasi-MCP at the refit;
  * deletion path: for the edge whose deletion lowers the criterion most, trace x_e -> 0 with the
    other support coordinates re-optimized locally under the full criterion; report the barrier
    max_t F(t) - F(end).

Usage: python3 local_minima.py TAG [MAX_SAMPLE] [--workers W]
  e.g. python3 local_minima.py full 10    (samples 0-9 of every cell, as reported)
Writes results/local_minima_TAG[_dsMAX].csv and .txt.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import cycle as m
import fits as mf

HERE = Path(__file__).resolve().parent
POPULATION = HERE.parent.parent / "global_selection" / "results" / "population.json"
OUT = HERE / "results"
pop = json.loads(POPULATION.read_text())["designs"]
EMIN = {d: set(v["E_min_names"]) for d, v in pop.items()}
SLOPE = 2 * m.LAM / m.DELTA


def coords_of(B, O, tol=0.0):
    D = [e for e in m.DIRECTED if abs(B[e]) > tol]
    U = [e for e in m.PAIRS if abs(O[e]) > tol]
    return D, U


def analyze(args):
    row, B, O = args
    d, n, ds = row["design"], int(row["n"]), int(row["dataset"])
    S = m.sample_S(d, n, ds)
    rng = np.random.default_rng(int(row["start"]) + 1000 * ds)
    D, U = coords_of(B, O)
    Fend = m.objective(B, O, S)
    lend = m.lik(B, O, S)
    lsat = 0.5 * np.linalg.slogdet(S)[1] + 1.5
    vals = [abs(B[e]) for e in D] + [abs(O[e]) for e in U]
    out = dict(design=d, n=n, dataset=ds, start=int(row["start"]), support=row["support_raw"],
               count=len(D) + len(U), bowed=any(tuple(sorted(e)) in U for e in D),
               min_abs=min(vals) if vals else np.nan, F_end=Fend, l_minus_lsat=lend - lsat,
               at_B_bound=bool(np.any(np.abs(B) >= m.MB - 1e-9)))
    names = pop[d]["E_min_names"]
    lE = min(mf.fit(S, *m.parse_support(nm), rng=rng, n_random=2)[0] for nm in (names if d != "G3" else names[:4]))
    out["l_minus_lEmin"] = lend - lE
    out["F_minus_FEmin_over_lam"] = (Fend - (lE + m.LAM * pop[d]["e_star"])) / m.LAM
    gB, gO = m.grads(B, O, S)
    par = mf.Param(D, U)
    g = par.grad_vec(gB, gO)
    out["max_grad_support"] = float(np.max(np.abs(g)))
    addB = [e for e in m.DIRECTED if e not in D and m.acyclic(D + [e])]
    zU = [e for e in m.PAIRS if e not in U]
    out["n_addable_B"] = len(addB)
    out["max_offsupport_ratio_B"] = float(max([abs(gB[e]) / SLOPE for e in addB], default=0.0))
    out["max_offsupport_ratio_O"] = float(max([abs(2 * gO[e]) / SLOPE for e in zU], default=0.0))
    # restricted Hessian
    x = par.pack(B, O)
    h = 1e-6
    H = np.zeros((par.nx, par.nx))
    for k in range(par.nx):
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        gp = par.grad_vec(*m.grads(*par.unpack(xp), S))
        gm = par.grad_vec(*m.grads(*par.unpack(xm), S))
        H[:, k] = (gp - gm) / (2 * h)
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    out["hess_min_eig"] = float(ev[0])
    out["hess_n_flat"] = int(np.sum(np.abs(ev) < 1e-5))
    # perturbation test
    margin = (min(vals) - m.DELTA) if vals else 1.0
    worst = np.inf
    for r in (1e-5, 1e-4, 1e-3):
        if r >= margin / 2:
            continue
        for _ in range(1000):
            od = [o for o in m.ORDERS if all(o.index(i) < o.index(j) for i, j in D)]
            o = od[rng.integers(len(od))]
            fwd = [e for e in m.FORWARD[o]]
            Bp, Op = B.copy(), O.copy()
            dirs = []
            for e in fwd:
                dirs.append(("B", e))
            for i in range(3):
                dirs.append(("Od", (i, i)))
            for e in m.PAIRS:
                dirs.append(("O", e))
            v = rng.normal(size=len(dirs))
            v *= r / np.linalg.norm(v)
            for (kind, e), dv in zip(dirs, v):
                if kind == "B":
                    Bp[e] = np.clip(Bp[e] + dv, -m.MB, m.MB)
                elif kind == "Od":
                    Op[e] += dv
                else:
                    Op[e] += dv
                    Op[e[::-1]] = Op[e]
            w = np.linalg.eigvalsh(Op)
            if w[0] < m.LO or w[-1] > m.HI:
                continue
            worst = min(worst, m.objective(Bp, Op, S) - Fend)
    out["perturb_min_dF"] = float(worst)
    # deletion and refit
    dels = []
    for c in [("B",) + e for e in D] + [("O",) + e for e in U]:
        D2 = [e for e in D if ("B",) + e != c]
        U2 = [e for e in U if ("O",) + e != c]
        B2, O2 = B.copy(), O.copy()
        if c[0] == "B":
            B2[c[1], c[2]] = 0
        else:
            O2[c[1], c[2]] = O2[c[2], c[1]] = 0
        par2 = mf.Param(D2, U2)
        warm = [par2.pack(B2, O2)] if np.linalg.eigvalsh(O2)[0] > m.LO else []
        v, Bf, Of, _ = mf.fit(S, D2, U2, rng=rng, warm=warm, n_random=3)
        Fdel = m.objective(Bf, Of, S) if Bf is not None else np.inf
        dels.append((Fdel - Fend, c, v - lsat))
    dels.sort(key=lambda t: t[0])
    out["del_min_dF_over_lam"] = dels[0][0] / m.LAM if dels else np.nan
    out["del_max_dF_over_lam"] = dels[-1][0] / m.LAM if dels else np.nan
    out["del_all_increase"] = bool(all(t[0] > 0 for t in dels))
    out["del_n_decrease"] = int(sum(t[0] < 0 for t in dels))
    out["del_best_coord"] = str(dels[0][1]) if dels else ""
    # deletion path for the best deletion
    if dels and dels[0][0] < 0:
        c = dels[0][1]
        x0 = B[c[1], c[2]] if c[0] == "B" else O[c[1], c[2]]
        s = np.sign(x0)
        ts = np.unique(np.r_[np.linspace(abs(x0), m.DELTA, 15), np.linspace(m.DELTA, 0, 26)])[::-1]
        others = [("B",) + e for e in D if ("B",) + e != c] + [("O",) + e for e in U if ("O",) + e != c]
        prev_B, prev_O = B.copy(), O.copy()
        Fpath = []
        for t in ts:
            fixed = {c: s * t}
            par3 = mf.Param(D, U, fixed)
            v, Bf, Of, _ = mf.fit(S, D, U, rng=rng, warm=[par3.pack(prev_B, prev_O)], n_random=0, fixed=fixed,
                                  pen_coords=others, include_default=False)
            if Bf is None:
                Fpath.append(np.nan)
                continue
            Fpath.append(m.objective(Bf, Of, S))
            prev_B, prev_O = Bf, Of
        Fpath = np.array(Fpath)
        out["path_barrier_over_lam"] = float(np.nanmax(Fpath - Fend) / m.LAM)
        out["path_end_dF_over_lam"] = float((Fpath[-1] - Fend) / m.LAM)
        out["path_nonincreasing"] = bool(np.all(np.diff(Fpath) <= 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("max_sample", nargs="?", type=int, default=None, help="samples below this index only")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()
    tag = a.tag
    runs = pd.read_csv(OUT / f"rerun_{tag}.csv")
    ep = np.load(OUT / f"rerun_{tag}_endpoints.npz")
    runs["emin"] = [s in EMIN[d] for s, d in zip(runs.support_thr, runs.design)]
    keep = (runs.start_type == "random") & ~runs.emin
    if a.max_sample is not None:
        keep &= runs.dataset < a.max_sample
    sel = runs.index[keep]
    tag = tag + (f"_ds{a.max_sample}" if a.max_sample is not None else "")
    args = [(runs.loc[i].to_dict(), ep["B"][i], ep["O"][i]) for i in sel]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(analyze, args, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv(OUT / f"local_minima_{tag}.csv", index=False)
    L = []
    p = lambda *a: L.append(" ".join(str(x) for x in a))
    p(f"failed random-start end points analyzed: {len(df)}")
    for dname, G in df.groupby("design"):
        e = pop[dname]["e_star"]
        p(f"\n{dname}: {len(G)} end points")
        p("  count > e*:", int((G["count"] > e).sum()), "; bowed:", int(G.bowed.sum()),
          "; all coords >= delta:", int((G.min_abs >= m.DELTA).sum()), "; min |coord| min:", float(G.min_abs.min()))
        p("  l - l_sat: max", float(G.l_minus_lsat.max()), "; exact fits of S (<1e-9):", int((G.l_minus_lsat < 1e-9).sum()))
        p("  l - l_min(E_min): max", float(G.l_minus_lEmin.max()), "; fits S at least as well as E_min (<=1e-9):",
          int((G.l_minus_lEmin <= 1e-9).sum()), "; (F_end - F_Emin)/lambda quantiles:",
          G.F_minus_FEmin_over_lam.quantile([0, .5, 1]).round(3).to_dict())
        p("  max |grad| on support:", float(G.max_grad_support.max()))
        p("  off-support ratio |dl|/(2lam/delta) max: B", float(G.max_offsupport_ratio_B.max()), " O",
          float(G.max_offsupport_ratio_O.max()))
        p("  restricted Hessian min eig:", float(G.hess_min_eig.min()), "; flat directions: ",
          G.hess_n_flat.value_counts().sort_index().to_dict())
        p("  perturbation min dF:", float(G.perturb_min_dF.min()), "; negative (< -1e-12):", int((G.perturb_min_dF < -1e-12).sum()))
        p("  deletion+refit: end points where EVERY single deletion increases F:", int(G.del_all_increase.sum()), "of", len(G))
        p("  deletion+refit: best deletion dF/lambda quantiles:", G.del_min_dF_over_lam.quantile([0, .1, .5, .9, 1]).round(3).to_dict())
        if "path_barrier_over_lam" in G:
            H = G.dropna(subset=["path_barrier_over_lam"])
            p("  deletion path (best edge): barrier/lambda quantiles:", H.path_barrier_over_lam.quantile([0, .5, .9, 1]).round(6).to_dict(),
              "; non-increasing paths:", int(H.path_nonincreasing.sum()), "of", len(H),
              "; path end dF/lambda median:", round(float(H.path_end_dF_over_lam.median()), 3))
    (OUT / f"local_minima_{tag}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Compare a rerun by cycle.py (results/rerun_TAG.csv) with the run table of the random-start study.

Usage: python3 compare_rerun.py TAG [--small]
Writes results/compare_TAG.txt and results/compare_TAG_joined.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY_RESULTS = HERE.parent / "results"
POPULATION = HERE.parent.parent / "global_selection" / "results" / "population.json"
OUT = HERE / "results"
tag = sys.argv[1]
small = "--small" in sys.argv
rerun = pd.read_csv(OUT / f"rerun_{tag}.csv")
study = pd.read_csv(STUDY_RESULTS / ("runs_small_starts.csv.gz" if small else "runs.csv.gz"))
study = study[study.start_type.isin(["random", "deterministic"])]
key = ["design", "n", "dataset", "start_type", "start"]
j = rerun.merge(study, on=key, suffixes=("_me", "_st"), how="left", validate="one_to_one")
assert j.objective_st.notna().all()
pop = json.loads(POPULATION.read_text())["designs"]
emin = {d: set(v["E_min_names"]) for d, v in pop.items()}
L = []
p = lambda *a: L.append(" ".join(str(x) for x in a))
p(f"runs compared: {len(j)} ({tag})")
so = j.start_type == "random"
p("start order identical:", int((j[so].start_order_me.astype(float).astype(int) == j[so].start_order_st.astype(float).astype(int)).sum()), "of", int(so.sum()))
p("start omega scale identical:", int((np.abs(j[so].start_omega_scale_me - j[so].start_omega_scale_st) < 1e-12).sum()), "of", int(so.sum()))
p("start objective max |diff|:", float((j.start_objective_me - j.start_objective_st).abs().max()))
for c in ("support_thr", "support_raw", "support_half"):
    p(f"{c} identical:", int((j[c + "_me"] == j[c + "_st"]).sum()), "of", len(j))
j["emin_me"] = [s in emin[d] for s, d in zip(j.support_thr_me, j.design)]
j["emin_st"] = [s in emin[d] for s, d in zip(j.support_thr_st, j.design)]
p("E_min status identical:", int((j.emin_me == j.emin_st).sum()), "of", len(j))
p("status identical:", int((j.status_me == j.status_st).sum()), "of", len(j))
dob = (j.objective_me - j.objective_st).abs()
p("objective |diff|: max", float(dob.max()), "median", float(dob.median()), "; > 1e-10:", int((dob > 1e-10).sum()),
  "; > 1e-8:", int((dob > 1e-8).sum()))
dc = (j.cycles_me - j.cycles_st).abs()
p("cycles identical:", int((dc == 0).sum()), "; |diff| <= 5:", int((dc <= 5).sum()), "; max |diff|:", int(dc.max()))
p("conic calls in rerun:", int(j.conic.sum()), "; B-step order ties changing the matrix:", int(j.order_ties.sum()))
p("max sampled objective increase in rerun:", float(j.max_increase_sampled.max()))
dis = j[(j.support_thr_me != j.support_thr_st) | (dob > 1e-8)]
p("\nDisagreements (support_thr or objective > 1e-8):", len(dis))
if len(dis):
    p(dis[key + ["support_thr_me", "support_thr_st", "objective_me", "objective_st", "cycles_me", "cycles_st",
                 "status_me", "status_st"]].to_string())
# per-cell rates, rerun vs study
g = j[j.start_type == "random"].groupby(["design", "n"])
p("\nsingle-start E_min rate (random), rerun vs study, same runs:")
p(pd.concat([g.emin_me.mean().rename("rerun"), g.emin_st.mean().rename("study"), g.size().rename("runs")], axis=1).to_string())
g = j[j.start_type == "deterministic"].groupby(["design", "n"])
p("deterministic E_min, rerun vs study:")
p(pd.concat([g.emin_me.sum().rename("rerun"), g.emin_st.sum().rename("study"), g.size().rename("runs")], axis=1).to_string())
(OUT / f"compare_{tag}.txt").write_text("\n".join(L) + "\n")
j.to_csv(OUT / f"compare_{tag}_joined.csv", index=False)
print("\n".join(L))

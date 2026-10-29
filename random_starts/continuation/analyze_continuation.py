"""Compare the outcomes of the random-start study at the 20,000-cycle cap with the outcomes
after continuing the capped runs to 400,000 cycles (Supplementary Section S18, Anomalies).

Reads ../results/runs.csv.gz and ../results/datasets.csv, and the continued runs from
results/runs_continued.jsonl (written by continue_capped.py --run) when present, otherwise
from results/runs_continued.csv.gz.  Writes to results/:
  runs_continued.csv.gz              one row per capped run (1,805): rerun state at cycle
                                     20,000 (cap20k_*) and final state (new_*); from the jsonl only
  endpoints_continued.npz            B and Omega at cycle 20,000 and at the end; from the jsonl only
  repro_all_capped.csv.gz            state at cycle 20,000 of each rerun vs the recorded row
  runs_with_continuation.csv.gz      all 63,900 runs with old_* (cap 20,000) and new_* outcomes
  datasets_reference_continued.csv.gz  old and new reference minimum per sample (900)
  cell_comparison_old_vs_new.csv     per (design, n): rates, best of K, capped counts
  capped_fate_by_cell.csv            per (design, n): fate of the capped random and refit runs
  capped_support_transitions.csv     support at cycle 20,000 -> final support, capped runs
  bestofK_emin_not_reference.csv     best of K at a minimum-edge support but not at the reference
Success: the final support thresholded at delta is a minimum-edge support (is_emin).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from continue_capped import compare_snapshot, OUT, PARTIAL_NAME  # noqa: E402
import runner  # noqa: E402

STUDY_RESULTS = runner.STUDY_RESULTS
DESIGNS, NS, KS = ["G1", "G2", "G3"], [250, 1000, 4000], [1, 2, 5, 10, 20]
TOL = 1e-8
KEY = ["design", "n", "dataset", "start_type", "start"]
ARRAYS = ("cap20k_B", "cap20k_O", "final_B", "final_O")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    old = pd.read_csv(STUDY_RESULTS / "runs.csv.gz")
    dsets = pd.read_csv(STUDY_RESULTS / "datasets.csv")
    partial = out / PARTIAL_NAME
    from_jsonl = partial.exists()
    if from_jsonl:
        rows = [json.loads(l) for l in open(partial) if l.strip()]
    else:
        rows = pd.read_csv(out / "runs_continued.csv.gz", float_precision="round_trip").to_dict("records")
    keys = [tuple(r[k] for k in KEY) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate continued runs"
    capped = old[old.status == "cycle_cap"]
    assert len(rows) == len(capped) == 1805, (len(rows), len(capped))

    # ---- reproduction of the cap-time state for every capped run
    oldidx = old.set_index(KEY)
    rep = []
    for r in rows:
        o = oldidx.loc[tuple(r[k] for k in KEY)]
        rep.append(compare_snapshot(r, o))
    rep = pd.DataFrame(rep)
    rep.to_csv(out / "repro_all_capped.csv.gz", index=False)

    # ---- per-run table and end points
    cont = pd.DataFrame([{k: v for k, v in r.items() if k not in ARRAYS}
                         for r in rows]).sort_values(KEY).reset_index(drop=True)
    if from_jsonl:
        order = {tuple(r[k] for k in KEY): i for i, r in enumerate(rows)}
        idx = [order[tuple(x)] for x in cont[KEY].itertuples(index=False)]
        np.savez_compressed(out / "endpoints_continued.npz",
                            key=np.array(["|".join(map(str, tuple(cont.loc[i, KEY]))) for i in range(len(cont))]),
                            cap20k_B=np.array([rows[i]["cap20k_B"] for i in idx]).reshape(-1, 3, 3),
                            cap20k_O=np.array([rows[i]["cap20k_O"] for i in idx]).reshape(-1, 3, 3),
                            final_B=np.array([rows[i]["final_B"] for i in idx]).reshape(-1, 3, 3),
                            final_O=np.array([rows[i]["final_O"] for i in idx]).reshape(-1, 3, 3))
        cont.to_csv(out / "runs_continued.csv.gz", index=False)

    # ---- merged run table
    m = old.merge(cont[KEY + ["new_objective", "new_status", "new_cycles", "new_support_thr", "new_is_emin",
                              "new_support_count", "new_bowed", "new_more_than_e_star", "new_fewer_than_e_star",
                              "new_support_half", "new_min_nonzero_coord", "new_n_coords_in_(0,delta)",
                              "new_max_objective_increase", "new_max_curvature_ratio_B",
                              "new_max_curvature_ratio_Omega", "new_runtime_seconds", "new_final_step"]],
                  on=KEY, how="left")
    was = m.new_objective.notna()
    assert int(was.sum()) == 1805
    m["continued"] = was
    for a in ("objective", "status", "cycles", "support_thr", "is_emin", "support_count", "bowed",
              "more_than_e_star", "fewer_than_e_star", "support_half", "min_nonzero_coord",
              "n_coords_in_(0,delta)", "max_objective_increase", "max_curvature_ratio_B",
              "max_curvature_ratio_Omega", "final_step"):
        m[f"new_{a}"] = m[f"new_{a}"].where(was, m[a])
    m["new_is_emin"] = m.new_is_emin.astype(bool)
    m["is_emin"] = m.is_emin.astype(bool)
    ref_new = m.groupby(["design", "n", "dataset"]).new_objective.min().rename("new_reference_objective")
    m = m.merge(ref_new.reset_index(), on=["design", "n", "dataset"])
    m["new_gap_to_reference"] = m.new_objective - m.new_reference_objective
    m["new_attains_reference"] = m.new_objective <= m.new_reference_objective + TOL
    m["attains_reference"] = m.attains_reference.astype(bool)
    out_cols = KEY + ["continued", "status", "cycles", "objective", "support_thr", "is_emin", "attains_reference",
                      "gap_to_reference", "new_status", "new_cycles", "new_objective", "new_support_thr",
                      "new_is_emin", "new_support_count", "new_bowed", "new_more_than_e_star",
                      "new_fewer_than_e_star", "new_support_half", "new_n_coords_in_(0,delta)",
                      "new_max_objective_increase", "new_reference_objective", "new_gap_to_reference",
                      "new_attains_reference"]
    m = m.sort_values(KEY).reset_index(drop=True)
    m[out_cols].rename(columns={c: f"old_{c}" for c in ["status", "cycles", "objective", "support_thr", "is_emin",
                                                        "attains_reference", "gap_to_reference"]}) \
        .to_csv(out / "runs_with_continuation.csv.gz", index=False)

    # ---- per-dataset reference
    D = dsets[["design", "n", "dataset", "reference_objective", "reference_supports", "reference_all_emin",
               "l0_fixed_tuning_min"]].merge(ref_new.reset_index(), on=["design", "n", "dataset"])
    D["reference_change"] = D.new_reference_objective - D.reference_objective
    at = m[m.new_attains_reference]
    g = at.groupby(["design", "n", "dataset"])
    D = D.merge(g.new_support_thr.agg(lambda s: ";".join(sorted(set(s)))).rename("new_reference_supports").reset_index(),
                on=["design", "n", "dataset"])
    D = D.merge(g.new_is_emin.all().rename("new_reference_all_emin").reset_index(), on=["design", "n", "dataset"])
    D = D.merge(g.start_type.agg(lambda s: "random" in set(s)).rename("new_reference_attained_by_random").reset_index(),
                on=["design", "n", "dataset"])
    D["new_reference_minus_l0_min"] = D.new_reference_objective - D.l0_fixed_tuning_min
    D.to_csv(out / "datasets_reference_continued.csv.gz", index=False)

    # ---- per-cell comparison
    cells, fates, trans = [], [], []
    for d in DESIGNS:
        for n in NS:
            C = m[(m.design == d) & (m.n == n)]
            R = C[C.start_type == "random"]
            F = C[C.start_type == "refit"]
            row = dict(design=d, n=n, random_runs=len(R),
                       single_old=R.is_emin.mean(), single_new=R.new_is_emin.mean(),
                       successes_old=int(R.is_emin.sum()), successes_new=int(R.new_is_emin.sum()),
                       single_ref_old=R.attains_reference.mean(), single_ref_new=R.new_attains_reference.mean())
            by = {k: v.sort_values("start") for k, v in R.groupby("dataset")}
            for K in KS:
                eo = en = ro = rn = 0
                for v in by.values():
                    f = v[v.start <= K]
                    bo = f.sort_values(["objective", "start"], kind="stable").iloc[0]
                    bn = f.sort_values(["new_objective", "start"], kind="stable").iloc[0]
                    eo += bool(bo.is_emin); en += bool(bn.new_is_emin)
                    ro += bool(bo.attains_reference); rn += bool(bn.new_attains_reference)
                row[f"bestof{K}_old"] = eo / len(by)
                row[f"bestof{K}_new"] = en / len(by)
                row[f"bestof{K}_ref_old"] = ro / len(by)
                row[f"bestof{K}_ref_new"] = rn / len(by)
            row.update(capped_random_old=int((R.status == "cycle_cap").sum()),
                       capped_random_new=int((R.new_status == "cycle_cap").sum()),
                       capped_refit_old=int((F.status == "cycle_cap").sum()),
                       capped_refit_new=int((F.new_status == "cycle_cap").sum()),
                       capped_all_old=int((C.status == "cycle_cap").sum()),
                       capped_all_new=int((C.new_status == "cycle_cap").sum()),
                       median_cycles_random_old=float(R.cycles.median()),
                       median_cycles_random_new=float(R.new_cycles.median()),
                       max_cycles_random_new=int(R.new_cycles.max()),
                       reference_lowered_datasets=int((D[(D.design == d) & (D.n == n)].reference_change < -TOL).sum()),
                       reference_all_emin_old=float(D[(D.design == d) & (D.n == n)].reference_all_emin.mean()),
                       reference_all_emin_new=float(D[(D.design == d) & (D.n == n)].new_reference_all_emin.mean()))
            cells.append(row)
            Rc = R[R.status == "cycle_cap"]
            Fc = F[F.status == "cycle_cap"]
            fates.append(dict(
                design=d, n=n, capped_random=len(Rc),
                capped_random_successes=int(Rc.is_emin.sum()),
                remain_successes=int((Rc.is_emin & Rc.new_is_emin).sum()),
                successes_lost=int((Rc.is_emin & ~Rc.new_is_emin).sum()),
                capped_random_failures=int((~Rc.is_emin).sum()),
                failures_become_successes=int((~Rc.is_emin & Rc.new_is_emin).sum()),
                capped_random_support_changed=int((Rc.support_thr != Rc.new_support_thr).sum()),
                capped_random_still_capped=int((Rc.new_status == "cycle_cap").sum()),
                capped_random_now_attain_reference=int(Rc.new_attains_reference.sum()),
                capped_random_successes_now_attain_reference=int((Rc.new_is_emin & Rc.new_attains_reference).sum()),
                capped_refit=len(Fc), capped_refit_successes_old=int(Fc.is_emin.sum()),
                capped_refit_successes_new=int(Fc.new_is_emin.sum()),
                capped_refit_support_changed=int((Fc.support_thr != Fc.new_support_thr).sum()),
                capped_refit_still_capped=int((Fc.new_status == "cycle_cap").sum())))
    cells = pd.DataFrame(cells)
    fates = pd.DataFrame(fates)
    tot = fates.drop(columns=["design", "n"]).sum()
    fates = pd.concat([fates, pd.DataFrame([dict(design="all", n="all", **tot.to_dict())])], ignore_index=True)
    cells.to_csv(out / "cell_comparison_old_vs_new.csv", index=False)
    fates.to_csv(out / "capped_fate_by_cell.csv", index=False)
    Cp = m[m.continued]
    tr = Cp.groupby(["design", "start_type", "support_thr", "new_support_thr", "is_emin", "new_is_emin"]).size() \
        .rename("runs").reset_index().sort_values(["design", "start_type", "runs"], ascending=[True, True, False])
    tr.to_csv(out / "capped_support_transitions.csv", index=False)

    # ---- printout
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print("reproduction of the state at cycle 20,000:", int(rep.reproduces.sum()), "of", len(rep),
          "| max |objective diff|", rep.objective_absdiff.max(),
          "| support_thr mismatches", int((~rep.support_thr_match).sum()),
          "| raw mismatches", int((~rep.support_raw_match).sum()),
          "| half mismatches", int((~rep.support_half_match).sum()),
          "| max final_step reldiff", rep.final_step_reldiff.max(),
          "| max start objective diff", rep.start_objective_absdiff.max())
    rnd = m[m.start_type == "random"]
    print("random successes old", int(rnd.is_emin.sum()), "new", int(rnd.new_is_emin.sum()), "of", len(rnd))
    print("capped random old", int((rnd.status == "cycle_cap").sum()), "still capped", int((rnd.new_status == "cycle_cap").sum()))
    print("capped refit old", int(((m.start_type == "refit") & (m.status == "cycle_cap")).sum()),
          "still capped", int(((m.start_type == "refit") & (m.new_status == "cycle_cap")).sum()))
    print(cells[["design", "n", "single_old", "single_new", "successes_old", "successes_new"] +
                [f"bestof{K}_{w}" for K in KS for w in ("old", "new")] +
                ["capped_random_old", "capped_random_new", "capped_refit_old", "capped_refit_new"]].round(4).to_string())
    print(fates.to_string())
    print("datasets with lower reference (by > 1e-8):", int((D.reference_change < -TOL).sum()),
          "| min change", D.reference_change.min(), "| max change", D.reference_change.max())
    print("reference all E_min: old", int(D.reference_all_emin.sum()), "new", int(D.new_reference_all_emin.sum()))
    print("continued runs: max objective increase", Cp.new_max_objective_increase.max(),
          "| max curvature B", Cp.new_max_curvature_ratio_B.max(), "| Omega", Cp.new_max_curvature_ratio_Omega.max(),
          "| runs ending with coords in (0,delta)", int((Cp["new_n_coords_in_(0,delta)"] > 0).sum()),
          "| new fewer than e*", int(Cp.new_fewer_than_e_star.sum()))
    print("all-runs attain-reference flips: old->new True->False",
          int((m.attains_reference & ~m.new_attains_reference).sum()), "False->True",
          int((~m.attains_reference & m.new_attains_reference).sum()))
    # best of K at a minimum-edge support but not at the reference, continued outcomes
    gaps = []
    for (d, n, ds), v in rnd.groupby(["design", "n", "dataset"]):
        v = v.sort_values("start")
        for K in KS:
            f = v[v.start <= K]
            b = f.sort_values(["new_objective", "start"], kind="stable").iloc[0]
            if b.new_is_emin and not b.new_attains_reference:
                gaps.append(dict(design=d, n=n, dataset=ds, K=K, start=int(b.start), gap=b.new_gap_to_reference,
                                 status=b.new_status))
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(out / "bestofK_emin_not_reference.csv", index=False)
    print("best-of-K E_min but not at reference (new):", len(gaps),
          (gaps.groupby(["design", "n"]).size().to_dict() if len(gaps) else {}))


if __name__ == "__main__":
    main()

"""Per-sample summary of the certification (Section S18, "Certified fits and criterion bounds").

Reads results/subset.csv, support_bounds.csv.gz, support_bounds_samples.csv,
criterion_samples.csv and criterion_regions.csv.gz.  Writes results/certify_summary.csv
and results/certify_summary_table.md, and prints the counts reported in Section S18.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
sub = pd.read_csv(R / "subset.csv")
sub["sample"] = np.arange(len(sub))
A = pd.read_csv(R / "support_bounds.csv.gz", float_precision="round_trip", low_memory=False)
As = pd.read_csv(R / "support_bounds_samples.csv")
Bs = pd.read_csv(R / "criterion_samples.csv", float_precision="round_trip")
Br = pd.read_csv(R / "criterion_regions.csv.gz", float_precision="round_trip", low_memory=False)

a_agg = A.groupby("sample").agg(supports_certified=("certified_final", "sum"), supports=("support", "size"),
                                max_gap_certified=("found_minus_lb_final", lambda x: float(np.max(x[A.loc[x.index, "certified_final"]]))),
                                max_ub_gap=("ub_minus_found", "max"),
                                fits_spectral_runtime_s=("spectral_runtime_s", lambda x: float(np.nansum(x)))).reset_index()
m = sub.merge(a_agg, on="sample").merge(As[["sample", "runtime_s"]].rename(columns={"runtime_s": "fits_runtime_s"}),
                                        on="sample")
m = m.merge(Bs[["sample", "s_star", "regions_screened", "regions_bnb", "regions_certified", "all_certified",
                "min_region_lb_minus_s_star", "boxes", "runtime_s"]].rename(
                    columns={"runtime_s": "criterion_runtime_s", "boxes": "criterion_boxes"}),
            on="sample", how="left")
m["fits_runtime_s"] = m["fits_runtime_s"] + m["fits_spectral_runtime_s"]
m["group"] = np.where(~m.selected_is_emin & m.design.isin(["G1", "G2"]), "non-Emin chain/confounded",
                      np.where(~m.selected_is_emin, "front-door underfit", "random other"))
m.to_csv(R / "certify_summary.csv", index=False)
print("samples", len(m), "with criterion result", int(m.all_certified.notna().sum()))
print("support fits certified", int(m.supports_certified.sum()), "of", int(m.supports.sum()))
print("largest certified gap", m.max_gap_certified.max(), "largest upper-bound gap", m.max_ub_gap.max())
print("samples with certified criterion bound", int((m.all_certified == True).sum()))  # noqa: E712
print(m.groupby("group").agg(N=("sample", "size"), fits_all=("supports_certified", lambda x: int((x == 200).sum())),
                             criterion=("all_certified", lambda x: int((x == True).sum())),  # noqa: E712
                             fits_s_median=("fits_runtime_s", "median"),
                             criterion_s_median=("criterion_runtime_s", "median"),
                             criterion_s_max=("criterion_runtime_s", "max")).to_string())
print("region stages:", Br["tried"].value_counts().to_dict())
bad = ~Br.certified.astype(bool)
print("regions not certified:", int(bad.sum()))
if bad.any():
    print(Br[bad][["sample", "design", "n", "rep", "order", "L", "H_D", "H_U", "H_margin", "boxes",
                   "remaining"]].to_string())
with open(R / "certify_summary_table.md", "w") as f:
    f.write("| sample | design | n | rep | group | selected | fits certified | largest gap | "
            "criterion certified | regions screened / bounded | fits (s) | criterion (s) |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in m.sort_values(["group", "design", "n", "rep"]).iterrows():
        bc = "yes" if r.all_certified == True else ("no" if r.all_certified == False else "not run")  # noqa: E712
        f.write(f"| {r['sample']} | {r.design} | {r.n} | {r.rep} | {r.group} | {r.selected_support} | "
                f"{int(r.supports_certified)}/200 | {r.max_gap_certified:.1e} | {bc} | "
                f"{r.regions_screened}/{r.regions_bnb} | {r.fits_runtime_s:.1f} | {r.criterion_runtime_s:.0f} |\n")
print("wrote", R / "certify_summary_table.md")

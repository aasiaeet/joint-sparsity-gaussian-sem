"""Compare the certified support bounds with the fits of the independent implementation.

../global_selection/independent_check.py (joint barrier method, no solver code shared with
global_selection.py) stores, in its samples stage, the raw fitted value of all 200 supports
for every subset sample in ../global_selection/results/independent_check_samples_raw.json
(about 3.7 MB, not stored in the repository; regenerate it with
`python3 independent_check.py samples`).  Its values are attained at feasible points, so
they can never fall below a certified lower bound.  Over all 17,200 fits we check
  (i)  independent value >= lb_final - 1e-12   (consistency of the certificate), and
  (ii) |independent value - stored value| <= 1e-8 where certified (agreement of the fits).
Writes results/validation/compare_independent_fits.json (and a per-fit csv.gz).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GS = ROOT.parent / "global_selection"
sys.path.insert(0, str(GS))
import independent_check as ic  # noqa: E402  (only the support indexing is used)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--raw", type=str, default=str(GS / "results" / "independent_check_samples_raw.json"))
a = ap.parse_args()
if not Path(a.raw).exists():
    sys.exit(f"{a.raw} not found; run the samples stage of independent_check.py first")
d = pd.read_csv(ROOT / "results" / "support_bounds.csv.gz", float_precision="round_trip")
raw = json.load(open(a.raw))
lookup = {(o["design"], o["n"], o["rep"]): o for o in raw}
rows = []
for _, r in d.iterrows():
    o = lookup.get((r.design, int(r.n), int(r.rep)))
    if o is None:
        continue
    j = ic.KEY2I[ic.key_of(r.support)]
    v = float(o["raw"][j])
    rows.append(dict(sample=int(r["sample"]), design=r.design, n=int(r.n), rep=int(r.rep), support=r.support,
                     found=r.found, independent=v, lb_final=r.lb_final, ub=r.ub,
                     certified=bool(r.certified_final),
                     independent_minus_lb=v - r.lb_final, independent_minus_found=v - r.found))
s = pd.DataFrame(rows)
s.to_csv(ROOT / "results" / "validation" / "compare_independent_fits.csv.gz", index=False)
c = s[s.certified]
out = dict(
    fits_compared=len(s),
    independent_below_certified_lb=int(np.sum(s.independent_minus_lb < -1e-12)),
    min_independent_minus_lb=float(s.independent_minus_lb.min()),
    certified_fits=len(c),
    certified_with_independent_within_1e8=int(np.sum(np.abs(c.independent_minus_found) <= 1e-8)),
    certified_with_independent_above_by_more_than_1e8=int(np.sum(c.independent_minus_found > 1e-8)),
    certified_with_independent_below_by_more_than_1e8=int(np.sum(c.independent_minus_found < -1e-8)),
    max_abs_independent_minus_found_certified=float(np.max(np.abs(c.independent_minus_found))),
)
picks = []
for des, sup in [("G1", "1->2,2->3"), ("G1", "2->3,1<->2"), ("G2", "1->2,2<->3"), ("G2", "2->1,2<->3"),
                 ("G3", "1->2,2->3,1<->3")]:
    row = s[(s.design == des) & (s.support == sup)].iloc[0]
    picks.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()})
out["five_named_supports"] = picks
print(json.dumps(out, indent=1, default=str))
(ROOT / "results" / "validation" / "compare_independent_fits.json").write_text(
    json.dumps(out, indent=1, default=str) + "\n")

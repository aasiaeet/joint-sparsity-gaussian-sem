"""Print Table 1 of the main article from the stored results of global_selection.py.

For each design: e*, the number of supports in the minimum-edge fiber and how many
of them carry a bidirected edge or a bow (results/population.json); the smallest
likelihood gap l_min(H; Sigma*) - l_* over supports with at most e* edges that do not
represent Sigma* exactly; and the proportion of the 200 samples whose thresholded
selected support is a minimum-edge support at n = 100, 200, 400 (results/summary.csv).

Usage: python3 table1.py
"""
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
LABELS = {"G1": "Chain", "G2": "Confounded pair", "G3": "Front-door"}


def main():
    pop = json.loads((RESULTS / "population.json").read_text())["designs"]
    with open(RESULTS / "summary.csv") as f:
        summary = list(csv.DictReader(f))
    print("Design            e*  All  Bidirected  Bowed  Smallest gap  n=100  n=200  n=400")
    for d, label in LABELS.items():
        info = pop[d]
        emin = info["E_min"]
        n_bidirected = sum("<->" in s["support"] for s in emin)
        n_bowed = sum(s["bowed"] for s in emin)
        gap = info["smallest_gap_nonfitting_count_le_e_star"]
        prop = {int(r["n"]): float(r["prop_emin"]) for r in summary if r["design"] == d}
        print(f"{label:<16} {info['e_star']:>3} {len(emin):>4} {n_bidirected:>11} {n_bowed:>6}"
              f" {gap:>13.3f} {prop[100]:>6.3f} {prop[200]:>6.3f} {prop[400]:>6.3f}")


if __name__ == "__main__":
    main()

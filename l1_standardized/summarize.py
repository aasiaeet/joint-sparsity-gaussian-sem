"""Summary tables of the standardized l1 search and its verification (Supplement S13).

Usage: python3 summarize.py > results/summary_tables.txt
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_common import RESULTS, open_table


def load(name):
    try:
        with open_table(name) as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return None


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def p3():
    rows = load("p3_search.csv")
    if rows is None:
        return
    print("## p = 3")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["part"], r["family"])].append(r)
    print("| part | family | n Sigma | max margin | max |S1-S2| global | max |S1-S2| class | global optimum in class | extra vanishing | median s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(groups):
        g = groups[k]
        m = np.array([fnum(r["margin"]) for r in g])
        dg = np.array([abs(fnum(r["s1_s2_global_diff"])) for r in g])
        dc = np.array([abs(fnum(r["s1_s2_class_diff"])) for r in g])
        inc = sum(int(r["global_support_in_class"]) for r in g)
        ev = sum(1 for r in g if r["extra_vanishing"])
        sec = np.median([fnum(r["seconds"]) for r in g])
        print(f"| {k[0]} | {k[1]} | {len(g)} | {m.max():.2e} | {np.nanmax(dg):.1e} | {np.nanmax(dc):.1e} | {inc}/{len(g)} | {ev} | {sec:.1f} |")
    m = np.array([fnum(r["margin"]) for r in rows])
    print(f"\nall: {len(rows)} Sigma, max margin {m.max():.3e}, min margin {m.min():.3e}, n >= 1e-3: {(m >= 1e-3).sum()}")
    bo = np.array([fnum(r["best_actual_outside"]) - fnum(r["class_min"]) for r in rows])
    bo = bo[np.isfinite(bo)]
    print(f"best outside-class solution minus class minimum: min {bo.min():.3e}, median {np.median(bo):.3e}")
    gs = defaultdict(int)
    for r in rows:
        gs[r["gstar"]] += 1
    print(f"generating graphs covered: {len(gs)}")
    tot = sum(fnum(r["seconds"]) for r in rows)
    print(f"CPU seconds {tot:.0f}")


def p4(name="p4_search.csv", vname="p4_verify.csv"):
    rows = load(name)
    if rows is None:
        return
    print("## p = 4")
    m = np.array([fnum(r["margin"]) for r in rows])
    print(f"{len(rows)} Sigma over {len(set(r['gstar'] for r in rows))} graphs; max margin {m.max():.4f}; "
          f"n >= 1e-3: {(m >= 1e-3).sum()}; global optimum outside class: "
          f"{sum(1 for r in rows if r['global_support_in_class'] == '0')}; class solves run: "
          f"{sum(int(r['ran_class']) for r in rows)}; CPU s {sum(fnum(r['seconds']) for r in rows):.0f}")
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(fnum(r["margin"]))
    for f_, ms in fam.items():
        ms = np.array(ms)
        print(f"  {f_}: n {len(ms)}, n>=1e-3 {(ms >= 1e-3).sum()}, max {ms.max():.4f}")
    byg = defaultdict(list)
    for r in rows:
        byg[r["gstar"]].append(r)
    print("\n| G* | dim | vanishing | n Sigma | n margin>=1e-3 | max margin | outside support at max |")
    print("|---|---|---|---|---|---|---|")
    hits = []
    for g, rs in byg.items():
        ms = np.array([fnum(r["margin"]) for r in rs])
        if (ms >= 1e-3).any():
            i = int(np.argmax(ms))
            hits.append((ms.max(), g, rs[0]["dim_star"], rs[0]["vanishing_star"], len(rs), int((ms >= 1e-3).sum()), rs[i]["global_support"]))
    for h in sorted(hits, reverse=True):
        print(f"| {h[1]} | {h[2]} | {h[3]} | {h[4]} | {h[5]} | {h[0]:.4f} | {h[6]} |")
    print(f"\ngraphs with a candidate: {len(hits)} of {len(byg)}")
    ver = load(vname)
    if ver:
        print(f"\n### verification ({len(ver)} candidates)")
        keys = ["phi_err_double", "phi_err_mp", "omega_min_eig", "min_entry_in_support", "max_entry_off_support"]
        for k in keys:
            v = np.array([fnum(r[k]) for r in ver])
            print(f"  {k}: min {np.nanmin(v):.3e} max {np.nanmax(v):.3e}")
        s2c = np.array([fnum(r["s2_class"]) - fnum(r["s1_class"]) for r in ver])
        print(f"  S2 class - S1 class: min {np.nanmin(s2c):.3e} max {np.nanmax(s2c[np.isfinite(s2c)]):.3e}")
        mv = np.array([fnum(r["margin_verified"]) for r in ver])
        print(f"  verified margin: min {mv.min():.4e} max {mv.max():.4f}; n >= 1e-3: {(mv >= 1e-3).sum()}")
        oc = sum(int(r["outside_in_class"]) for r in ver)
        print(f"  outside representation in class (should be 0): {oc}")
        g2 = [r for r in ver if r["s2_global"] not in ("", "nan")]
        if g2:
            d = np.array([fnum(r["s1_minus_s2_global"]) for r in g2])
            print(f"  S2 global check on {len(g2)}: S1 - S2 in [{d.min():.3e}, {d.max():.3e}]")
        pl = sum(1 for r in ver if int(r["s2_param_level_supports"]) > 0)
        print(f"  candidates with class supports needing parameter-level consistency in S2: {pl}")


if __name__ == "__main__":
    p3()
    p4()

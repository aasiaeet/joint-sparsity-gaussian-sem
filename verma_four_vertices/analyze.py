"""Combine fits, 50-digit rechecks and classes into the summary tables (Supplement S18).

Reads results/fits.csv, results/classes.csv, results/exact_solutions_le5.json,
results/hp_recheck.csv, results/bound_scan.csv and, when present,
results/bound_scan_near_limit.csv (written by near_limit_scan.py after a first run).
Writes results/by_edge_count.csv, results/min_edge_supports.csv,
results/markov_class_e_star.csv, results/gstar_class_members.csv,
results/support_table.csv (one row per support, final columns), results/summary.json,
and prints them as Markdown tables.

Final classification of a support:
  gap_hp    =  min of the 50-digit KL gaps at the KL optimum and at the residual
               point (Omega = R(B) on the pattern), both with |B| <= 1000;
  exact     <=> gap_hp < 1e-10; the residual criterion
                (max|rho| < 1e-9) is reported as a cross-check, with disagreements listed;
  separated <=> some polynomial that vanishes identically on M_H (a CI minor or a
                relabeled Verma cubic) is nonzero at Sigma*; then Sigma* is not in
                V_H, which contains the closure of M_H, so H has no exact
                representation and no sequence of representations converging to Sigma*.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import P, RESULTS, SUPPORTS, GSTAR, VERMA_PERMS, CI_STATEMENTS, ci_name, covariance
from fit_supports import SIGMA, L_SAT, B_STAR, O_STAR

E_STAR = GSTAR.count
EXACT_TOL = 1e-10
fits = pd.read_csv(RESULTS / "fits.csv")
cls = pd.read_csv(RESULTS / "classes.csv", dtype={"ci_vanish": str, "verma_vanish": str, "msep": str})
hp = pd.read_csv(RESULTS / "hp_recheck.csv")
df = fits.merge(cls.drop(columns=["name", "edges", "n_directed", "n_bidirected", "bowed"]), on="index")
df = df.merge(hp, on="index")
df["gap_hp_fullrun"] = df[["gap_hp_klpoint", "gap_hp_residpoint"]].min(axis=1, skipna=True)
# fold in the bound scans (refits with |B| <= 10, 100, 1000, 10000; 50-digit gaps)
SCAN_COLS = ["gap_hp_M10", "gap_hp_M100", "gap_hp_M1000", "gap_hp_M10000"]
NEAR_LIMIT = RESULTS / "bound_scan_near_limit.csv"
_scans = [pd.read_csv(RESULTS / "bound_scan.csv").assign(scan="le5_at_bound")]
if NEAR_LIMIT.exists():
    _scans.append(pd.read_csv(NEAR_LIMIT).assign(scan="near_limit"))
_scans = pd.concat(_scans)
_scans["gap_scan_min"] = _scans[SCAN_COLS].min(axis=1)
_scans["decade_ratio_1e3_1e4"] = _scans.gap_hp_M10000 / _scans.gap_hp_M1000
df = df.merge(_scans[["index", "scan", "gap_scan_min", "decade_ratio_1e3_1e4"] + SCAN_COLS], on="index", how="left")
df["gap_hp"] = df[["gap_hp_fullrun", "gap_scan_min"]].min(axis=1, skipna=True)
df["exact"] = df.gap_hp < EXACT_TOL
df["exact_kl_hp"] = df.exact
df["disagree_float"] = df.exact_kl != df.exact_resid
df["disagree_hp"] = df.exact != (df.exact_resid == 1)
df["near_limit"] = ~df.exact & (df.gap_hp < 1e-6)
df["category"] = np.where(df.exact, "exact", np.where(df.near_limit, "near_limit", "nonexact"))
df["kl_at_bound"] = df.kl_maxabsB > 999
sols = json.loads((RESULTS / "exact_solutions_le5.json").read_text())
scan = pd.read_csv(RESULTS / "bound_scan.csv")

k_ci = CI_STATEMENTS.index((0, 2, (1,)))
k_v = VERMA_PERMS.index((0, 1, 2, 3))
df["f_CI_vanishes"] = df.ci_vanish.str[k_ci] == "1"
df["f_V_vanishes"] = df.verma_vanish.str[k_v] == "1"


# --- separating polynomials (algebraic certificate of non-exactness)
def _pcorr(i, j, Sx):
    idx = [i, j] + list(Sx)
    M = np.linalg.inv(SIGMA[np.ix_(idx, idx)])
    return -M[0, 1] / np.sqrt(M[0, 0] * M[1, 1])


def _verma_norm(pm):
    a, b, c, d = pm
    s = SIGMA
    v = s[a, c] * s[b, b] * s[c, d] - s[a, d] * s[b, b] * s[c, c] - s[a, c] * s[b, c] * s[b, d] + s[a, d] * s[b, c] ** 2
    return v / (np.sqrt(s[a, a] * s[c, c] * s[d, d]) * s[b, b])


CI_AT_STAR = np.array([_pcorr(*st) for st in CI_STATEMENTS])
V_AT_STAR = np.array([_verma_norm(pm) for pm in VERMA_PERMS])
ZERO_CI = np.abs(CI_AT_STAR) < 1e-12
ZERO_V = np.abs(V_AT_STAR) < 1e-12


def separation(r):
    a = np.array([ch == "1" for ch in r.ci_vanish])
    b = np.array([ch == "1" for ch in r.verma_vanish])
    vals = np.r_[np.abs(CI_AT_STAR[a & ~ZERO_CI]), np.abs(V_AT_STAR[b & ~ZERO_V])]
    return float(vals.max()) if len(vals) else 0.0


df["separation"] = df.apply(separation, axis=1)
df["separated"] = df.separation > 0


def ci_list(bitstr):
    return [ci_name(st) for st, b in zip(CI_STATEMENTS, bitstr) if b == "1"]


def verma_list(bitstr):
    return ["".join(str(v + 1) for v in pm) for pm, b in zip(VERMA_PERMS, bitstr) if b == "1"]


# --- by edge count
rows = []
for k, g in df.groupby("edges"):
    ne = g[~g.exact]
    rows.append(dict(edges=k, supports=len(g), exact=int(g.exact.sum()),
                     exact_kl_float=int(g.exact_kl.sum()), exact_kl_hp=int(g.exact_kl_hp.sum()),
                     exact_resid=int(g.exact_resid.sum()),
                     disagree_float=int(g.disagree_float.sum()), disagree_hp=int(g.disagree_hp.sum()),
                     negative_float_gaps=int((g.gap < -1e-12).sum()),
                     nonexact=len(ne), near_limit=int(g.near_limit.sum()), nonexact_separated=int(ne.separated.sum()),
                     nonexact_kl_at_bound=int(ne.kl_at_bound.sum()),
                     min_gap_hp_nonexact=float(ne.gap_hp.min()) if len(ne) else None,
                     min_gap_hp_nonexact_unseparated=float(ne[~ne.separated].gap_hp.min()) if (~ne.separated).any() else None,
                     min_resid_nonexact=float(ne.resid.min()) if len(ne) else None,
                     rho_vanishes_at_infinity=int(g.rho_vanishes_at_infinity.sum()),
                     in_gstar_class=int(g.in_gstar_class.sum()), markov_equiv_gstar=int(g.markov_equiv_gstar.sum())))
by_k = pd.DataFrame(rows)
by_k.to_csv(RESULTS / "by_edge_count.csv", index=False)
min_exact = int(df[df.exact].edges.min())
df[["index", "name", "edges", "bowed", "rank", "in_gstar_class", "markov_equiv_gstar", "exact", "gap", "gap_hp",
    "gap_hp_klpoint", "gap_hp_residpoint", "resid", "exact_resid", "exact_kl", "exact_kl_hp", "separated",
    "separation", "kl_maxabsB", "resid_maxabsB", "rho_vanishes_at_infinity", "ci_vanish", "verma_vanish",
    "msep", "category", "gap_hp_fullrun", "gap_scan_min"]].to_csv(RESULTS / "support_table.csv", index=False)


# --- exact minimum-edge supports
def rep_from_B(Bl, s):
    B = np.array(Bl)
    A = np.eye(P) - B
    R = A.T @ SIGMA @ A
    O = np.diag(np.diag(R))
    for (i, j) in s.U:
        O[i, j] = O[j, i] = R[i, j]
    err = float(np.max(np.abs(covariance(B, O) - SIGMA)))
    ev = np.linalg.eigvalsh(O)
    inK = bool(np.max(np.abs(B)) <= 2 and ev[0] >= 0.05 and ev[-1] <= 10)
    return B, O, err, ev, inK


mins = []
for _, r in df[df.exact & (df.edges == min_exact)].iterrows():
    s = SUPPORTS[int(r["index"])]
    reps = []
    for Bl in sols.get(str(int(r["index"])), []):
        B, O, err, ev, inK = rep_from_B(Bl, s)
        reps.append(dict(B={f"{i+1}->{j+1}": round(float(B[i, j]), 6) for (i, j) in s.D},
                         Omega_offdiag={f"{i+1}<->{j+1}": round(float(O[i, j]), 6) for (i, j) in s.U},
                         Omega_diag=[round(float(x), 6) for x in np.diag(O)],
                         reconstruction_error=err, omega_eig=[float(ev[0]), float(ev[-1])], in_domain_K=inK))
    mins.append(dict(index=int(r["index"]), name=r["name"], bowed=int(r.bowed), rank=int(r["rank"]),
                     f_CI_vanishes=bool(r.f_CI_vanishes), f_V_vanishes=bool(r.f_V_vanishes),
                     vanishing_CI=ci_list(r.ci_vanish), vanishing_verma=verma_list(r.verma_vanish),
                     in_gstar_class=bool(r.in_gstar_class), markov_equiv_gstar=bool(r.markov_equiv_gstar),
                     gap_hp=float(r.gap_hp), resid=float(r.resid),
                     n_exact_solutions=int(r.n_exact_solutions) if not pd.isna(r.n_exact_solutions) else None,
                     representations=reps))
pd.DataFrame([{k: v for k, v in m.items() if k != "representations"} for m in mins]).to_csv(
    RESULTS / "min_edge_supports.csv", index=False)

# --- Markov class of G* at e* edges
mk = df[(df.markov_equiv_gstar == 1) & (df.edges == E_STAR)].copy()
mk["vanishing_CI"] = mk.ci_vanish.map(lambda b: " ".join(ci_list(b)))
mk["vanishing_verma"] = mk.verma_vanish.map(lambda b: " ".join(verma_list(b)))
mk_out = mk[["index", "name", "bowed", "rank", "vanishing_CI", "vanishing_verma", "f_V_vanishes", "in_gstar_class",
             "exact", "gap_hp", "resid", "separation"]]
mk_out.to_csv(RESULTS / "markov_class_e_star.csv", index=False)

# --- G* class members, all edge counts
gc = df[df.in_gstar_class == 1]
gc[["index", "name", "edges", "bowed", "rank", "exact", "gap_hp", "resid"]].to_csv(
    RESULTS / "gstar_class_members.csv", index=False)

# --- bound scan summary
scan["ratio_1e4_1e3"] = scan.gap_hp_M10000 / scan.gap_hp_M1000
scan_summary = {}
for k, g in scan.groupby("edges"):
    scan_summary[int(k)] = dict(n=len(g), min_gap_M1000=float(g.gap_hp_M1000.min()),
                                min_gap_M10000=float(g.gap_hp_M10000.min()),
                                min_ratio=float(g.ratio_1e4_1e3.min()), median_ratio=float(g.ratio_1e4_1e3.median()),
                                n_ratio_below_0_99=int((g.ratio_1e4_1e3 < 0.99).sum()))

nl = df[df.near_limit]
near_limit_summary = dict(
    n=len(nl), by_edges={int(k): int(v) for k, v in nl.groupby("edges").size().items()},
    n_full_run_near_limit_found_exact_by_scan=int(((df.scan == "near_limit") & df.exact).sum()),
    found_exact_by_scan=df[(df.scan == "near_limit") & df.exact][["name", "gap_hp_fullrun", "gap_scan_min"]].to_dict("records"),
    n_decreasing_ratio_below_0_5=int((nl.decade_ratio_1e3_1e4 < 0.5).sum()),
    decreasing=nl[nl.decade_ratio_1e3_1e4 < 0.5][["name", "gap_hp_M100", "gap_hp_M1000", "gap_hp_M10000"]].to_dict("records"),
    n_separated=int(nl.separated.sum()), min_gap=float(nl.gap_hp.min()) if len(nl) else None)

# --- summary
le = df[df.edges <= E_STAR]
le5 = df[df.edges <= E_STAR + 1]
summary = dict(
    generating=dict(B={"1->2": B_STAR[0, 1], "2->3": B_STAR[1, 2], "3->4": B_STAR[2, 3]},
                    Omega_diag=np.diag(O_STAR).tolist(), Omega_24=O_STAR[1, 3], Sigma_star=SIGMA.tolist(),
                    l_sat=L_SAT),
    sigma_star_zero_partial_correlations=[ci_name(st) for st, z in zip(CI_STATEMENTS, ZERO_CI) if z],
    sigma_star_min_abs_nonzero_partial_correlation=float(np.min(np.abs(CI_AT_STAR[~ZERO_CI]))),
    sigma_star_zero_verma=["".join(str(v + 1) for v in pm) for pm, z in zip(VERMA_PERMS, ZERO_V) if z],
    sigma_star_min_abs_nonzero_verma_normalized=float(np.min(np.abs(V_AT_STAR[~ZERO_V]))),
    n_supports=len(df), min_exact_edge_count=min_exact, e_star=E_STAR,
    n_exact_total=int(df.exact.sum()),
    n_exact_min_edge=len(mins),
    all_min_edge_in_gstar_class=all(m["in_gstar_class"] for m in mins),
    gstar_class_size_by_edges={int(k): int(v) for k, v in gc.groupby("edges").size().items()},
    gstar_class_members=gc.name.tolist(),
    gstar_class_e_star_not_exact=gc[(gc.edges == E_STAR) & (~gc.exact)].name.tolist(),
    markov_class_size_by_edges={int(k): int(v) for k, v in df[df.markov_equiv_gstar == 1].groupby("edges").size().items()},
    markov_e_star_outside_class=mk[mk.in_gstar_class == 0].name.tolist(),
    markov_e_star_outside_class_exact=mk[(mk.in_gstar_class == 0) & mk.exact].name.tolist(),
    nonexact_le_e_star=int((~le.exact).sum()),
    nonexact_le_e_star_separated=int((~le.exact & le.separated).sum()),
    nonexact_le_e_star_min_separation=float(le[~le.exact].separation.min()),
    nonexact_e_star_plus_1_unseparated=int(((le5.edges == E_STAR + 1) & ~le5.exact & ~le5.separated).sum()),
    min_gap_hp_nonexact_le_e_star=float(le[~le.exact].gap_hp.min()),
    argmin_gap_hp_nonexact_le_e_star=le[~le.exact].sort_values("gap_hp").name.iloc[0],
    min_gap_hp_nonexact_e_star_plus_1=float(le5[(le5.edges == E_STAR + 1) & ~le5.exact].gap_hp.min()),
    argmin_gap_hp_nonexact_e_star_plus_1=le5[(le5.edges == E_STAR + 1) & ~le5.exact].sort_values("gap_hp").name.iloc[0],
    min_gap_hp_nonexact_unseparated_e_star_plus_1=float(le5[(le5.edges == E_STAR + 1) & ~le5.exact & ~le5.separated].gap_hp.min()),
    max_gap_hp_exact=float(df[df.exact].gap_hp.max()),
    max_resid_exact=float(df[df.exact].resid.max()),
    min_resid_nonexact_le_e_star=float(le[~le.exact].resid.min()),
    min_gap_hp_nonexact_all=float(df[~df.exact].gap_hp.min()),
    argmin_gap_hp_nonexact_all=df[~df.exact].sort_values("gap_hp").name.iloc[0],
    n_nonexact_gap_hp_below_1e6=int((~df.exact & (df.gap_hp < 1e-6)).sum()),
    nonexact_gap_hp_below_1e6_by_edges={int(k): int(v) for k, v in df[~df.exact & (df.gap_hp < 1e-6)].groupby("edges").size().items()},
    n_nonexact_gap_hp_below_1e10=int((~df.exact & (df.gap_hp < EXACT_TOL)).sum()),
    n_disagree_float=int(df.disagree_float.sum()), disagree_float=df[df.disagree_float].name.tolist(),
    n_disagree_hp=int(df.disagree_hp.sum()),
    disagree_hp=df[df.disagree_hp][["name", "gap_hp", "resid", "kl_maxabsB", "resid_maxabsB"]].to_dict("records"),
    nonexact_gap_hp_below_1e6_at_bound=int((~df.exact & (df.gap_hp < 1e-6) & (df.kl_at_bound | (df.resid_maxabsB > 999))).sum()),
    n_disagree_independent_kl=int((df.exact_kl_independent != df.exact_resid).sum()),
    n_negative_float_gaps=int((df.gap < -1e-12).sum()),
    min_gap_hp_all=float(df.gap_hp.min()),
    max_abs_float_minus_hp_gap_interior=float((df[~df.kl_at_bound].gap - df[~df.kl_at_bound].gap_hp_klpoint).abs().max()),
    n_rho_vanishes_at_infinity=int(df.rho_vanishes_at_infinity.sum()),
    rho_at_infinity_le_e_star=df[(df.edges <= E_STAR) & (df.rho_vanishes_at_infinity == 1)].name.tolist(),
    rho_at_infinity_le_e_star_min_gap_hp=float(df[(df.edges <= E_STAR) & (df.rho_vanishes_at_infinity == 1)].gap_hp.min()),
    bound_scan=scan_summary,
    near_limit=near_limit_summary,
    n_le5_gap_lowered_by_scan=int((df.edges <= 5).mul(df.gap_scan_min < df.gap_hp_fullrun - 1e-12).sum()),
    max_le5_gap_lowering_by_scan=float((df[df.edges <= 5].gap_hp_fullrun - df[df.edges <= 5].gap_hp).max()),
    exact_resid_maxabsB_max=float(df[df.exact].resid_maxabsB.max()),
    rank_float_mismatches=int((df["rank"] != df.rank_float).sum()),
    msep_ci_mismatches=int((df.msep_matches_ci == 0).sum()),
    exact_but_separated=int((df.exact & df.separated).sum()),
    kl_fail_starts=int(df.kl_fail.sum()),
    min_edge_supports=mins,
)
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(by_k.to_markdown(index=False))
print()
print(pd.DataFrame([{k: v for k, v in m.items() if k not in ("representations", "vanishing_CI")} for m in mins]).to_markdown(index=False))
print()
print(mk_out.to_markdown(index=False))
print()
print(json.dumps({k: v for k, v in summary.items() if k not in ("generating",)}, indent=1, default=str))

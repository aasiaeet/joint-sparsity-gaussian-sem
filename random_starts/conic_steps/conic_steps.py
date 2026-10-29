"""The conic Omega-steps of the random-start study and the conditions of Proposition 15.

Supplementary Section S18 (Steps).  Four runs of the study used the conic Omega-step:
refit-started runs of G2, n = 250, sample 49 (refit start 50) and sample 92 (refit
starts 44, 45, 46).  They are rerun as in the study (same data, same refit starts, one
shared NoiseProx per sample used in the order of the study, same tuning and steps, cap
20,000) with a read-only hook after every Omega-step.  With G = grad_Omega l(B^{t+1}, Omega^t)
(trace inner product), Z = Omega^t - beta G and V = Omega^{t+1}, the hook records for every
cycle:

 (i)  the change of the proximal objective dQ = Q(V) - Q(Omega^t), where
      Q(V) = <G, V - Omega^t> + ||V - Omega^t||_F^2 / (2 beta) + lambda P_Omega(V).
      Proposition 15 requires dQ <= 0 (and V feasible).
 (ii) the residual r = min ||G + (V - Omega^t)/beta + xi||_F over xi in the limiting
      subdifferential of g_Omega = lambda P_Omega + indicator(spec(V) in [0.05, 10]) at V,
      and the ratio r / ||theta^{t+1} - theta^t||_F.  Three versions:
      * strict:  at the returned V exactly (an off-diagonal entry is zero only if it is
                 exactly 0; an eigenvalue is active only if it equals a bound exactly).
                 This is the condition of Proposition 15 as stated.
      * relaxed (columns *_snapped): off-diagonal entries |V_ij| <= 1e-9 set to 0 and
                 treated as zeros, and eigenvalues within 1e-5 of a bound treated as active
                 (normal cone spanned by V's own eigenvector); same denominator.
      * exact:   at V*, the exact Omega-step.  When every nonzero off-diagonal entry of the
                 entrywise solution C has magnitude >= delta, the penalty is locally constant
                 and V* is the Frobenius projection of Z onto {C's zero pattern} intersected
                 with the spectral box, computed by Dykstra's algorithm.  The residual at V*
                 certifies optimality, since Q is strongly convex.
      The same residual is computed for the B-step (exact dynamic program), with the normal
      cone of the order subspace and of the box |B_ij| <= 2.
 Q is mu-strongly convex with mu = 1/beta - lambda/delta^2, so Q(V) - Q(V*) >= 0 is the
 suboptimality of the returned step; ||V - V*||_F is reported as well.

A second pass reruns the four runs with the exact step in place of the conic solve
(accepted only with a KKT residual <= 1e-9; otherwise the conic solve is kept and counted).
A third pass reruns each run alone with a fresh NoiseProx: cvxpy caches the Clarabel
solver across solves of the same Problem object (warm_start=True by default), and the
study shared one NoiseProx per sample.

Seeds: data SeedSequence([20260925, 1, 250, sample]); refit solver
SeedSequence([20260925, 1, 250, sample, 1]).

Usage: python3 conic_steps.py [OUT_DIR]   (default results/)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "continuation"))
import runner as ec  # noqa: E402

RUNS = {49: [50], 92: [44, 45, 46]}
D, N = "G2", 250
EPS_ZERO, EPS_EIG = 1e-9, 1e-5


def pen_omega(V, delta):
    return float(ec.penalty(V[ec.IU], delta).sum())


def pen_B(B, delta):
    return float(ec.penalty(B[ec.OFFDIAG], delta).sum())


def qprime(t, delta):
    """Derivative of q_delta at t != 0."""
    return (2 * np.sign(t) / delta - 2 * t / delta ** 2) if abs(t) < delta else 0.0


def prox_obj_change(G, Vold, V, beta, lam, delta):
    dO = V - Vold
    return float(np.sum(G * dO)) + float(np.sum(dO * dO)) / (2 * beta) \
        + lam * (pen_omega(V, delta) - pen_omega(Vold, delta))


def omega_residual(G, Vold, V, beta, lam, delta, lo, hi, zero_mask=None, act_tol=0.0):
    """min ||G + (V - Vold)/beta + lam*gradP(V) + N||_F over the subgradient interval of q at
    zero entries (|s| <= 2/delta, entering as lam*s/2 on (i,j) and (j,i)) and over N in the
    normal cone of {lo I <= V <= hi I} (one-dimensional active eigenspaces: N = -a u u^T,
    a >= 0 at the floor, +c u u^T, c >= 0 at the ceiling).  Bounded least squares (BVLS)."""
    p = V.shape[0]
    R0 = G + (V - Vold) / beta
    cols, lb, ub = [], [], []
    nzero = 0
    for i in range(p):
        for j in range(i + 1, p):
            is_zero = (V[i, j] == 0.0) if zero_mask is None else bool(zero_mask[i, j])
            if is_zero:
                E = np.zeros((p, p)); E[i, j] = E[j, i] = lam / 2
                cols.append(E.ravel()); lb.append(-2 / delta); ub.append(2 / delta)
                nzero += 1
            else:
                R0[i, j] += lam * qprime(V[i, j], delta) / 2
                R0[j, i] += lam * qprime(V[i, j], delta) / 2
    ev, U = np.linalg.eigh(V)
    act_lo = [k for k in range(p) if abs(ev[k] - lo) <= act_tol]
    act_hi = [k for k in range(p) if abs(ev[k] - hi) <= act_tol]
    for k in act_lo:
        cols.append((-np.outer(U[:, k], U[:, k])).ravel()); lb.append(0.0); ub.append(np.inf)
    for k in act_hi:
        cols.append(np.outer(U[:, k], U[:, k]).ravel()); lb.append(0.0); ub.append(np.inf)
    info = dict(zero_pairs=nzero, active_floor=len(act_lo), active_ceiling=len(act_hi),
                multidim_active=bool(len(act_lo) > 1 or len(act_hi) > 1))
    if not cols:
        return float(np.linalg.norm(R0)), info
    A = np.array(cols).T
    res = lsq_linear(A, -R0.ravel(), bounds=(np.array(lb), np.array(ub)), method="bvls", tol=1e-15)
    return float(np.linalg.norm(R0.ravel() + A @ res.x)), info


def b_residual(gb, Bold, Bn, alpha, lam, delta, M, order):
    """B-step residual with the normal cone of the DP order's subspace (coordinates not
    forward in `order` are free) and of the box |B_ij| <= M."""
    p = Bn.shape[0]
    pos = {v: k for k, v in enumerate(order)}
    r2 = 0.0
    for i in range(p):
        for j in range(p):
            if i == j or pos[i] > pos[j]:
                continue
            g = gb[i, j] + (Bn[i, j] - Bold[i, j]) / alpha
            u = Bn[i, j]
            if u == 0.0:
                r = max(abs(g) - 2 * lam / delta, 0.0)
            else:
                g = g + lam * qprime(u, delta)
                r = max(0.0, np.sign(u) * g) if abs(u) >= M else abs(g)
            r2 += r * r
    return math.sqrt(r2)


def relax(V, eps_zero=EPS_ZERO):
    """Returned V with off-diagonal entries |V_ij| <= eps_zero set to 0 (and flagged as zeros);
    eigenvalues are not moved: the residual call treats those within EPS_EIG of a bound as
    active, with V's own eigenvectors."""
    p = V.shape[0]
    zmask = np.zeros((p, p), bool)
    for i in range(p):
        for j in range(i + 1, p):
            zmask[i, j] = zmask[j, i] = abs(V[i, j]) <= eps_zero
    return np.where(zmask, 0.0, V), zmask


def analytic_candidate(z, beta, lam, delta):
    z = (z + z.T) / 2
    c = ec.rs.scalar_prox(z, beta / 2, lam, delta)
    np.fill_diagonal(c, np.diag(z))
    return c


def dykstra_projection(Z, zmask, lo, hi, iters=20000, tol=1e-15):
    """Frobenius projection of Z onto {V : V_ij = 0 on zmask} intersected with the spectral box,
    by Dykstra's algorithm (exact projections onto each set)."""
    X = Z.copy()
    Pa = np.zeros_like(Z)
    Pb = np.zeros_like(Z)
    for it in range(iters):
        Y = np.where(zmask, 0.0, X + Pa)            # subspace: projection is linear
        Pa = X + Pa - Y
        W = Y + Pb
        ev, U = np.linalg.eigh(W)
        Xn = (U * np.clip(ev, lo, hi)) @ U.T
        Xn = (Xn + Xn.T) / 2
        Pb = W - Xn
        if np.linalg.norm(Xn - X) < tol and np.max(np.abs(Xn[zmask]), initial=0.0) < tol:
            X = Xn
            break
        X = Xn
    return np.where(zmask, 0.0, X), it + 1


def exact_step(z, beta, lam, delta, lo, hi):
    """Exact Omega-step when the penalty is locally constant on the solution's nonzero
    off-diagonal entries.  Returns (V*, zero_mask, applies).  If the entrywise solution C is
    inside the spectral box it is exact.  Otherwise, keeping C's zero pattern and assuming
    every nonzero off-diagonal entry has magnitude >= delta, the step reduces to the Frobenius
    projection of Z onto {zeros of C} intersected with the box (Dykstra; one eigenvalue clip
    when the zeros make C block diagonal).  `applies` requires those magnitudes at the
    result; optimality is then verified separately by the residual (KKT) check."""
    C = analytic_candidate(z, beta, lam, delta)
    ev = np.linalg.eigvalsh(C)
    p = C.shape[0]
    zmask = np.zeros((p, p), bool)
    for i in range(p):
        for j in range(i + 1, p):
            zmask[i, j] = zmask[j, i] = C[i, j] == 0.0
    if ev[0] >= lo and ev[-1] <= hi:
        return C, zmask, True
    off = np.abs(C[ec.IU])
    ok = bool(np.all((off == 0) | (off >= delta)))
    V, _ = dykstra_projection((z + z.T) / 2, zmask, lo, hi)
    offV = np.abs(V[ec.IU])
    ok = ok and bool(np.all((offV == 0) | (offV >= delta)))
    return V, zmask, ok


class ExactProx(ec.NoiseProx):
    """NoiseProx with the closed-form exact step in place of the conic solve when it applies."""

    def conic(self, z):
        V, _, ok = exact_step(z, self.gamma, self.lam, self.delta, self.floor, self.ceiling)
        if ok:
            # accept only with a KKT certificate: residual of the step at V (G and Omega^t
            # enter only through Z = Omega^t - beta G, so take G = -Z/beta, Omega^t = 0)
            zs = (z + z.T) / 2
            r, _ = omega_residual(-zs / self.gamma, np.zeros_like(zs), V, self.gamma, self.lam,
                                  self.delta, self.floor, self.ceiling,
                                  zero_mask=(V == 0) & ~np.eye(len(V), dtype=bool), act_tol=1e-10)
            ok = r <= 1e-9
            self.max_kkt = max(getattr(self, "max_kkt", 0.0), r)
        if ok:
            self.statuses["exact_clip"] = self.statuses.get("exact_clip", 0) + 1
            return V
        return super().conic(z)


def instrumented_pass(cfg, pop, saved):
    lam, delta, alpha, beta = cfg["lam"], cfg["delta"], cfg["alpha"], cfg["beta"]
    lo, hi = cfg["omega_spectrum"]
    M = cfg["coefficient_bound"]
    emin, e_star = set(pop["E_min_indices"]), pop["e_star"]
    rows, runs, starts = [], [], {}
    for ds, ks in RUNS.items():
        S = ec.sample_S(cfg, D, N, ds)
        refits = ec.refit_starts(cfg, S, D, N, ds)
        starts[ds] = (S, refits)
        prox = ec.NoiseProx(ec.P, beta, lam, delta, lo, hi)     # shared within a sample, as in the study
        for k in ks:
            B0, O0, sup, crit = refits[k]
            seen = dict(prox.statuses)
            cyc = []

            def hook(t, b, o, gb, go_mid, bn, on, z, prox_, order):
                before = dict(seen)
                seen.clear(); seen.update(prox_.statuses)
                diff = {kk: prox_.statuses.get(kk, 0) - before.get(kk, 0) for kk in prox_.statuses}
                conic = diff.get("analytic_interior", 0) == 0
                C = analytic_candidate(z, beta, lam, delta)
                evC, evV = np.linalg.eigvalsh(C), np.linalg.eigvalsh(on)
                dB, dO = bn - b, on - o
                step = math.sqrt(float(np.sum(dB * dB)) + float(np.sum(dO * dO)))
                r_strict, inf_s = omega_residual(go_mid, o, on, beta, lam, delta, lo, hi)
                Vs, zm = relax(on)
                r_snap, inf_n = omega_residual(go_mid, o, Vs, beta, lam, delta, lo, hi,
                                               zero_mask=zm, act_tol=EPS_EIG)
                Vx, zx, ok = exact_step(z, beta, lam, delta, lo, hi)
                r_exact, inf_x = omega_residual(go_mid, o, Vx, beta, lam, delta, lo, hi,
                                                zero_mask=zx, act_tol=1e-10)
                rB = b_residual(gb, b, bn, alpha, lam, delta, M, order)
                dQ = prox_obj_change(go_mid, o, on, beta, lam, delta)
                dQx = prox_obj_change(go_mid, o, Vx, beta, lam, delta)
                cyc.append(dict(
                    dataset=ds, start=k, cycle=t, branch="conic" if conic else "analytic",
                    conic_status=prox_.problem.status if conic else "",
                    conic_iters=prox_.problem.solver_stats.num_iters if conic else np.nan,
                    spectrum_repaired=bool(diff.get("spectrum_repaired", 0)),
                    candidate_min_eig=float(evC[0]), candidate_max_eig=float(evC[-1]),
                    new_min_eig=float(evV[0]), new_max_eig=float(evV[-1]),
                    ceiling_gap=float(hi - evV[-1]),
                    feasible=bool(evV[0] >= lo and evV[-1] <= hi),
                    offdiag_new=json.dumps([float(on[i, j]) for i, j in zip(*ec.IU)]),
                    cycle_length=step, dB=float(np.linalg.norm(dB)), dOmega=float(np.linalg.norm(dO)),
                    dQ=dQ, dQ_positive=bool(dQ > 0),
                    res_strict=r_strict, ratio_strict=r_strict / step,
                    snap_change=float(np.linalg.norm(Vs - on)),
                    res_snapped=r_snap, ratio_snapped=r_snap / step,
                    snapped_zero_pairs=inf_n["zero_pairs"], snapped_active_floor=inf_n["active_floor"],
                    snapped_active_ceiling=inf_n["active_ceiling"],
                    exact_closed_form_applies=ok, res_exact=r_exact,
                    dist_to_exact=float(np.linalg.norm(on - Vx)),
                    Q_minus_Qexact=dQ - dQx, dQ_exact=dQx,
                    res_B=rB, ratio_B=rB / step,
                    ratio_prop15_strict=(rB + r_strict) / step,
                    ratio_prop15_snapped=(rB + r_snap) / step))

            t0 = time.monotonic()
            b, o, info, _ = ec.run_alternating_snap(S, B0, O0, cfg, prox=prox, snap_at=-1, hook=hook)
            old = saved[(saved.design == D) & (saved.n == N) & (saved.dataset == ds) &
                        (saved.start_type == "refit") & (saved.start == k)].iloc[0]
            desc = ec.describe(b, o, cfg, emin, e_star)
            cc = [r for r in cyc if r["branch"] == "conic"]
            runs.append(dict(
                dataset=ds, start=k, start_support=sup, start_support_match=sup == old.start_support,
                cycles=info["cycles"], cycles_saved=int(old.cycles), status=info["status"],
                objective=info["objective"], objective_absdiff_vs_saved=abs(info["objective"] - old.objective),
                support_thr=desc["support_thr"], support_thr_match=desc["support_thr"] == old.support_thr,
                support_raw_match=desc["support_raw"] == old.support_raw,
                max_objective_increase=info["max_objective_increase"],
                max_objective_increase_saved=float(old.max_objective_increase),
                omega_steps=len(cyc), conic_steps=len(cc), conic_steps_saved=int(old.noise_conic_calls),
                conic_statuses=";".join(sorted({r["conic_status"] for r in cc})),
                spectral_repairs=int(sum(r["spectrum_repaired"] for r in cc)),
                n_dQ_positive=int(sum(r["dQ_positive"] for r in cc)), max_dQ=max(r["dQ"] for r in cc),
                min_cycle_length=min(r["cycle_length"] for r in cc),
                max_cycle_length=max(r["cycle_length"] for r in cc),
                max_ceiling_gap=max(r["ceiling_gap"] for r in cc),
                max_res_strict=max(r["res_strict"] for r in cc),
                max_ratio_strict=max(r["ratio_strict"] for r in cc),
                max_res_snapped=max(r["res_snapped"] for r in cc),
                max_ratio_snapped=max(r["ratio_snapped"] for r in cc),
                max_ratio_prop15_strict=max(r["ratio_prop15_strict"] for r in cc),
                max_ratio_prop15_snapped=max(r["ratio_prop15_snapped"] for r in cc),
                closed_form_applies_all=bool(all(r["exact_closed_form_applies"] for r in cc)),
                max_res_exact=max(r["res_exact"] for r in cc),
                max_dQ_exact=max(r["dQ_exact"] for r in cc),
                max_dist_to_exact=max(r["dist_to_exact"] for r in cc),
                max_Q_minus_Qexact=max(r["Q_minus_Qexact"] for r in cc),
                min_Q_minus_Qexact=min(r["Q_minus_Qexact"] for r in cc),
                max_res_B=max(r["res_B"] for r in cyc),
                runtime_seconds=time.monotonic() - t0))
            rows.extend(cyc)
    return pd.DataFrame(rows), pd.DataFrame(runs), starts


def exact_pass(cfg, pop, starts, conic_runs):
    lam, delta, beta = cfg["lam"], cfg["delta"], cfg["beta"]
    lo, hi = cfg["omega_spectrum"]
    emin, e_star = set(pop["E_min_indices"]), pop["e_star"]
    out = []
    for ds, ks in RUNS.items():
        S, refits = starts[ds]
        prox = ExactProx(ec.P, beta, lam, delta, lo, hi)
        for k in ks:
            B0, O0, sup, _ = refits[k]
            before = dict(prox.statuses)
            b, o, info, _ = ec.run_alternating_snap(S, B0, O0, cfg, prox=prox, snap_at=-1)
            desc = ec.describe(b, o, cfg, emin, e_star)
            base = conic_runs[(conic_runs.dataset == ds) & (conic_runs.start == k)].iloc[0]
            out.append(dict(dataset=ds, start=k, exact_cycles=info["cycles"], exact_status=info["status"],
                            exact_objective=info["objective"],
                            exact_minus_conic_objective=info["objective"] - base.objective,
                            exact_support_thr=desc["support_thr"],
                            same_support_as_conic=desc["support_thr"] == base.support_thr,
                            exact_support_raw=desc["support_raw"],
                            exact_max_objective_increase=info["max_objective_increase"],
                            exact_steps=prox.statuses.get("exact_clip", 0) - before.get("exact_clip", 0),
                            conic_fallbacks=sum(v - before.get(kk, 0) for kk, v in prox.statuses.items()
                                                if kk not in ("analytic_interior", "exact_clip")),
                            max_kkt_residual_exact_steps_so_far=getattr(prox, "max_kkt", np.nan)))
    return pd.DataFrame(out)


def fresh_prox_pass(cfg, pop, starts, conic_runs):
    """Each run alone with a fresh NoiseProx (no earlier conic solves in the process), to test
    whether the conic step depends on the history of the cached cvxpy/Clarabel solver."""
    lam, delta, beta = cfg["lam"], cfg["delta"], cfg["beta"]
    lo, hi = cfg["omega_spectrum"]
    emin, e_star = set(pop["E_min_indices"]), pop["e_star"]
    out = []
    for ds, ks in RUNS.items():
        S, refits = starts[ds]
        for k in ks:
            B0, O0, _, _ = refits[k]
            prox = ec.NoiseProx(ec.P, beta, lam, delta, lo, hi)
            b, o, info, _ = ec.run_alternating_snap(S, B0, O0, cfg, prox=prox, snap_at=-1)
            desc = ec.describe(b, o, cfg, emin, e_star)
            base = conic_runs[(conic_runs.dataset == ds) & (conic_runs.start == k)].iloc[0]
            out.append(dict(dataset=ds, start=k, fresh_cycles=info["cycles"], shared_cycles=int(base.cycles),
                            fresh_objective=info["objective"],
                            fresh_minus_shared_objective=info["objective"] - base.objective,
                            fresh_support_thr=desc["support_thr"],
                            same_support=desc["support_thr"] == base.support_thr,
                            fresh_max_objective_increase=info["max_objective_increase"],
                            fresh_conic_steps=info["noise_conic_calls"]))
    return pd.DataFrame(out)


def main():
    t_all = time.monotonic()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    cfg = ec.load_cfg()                  # cap 20,000 as in the study
    pop = ec.load_population()[D]
    saved = pd.read_csv(ec.STUDY_RESULTS / "runs.csv.gz")
    cyc, runs, starts = instrumented_pass(cfg, pop, saved)
    ex = exact_pass(cfg, pop, starts, runs)
    fr = fresh_prox_pass(cfg, pop, starts, runs)
    fr.to_csv(out / "conic_runs_fresh_solver.csv", index=False)
    cyc.to_csv(out / "conic_steps_per_cycle.csv", index=False)
    runs.to_csv(out / "conic_runs_summary.csv", index=False)
    ex.to_csv(out / "conic_runs_exact_step.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print(runs.T.to_string())
    print(ex.T.to_string())
    print(fr.T.to_string())
    print(cyc[["dataset", "start", "cycle", "conic_iters", "ceiling_gap", "cycle_length", "dQ",
               "res_strict", "ratio_strict", "res_snapped", "ratio_snapped", "res_exact",
               "dist_to_exact", "Q_minus_Qexact", "res_B"]].to_string())
    print(f"total {time.monotonic() - t_all:.1f}s")


if __name__ == "__main__":
    main()

"""Local convergence of the alternating algorithm and refitting on four variables.

Section 7.4 of the main article (Figure 5), Section S18 of the Supplementary
Material ("Local convergence and refitting", Table S2), and the two-variable
exact-fiber example of Figure 3 (Section 5.3).

Parent-row convention: Sigma = (I-B)^(-T) Omega (I-B)^(-1); B[i, j] != 0 is
the edge i -> j.  One cycle of Algorithm 1 is the B-step (6), solved exactly
by the scalar proximal formula and dynamic programming over vertex subsets,
followed by the Omega-step (7) evaluated at the new B, solved by the scalar
formula when its solution lies in the spectral box and otherwise by a
strongly convex conic program.  Starting near a fixed representation, the
population run illustrates the contraction of Theorem 16; the sample runs,
thresholded at delta and refitted by local likelihood maximization on the
selected support, illustrate the refit rate of Theorem 10(iii).

Floating-point computations; the conditions of the theorems are checked
numerically, not certified.

Usage:  python3 alternating.py [--out DIR]     (about 30 s on one core)
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import platform
import time
import warnings

import cvxpy as cp
import numpy as np
import scipy
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIG = dict(p=4, alpha=0.02, beta=0.08, lam=0.008, delta=0.15,
              arrow_bound=2.0, noise_floor=0.1, noise_ceiling=4.0,
              ns=[500, 2000, 8000], seeds=list(range(1201, 1213)),
              distributions=["Gaussian", "Student t7"],
              population_seed=20260908, initialization_sd=0.0001,
              max_cycles=4000, step_tolerance=1e-8, numerical_zero=1e-8,
              trajectory_every=25)


def penalty(x, delta):
    a = np.minimum(np.abs(x) / delta, 1.0)
    return 2 * a - a * a


def scalar_prox(z, gamma, lam, delta, bound=np.inf):
    """Unique scalar MCP prox for gamma < delta^2 / (2 lambda)."""
    if not 0 < gamma < delta * delta / (2 * lam):
        raise ValueError("Scalar proximal model is not strongly convex")
    v = np.abs(z)
    cut = 2 * gamma * lam / delta
    slope = 1 - 2 * gamma * lam / delta**2
    ans = np.where(v <= cut, 0, np.where(v < delta, (v-cut)/slope, v))
    return np.clip(np.sign(z) * ans, -bound, bound)


def arrow_prox(z, gamma, lam, delta, bound=2.0):
    """B-step (6): exact minimization over all acyclic B by dynamic programming over vertex subsets."""
    p = len(z)
    vals = scalar_prox(z, gamma, lam, delta, bound)
    np.fill_diagonal(vals, 0)
    gain = z*z/(2*gamma) - ((vals-z)**2/(2*gamma) + lam*penalty(vals, delta))
    np.fill_diagonal(gain, 0)
    dp = np.full(1 << p, -np.inf)
    last = np.zeros(1 << p, dtype=int)
    dp[0] = 0
    for mask in range(1, 1 << p):
        nodes = [i for i in range(p) if mask & (1 << i)]
        for j in nodes:
            prev = mask ^ (1 << j)
            score = dp[prev] + sum(gain[i, j] for i in nodes if i != j)
            if score > dp[mask]:
                dp[mask], last[mask] = score, j
    order, mask = [], (1 << p)-1
    while mask:
        j = int(last[mask])
        order.append(j)
        mask ^= 1 << j
    order.reverse()
    out = np.zeros_like(z)
    for k, j in enumerate(order):
        for i in order[:k]:
            out[i, j] = vals[i, j]
    return out, float(dp[-1]), order


class NoiseProx:
    def __init__(self, p, gamma, lam, delta, floor=0.1, ceiling=4.0):
        self.p, self.gamma, self.lam, self.delta = p, gamma, lam, delta
        self.floor, self.ceiling = floor, ceiling
        self.statuses = {}
        self.problem = None

    def conic(self, z):
        if self.problem is None:
            self.z = cp.Parameter((self.p, self.p), symmetric=True)
            self.o = cp.Variable((self.p, self.p), symmetric=True)
            g, l, d = self.gamma, self.lam, self.delta
            terms = [cp.sum_squares(cp.diag(self.o)-cp.diag(self.z))/(2*g)]
            for i in range(self.p):
                for j in range(i+1, self.p):
                    t = self.o[i, j]
                    terms += [(1/g-l/d**2)*cp.square(t)-2*self.z[i,j]*t/g
                              +2*l*cp.abs(t)/d
                              +l*cp.square(cp.pos(cp.abs(t)-d))/d**2]
            self.problem = cp.Problem(cp.Minimize(sum(terms)),
                [self.o >> self.floor*np.eye(self.p), self.o << self.ceiling*np.eye(self.p)])
            assert self.problem.is_dcp()
        self.z.value = (z+z.T)/2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.problem.solve(solver="CLARABEL", tol_gap_abs=1e-10,
                               tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=200)
        status = self.problem.status
        self.statuses[status] = self.statuses.get(status, 0)+1
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("Noise proximal solve: " + status)
        return np.asarray(self.o.value)

    def __call__(self, z):
        z = (z+z.T)/2
        # Each unordered pair appears twice in the Frobenius square.
        # Its scalar gamma is beta/2, giving threshold beta*lambda/delta.
        out = scalar_prox(z, self.gamma/2, self.lam, self.delta)
        np.fill_diagonal(out, np.diag(z))
        eig = np.linalg.eigvalsh(out)
        if eig[0] >= self.floor and eig[-1] <= self.ceiling:
            self.statuses["analytic_interior"] = self.statuses.get("analytic_interior", 0)+1
            return out
        return self.conic(z)


def covariance(b, o):
    a = np.linalg.inv(np.eye(len(b))-b)
    return a.T @ o @ a


def loss_grad(b, o, s):
    a = np.eye(len(b))-b
    inv = np.linalg.inv(o)
    residual = a.T @ s @ a
    value = 0.5*(np.linalg.slogdet(o)[1]+np.trace(inv@residual))
    gb = -s @ a @ inv
    go = 0.5*(inv-inv@residual@inv)
    return float(value), gb, (go+go.T)/2


def score(b, o, s, c=CONFIG):
    return loss_grad(b, o, s)[0] + c["lam"]*(penalty(b, c["delta"]).sum()
        + penalty(o[np.triu_indices(len(b), 1)], c["delta"]).sum())


def target():
    b, o = np.zeros((4, 4)), np.eye(4)
    b[0,1], b[0,2], b[1,2], b[2,3] = 0.9, 0.8, -0.75, 0.85
    o[0,3] = o[3,0] = 0.3
    return b, o


def distance(b, o, bt, ot):
    return float(np.sqrt(np.sum((b-bt)**2)+np.sum((o-ot)**2)))


def acyclic(b):
    edges = np.abs(b) > 0
    degree = edges.sum(axis=0).astype(int)
    remaining = set(range(len(b)))
    while remaining:
        zero = [j for j in remaining if degree[j] == 0]
        if not zero:
            return False
        for j in zero:
            remaining.remove(j)
            degree -= edges[j].astype(int)
    return True


def support(b, o, cut):
    mb = np.abs(b) >= cut if cut > 0 else np.abs(b) > 0
    mo = np.abs(o) >= cut if cut > 0 else np.abs(o) > 0
    np.fill_diagonal(mb, False)
    np.fill_diagonal(mo, False)
    return mb, mo


def support_equal(b, o, bt, ot, cut):
    mb, mo = support(b, o, cut)
    return bool(np.array_equal(mb, bt != 0) and
                np.array_equal(mo[np.triu_indices(len(o), 1)], ot[np.triu_indices(len(o), 1)] != 0))


def run_alternating(s, seed, run_id, bt, ot, c=CONFIG):
    rng = np.random.default_rng(seed)
    b = bt.copy()
    allowed = np.triu_indices(c["p"], 1)
    b[allowed] += rng.normal(scale=c["initialization_sd"], size=len(allowed[0]))
    noise = rng.normal(scale=c["initialization_sd"], size=ot.shape)
    o = ot+(noise+noise.T)/2
    prox = NoiseProx(c["p"], c["beta"], c["lam"], c["delta"],
                     c["noise_floor"], c["noise_ceiling"])
    trajectory = []
    initial = distance(b, o, bt, ot)
    min_eig, max_eig = np.inf, -np.inf
    max_increase, max_block_increase = -np.inf, -np.inf
    all_acyclic, first_support = True, False
    old_score = score(b, o, s)
    status = "cycle_cap"
    started = time.monotonic()
    for t in range(c["max_cycles"]+1):
        eig = np.linalg.eigvalsh(o)
        min_eig, max_eig = min(min_eig, eig[0]), max(max_eig, eig[-1])
        all_acyclic = all_acyclic and acyclic(b)
        if t % c["trajectory_every"] == 0 or t == c["max_cycles"]:
            trajectory.append(dict(run_id=run_id, iteration=t, objective=old_score,
                frobenius_error=distance(b,o,bt,ot), minimum_noise_eigenvalue=eig[0]))
        if t == c["max_cycles"]:
            break
        _, gb, _ = loss_grad(b, o, s)
        bn, _, _ = arrow_prox(b-c["alpha"]*gb, c["alpha"], c["lam"],
                               c["delta"], c["arrow_bound"])
        after_b = score(bn, o, s)
        # Gauss-Seidel: the Omega-gradient is evaluated at the new B.
        _, _, go_new_b = loss_grad(bn, o, s)
        on = prox(o-c["beta"]*go_new_b)
        new_score = score(bn, on, s)
        max_increase = max(max_increase, new_score-old_score)
        max_block_increase = max(max_block_increase, after_b-old_score, new_score-after_b)
        step = distance(bn, on, b, o)
        b, o, old_score = bn, on, new_score
        if t == 0:
            first_support = support_equal(b,o,bt,ot,c["numerical_zero"])
        if step < c["step_tolerance"]:
            status = "step_tolerance"
            t += 1
            trajectory.append(dict(run_id=run_id, iteration=t, objective=old_score,
                frobenius_error=distance(b,o,bt,ot),
                minimum_noise_eigenvalue=np.linalg.eigvalsh(o)[0]))
            break
    final_eig = np.linalg.eigvalsh(o)
    min_eig, max_eig = min(min_eig, final_eig[0]), max(max_eig, final_eig[-1])
    all_acyclic = all_acyclic and acyclic(b)
    info = dict(run_id=run_id, cycles=t, alternating_status=status, final_step=step,
        initial_parameter_error=initial, alternating_parameter_error=distance(b,o,bt,ot),
        first_cycle_numerical_support=first_support,
        raw_support_match=support_equal(b,o,bt,ot,0),
        numerical_support_match=support_equal(b,o,bt,ot,c["numerical_zero"]),
        threshold_support_match=support_equal(b,o,bt,ot,c["delta"]),
        maximum_objective_increase=max_increase, maximum_block_increase=max_block_increase,
        minimum_noise_eigenvalue=min_eig, maximum_noise_eigenvalue=max_eig,
        all_acyclic=all_acyclic, objective=old_score,
        noise_analytic_calls=prox.statuses.get("analytic_interior", 0),
        noise_conic_calls=sum(v for k,v in prox.statuses.items() if k != "analytic_interior"),
        elapsed_seconds=time.monotonic()-started)
    return b, o, info, trajectory


def coordinates(mb, mo):
    ib = list(zip(*np.where(mb)))
    io = [(i,j) for i in range(len(mo)) for j in range(i,len(mo)) if i == j or mo[i,j]]
    def pack(b,o):
        return np.array([b[i,j] for i,j in ib]+[o[i,j] for i,j in io])
    def unpack(x):
        b, o = np.zeros_like(mb,dtype=float), np.zeros_like(mo,dtype=float)
        for v,(i,j) in zip(x,ib):
            b[i,j] = v
        for v,(i,j) in zip(x[len(ib):],io):
            o[i,j] = o[j,i] = v
        return b,o
    def gradient(gb,go):
        return np.array([gb[i,j] for i,j in ib]+[(1 if i==j else 2)*go[i,j] for i,j in io])
    return pack,unpack,gradient,ib,io


def refit(b, o, s, bt, ot, c=CONFIG):
    mb, mo = support(b, o, c["delta"])
    bz, oz = b*mb, o*mo
    np.fill_diagonal(oz, np.diag(o))
    pack,unpack,gradient,ib,io = coordinates(mb,mo)
    x0 = pack(bz,oz)
    initial_eig = float(np.linalg.eigvalsh(oz)[0])
    if initial_eig <= c["noise_floor"]:
        return dict(refit_status="threshold_start_outside_box", refit_success=False,
                    refit_parameter_error=np.nan, refit_gradient_norm=np.nan,
                    refit_covariance_error=np.nan, threshold_minimum_eigenvalue=initial_eig)
    def fun(x):
        bb,oo = unpack(x)
        if np.linalg.eigvalsh(oo)[0] <= 0:
            return 1e12, np.zeros_like(x)
        val,gb,go = loss_grad(bb,oo,s)
        return val,gradient(gb,go)
    bounds = [(-c["arrow_bound"],c["arrow_bound"]) for _ in ib]
    bounds += [(c["noise_floor"],c["noise_ceiling"]) if i==j else
               (-c["noise_ceiling"],c["noise_ceiling"]) for i,j in io]
    def spectral(x):
        eig = np.linalg.eigvalsh(unpack(x)[1])
        return np.r_[eig-c["noise_floor"], c["noise_ceiling"]-eig]
    result = minimize(fun,x0,jac=True,method="SLSQP",bounds=bounds,
        constraints=[dict(type="ineq",fun=spectral)],
        options=dict(ftol=1e-13,maxiter=1000))
    br,orr = unpack(result.x)
    _,gb,go = loss_grad(br,orr,s)
    feasible = bool(np.min(spectral(result.x)) >= -1e-8)
    return dict(refit_status=str(result.message),refit_success=bool(result.success and feasible),
        refit_parameter_error=distance(br,orr,bt,ot),
        refit_covariance_error=float(np.linalg.norm(covariance(br,orr)-covariance(bt,ot))),
        refit_gradient_norm=float(np.linalg.norm(gradient(gb,go))),
        refit_objective_improvement=float(fun(x0)[0]-result.fun),
        threshold_minimum_eigenvalue=initial_eig, refit_iterations=int(result.nit))


def checks(bt,ot,s):
    records = []
    def record(name,passed,**detail):
        records.append(dict(name=name,passed=bool(passed),**detail))
        if not passed:
            raise AssertionError(name)
    rng = np.random.default_rng(919)
    c=CONFIG
    excess = 0.0
    for z in [-2.3,-0.15,-0.01,0,0.002,0.02,0.14,0.2,2.3]:
        ans = scalar_prox(z,c["alpha"],c["lam"],c["delta"],c["arrow_bound"])
        f = lambda x: (x-z)**2/(2*c["alpha"])+c["lam"]*penalty(x,c["delta"])
        opt = minimize_scalar(f,bounds=(-2,2),method="bounded",options=dict(xatol=1e-13))
        excess = max(excess,float(f(ans)-min(f(0),opt.fun,f(-2),f(2))))
    record("scalar prox vs independent minimization",excess<1e-9,maximum_objective_excess=excess)
    errors=[]
    for p in [3,4,5]:
        for _ in range(3):
            z=rng.normal(scale=.4,size=(p,p))
            b,val,_=arrow_prox(z,c["alpha"],c["lam"],c["delta"])
            vals=scalar_prox(z,c["alpha"],c["lam"],c["delta"],2)
            gains=z*z/(2*c["alpha"])-((vals-z)**2/(2*c["alpha"])+c["lam"]*penalty(vals,c["delta"]))
            brute=max(sum(gains[o[i],o[j]] for i in range(p) for j in range(i+1,p))
                      for o in itertools.permutations(range(p)))
            errors.append(abs(val-brute))
            assert acyclic(b)
    record("DP vs enumeration of all orders",max(errors)<1e-10,cases=9,maximum_gain_error=max(errors))
    b=bt.copy(); b[0,1]+=.025
    _,gb,go=loss_grad(b,ot,s)
    h=1e-6
    db=np.zeros_like(b);db[0,1]=1
    do=np.zeros_like(ot);do[0,3]=do[3,0]=1
    eb=abs((loss_grad(b+h*db,ot,s)[0]-loss_grad(b-h*db,ot,s)[0])/(2*h)-gb[0,1])
    eo=abs((loss_grad(b,ot+h*do,s)[0]-loss_grad(b,ot-h*do,s)[0])/(2*h)-2*go[0,3])
    record("likelihood gradients including symmetric factor",max(eb,eo)<1e-7,maximum_error=max(eb,eo))
    prox=NoiseProx(4,c["beta"],c["lam"],c["delta"])
    z=ot.copy();z[0,1]=z[1,0]=c["beta"]*c["lam"]/(4*c["delta"])
    direct=prox(z);conic=prox.conic(z)
    record("interior noise prox vs independent conic solution",np.max(np.abs(direct-conic))<2e-6,
        maximum_error=float(np.max(np.abs(direct-conic))),conic_statuses=prox.statuses)
    record("symmetric noise threshold",direct[0,1]==0,threshold=c["beta"]*c["lam"]/c["delta"])
    bn,_,_=arrow_prox(b-c["alpha"]*gb,c["alpha"],c["lam"],c["delta"])
    _,_,new_go=loss_grad(bn,ot,s)
    gs=prox(ot-c["beta"]*new_go);jacobi=prox(ot-c["beta"]*go)
    record("new-B noise step differs from old-B joint update",np.linalg.norm(gs-jacobi)>1e-8,
        noise_step_difference=float(np.linalg.norm(gs-jacobi)))
    mb,mo=support(bt,ot,0)
    pack,unpack,grad,ib,io=coordinates(mb,mo)
    x=pack(bt,ot)
    jac=np.column_stack([(covariance(*unpack(x+h*np.eye(len(x))[k]))-
                          covariance(*unpack(x-h*np.eye(len(x))[k]))).ravel()/(2*h)
                         for k in range(len(x))])
    hess=np.column_stack([(grad(*loss_grad(*unpack(x+h*np.eye(len(x))[k]),s)[1:])-
                           grad(*loss_grad(*unpack(x-h*np.eye(len(x))[k]),s)[1:]))/(2*h)
                          for k in range(len(x))])
    # Convert one-coordinate-per-pair to product-Frobenius orthonormal coordinates.
    scale=np.array([1]*len(ib)+[1 if i==j else np.sqrt(2) for i,j in io])
    H=(hess+hess.T)/2/scale[:,None]/scale[None,:]
    k=len(ib);A,C,D=H[:k,:k],H[:k,k:],H[k:,k:]
    M=np.block([[np.eye(k)-c["alpha"]*A,-c["alpha"]*C],
        [-c["beta"]*C.T@(np.eye(k)-c["alpha"]*A),
         np.eye(len(x)-k)-c["beta"]*D+c["alpha"]*c["beta"]*C.T@C]])
    ev,U=np.linalg.eigh(H);sqrtH=(U*np.sqrt(ev))@U.T; invsqrtH=(U/np.sqrt(ev))@U.T
    q=float(np.linalg.svd(sqrtH@M@invsqrtH,compute_uv=False)[0])
    record("target mask numerical local regularity",np.linalg.svd(jac,compute_uv=False)[-1]>.01,
        covariance_jacobian_smallest_singular_value=float(np.linalg.svd(jac,compute_uv=False)[-1]),
        information_smallest_eigenvalue=float(ev[0]),free_parameters=len(x))
    record("coupled alternating Jacobian H-norm contraction",q<1,
        H_norm_operator_norm=q,spectral_radius=float(max(abs(np.linalg.eigvals(M)))),
        cross_block_frobenius_norm=float(np.linalg.norm(C)))
    return records


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out",default=str(OUT),help="output directory (default: results/)")
    OUT=Path(ap.parse_args().out)
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"config.json").write_text(json.dumps(CONFIG,indent=2)+"\n")
    bt,ot=target();sigma=covariance(bt,ot)
    validation=checks(bt,ot,sigma)
    b,o,pop,traj=run_alternating(sigma,CONFIG["population_seed"],"population",bt,ot)
    pop.update(refit(b,o,sigma,bt,ot))
    all_traj=traj;rows=[]
    chol=np.linalg.cholesky(sigma)
    for distribution in CONFIG["distributions"]:
        for n in CONFIG["ns"]:
            for seed in CONFIG["seeds"]:
                # Disjoint deterministic SeedSequence streams for every (law,n,seed).
                rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(distribution!="Gaussian")]))
                z=rng.normal(size=(n,4)) if distribution=="Gaussian" else rng.standard_t(7,size=(n,4))*np.sqrt(5/7)
                x=z@chol.T;x-=x.mean(axis=0)
                s=x.T@x/n
                run_id=f"{distribution.replace(' ','_')}_n{n}_s{seed}"
                b,o,info,traj=run_alternating(s,seed+17000,run_id,bt,ot)
                info.update(distribution=distribution,n=n,seed=seed,
                    covariance_error=float(np.linalg.norm(s-sigma)),
                    sample_minimum_eigenvalue=float(np.linalg.eigvalsh(s)[0]))
                info.update(refit(b,o,s,bt,ot))
                rows.append(info);all_traj.extend(traj)
                write_csv(OUT/"runs.csv",rows)
                print(json.dumps(dict(run_id=run_id,cycles=info["cycles"],
                    support=info["threshold_support_match"],refit=info["refit_success"])),flush=True)
    fiber=[]
    r=.5;delta=.1
    for t in np.linspace(0,r,501):
        bb=np.array([[0,t],[0,0]]);oo=np.array([[1,r-t],[r-t,1-2*r*t+t*t]])
        fiber.append(dict(t=t,l1=abs(t)+abs(r-t),quasi_mcp=float(penalty(t,delta)+penalty(r-t,delta)),
            count=int(t!=0)+int(r-t!=0),minimum_noise_eigenvalue=float(np.linalg.eigvalsh(oo)[0]),
            covariance_error=float(np.linalg.norm(covariance(bb,oo)-np.array([[1,r],[r,1]])))))
    write_csv(OUT/"fiber.csv",fiber);write_csv(OUT/"trajectories.csv",all_traj)
    write_csv(OUT/"population.csv",[pop])
    result=dict(config=CONFIG,target=dict(B=bt.tolist(),Omega=ot.tolist(),Sigma=sigma.tolist()),
        checks=validation,population=pop,elapsed_seconds=time.monotonic()-started,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cvxpy=cp.__version__),
        cycle="B proximal step, then Omega proximal step evaluated at the new B; fixed tuning and steps")
    (OUT/"metadata.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(checks=len(validation),runs=len(rows),elapsed_seconds=result["elapsed_seconds"])))


if __name__=="__main__":
    main()

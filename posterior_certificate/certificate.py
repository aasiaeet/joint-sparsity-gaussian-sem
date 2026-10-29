"""A posteriori certificate of local convergence in a two-variable example.

Supplementary Material, Section S18 ("A posteriori certificate") and Table S3;
Theorem 17 of the main article.  For the observed covariance
S = [[1.01, 0.49], [0.49, 1.23]], tunings alpha = beta = 1/20, lambda = 1/50,
delta = 1/10, coefficient bound 2 and spectral bounds [1/10, 4], we take a
rational point close to an exact representation and verify, on a ball of
radius 1e-6 in a quadratic metric, the support and order conditions, a
contraction bound q < 1 for the full alternating map and the invariance
inequality e0 + q r <= r.  Two candidates are checked:

  case "arrow":  one directed edge 1 -> 2 and independent errors
                 (results/directed_edge.json);
  case "noise":  no directed edge and one bidirected edge 1 <-> 2
                 (results/bidirected_edge.json).

Floating point is used only to propose the metric (a discrete Lyapunov
solution) and the nominal contraction level q0.  Every enclosure, threshold
comparison, order gap, norm bound and residual inequality is evaluated in
integer interval arithmetic with outward rounding to the grid 2^-90, and all
final comparisons are exact rational comparisons.

Coordinates: B01, B10, Omega00, Omega01 = Omega10, Omega11 (B[i, j] != 0 is
the edge i -> j, vertices indexed from 0).

Usage:  python3 certificate.py      (writes results/directed_edge.json and
                                     results/bidirected_edge.json)
"""
from fractions import Fraction as Fr
from pathlib import Path
import itertools
import json
import math
import numpy as np
import scipy.linalg
import sympy as sy

SCALE = 1 << 90


def ceildiv(a, b):
    return -((-a) // b)


class IV:
    def __init__(self, value=0, *, bounds=None):
        if bounds is not None:
            self.lo, self.hi = bounds
        elif isinstance(value, IV):
            self.lo, self.hi = value.lo, value.hi
        else:
            value = Fr(value)
            self.lo = value.numerator * SCALE // value.denominator
            self.hi = ceildiv(value.numerator * SCALE, value.denominator)
        assert self.lo <= self.hi

    def __add__(self, other):
        other = IV(other)
        return IV(bounds=(self.lo + other.lo, self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return IV(bounds=(-self.hi, -self.lo))

    def __sub__(self, other):
        return self + (-IV(other))

    def __rsub__(self, other):
        return IV(other) - self

    def __mul__(self, other):
        other = IV(other)
        corners = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return IV(bounds=(min(corners)//SCALE, ceildiv(max(corners), SCALE)))

    __rmul__ = __mul__

    def reciprocal(self):
        assert not self.lo <= 0 <= self.hi, "Interval division crosses zero"
        return IV(bounds=(SCALE*SCALE//self.hi, ceildiv(SCALE*SCALE, self.lo)))

    def __truediv__(self, other):
        return self * IV(other).reciprocal()

    def __pow__(self, power):
        assert int(power) == power
        if power < 0:
            return (self ** (-power)).reciprocal()
        ans = IV(1)
        for _ in range(int(power)):
            ans *= self
        return ans

    def absmax(self):
        return max(abs(self.lo), abs(self.hi))


def rat(value):
    return Fr(int(sy.numer(value)), int(sy.denom(value)))


def intervals(exprs, substitutions):
    memo = dict(substitutions)

    def evaluate(e):
        if e in memo:
            return memo[e]
        if e.is_Rational:
            ans = IV(rat(e))
        elif e.is_Add:
            ans = sum((evaluate(x) for x in e.args), IV(0))
        elif e.is_Mul:
            ans = IV(1)
            for x in e.args:
                ans *= evaluate(x)
        elif e.is_Pow and e.args[1].is_Integer:
            ans = evaluate(e.args[0]) ** int(e.args[1])
        else:
            raise ValueError(str(e))
        memo[e] = ans
        return ans

    return [evaluate(e) for e in exprs]


def mm(a, b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))), IV(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(row) for row in zip(*a)]


def ivmatrix(a):
    return [[IV(rat(a[i,j])) for j in range(a.cols)] for i in range(a.rows)]


def sqrt_upper(value):
    value = Fr(value)
    return Fr(math.isqrt(value.numerator*SCALE*SCALE//value.denominator)+1, SCALE)


def norm_upper(values):
    return Fr(math.isqrt(sum(v.absmax()**2 for v in values))+1, SCALE)


def ldl_positive(a):
    n = len(a)
    l = [[IV(int(i == j)) for j in range(n)] for i in range(n)]
    d = []
    for j in range(n):
        dj = a[j][j] - sum((l[j][k]*l[j][k]*d[k] for k in range(j)), IV(0))
        assert dj.lo > 0, "Verified LDL failed"
        d.append(dj)
        for i in range(j+1, n):
            l[i][j] = (a[i][j] - sum((l[i][k]*l[j][k]*d[k]
                         for k in range(j)), IV(0))) / dj
    return min(Fr(v.lo, SCALE) for v in d)


def gain(z, alpha, lam, delta, bound):
    z = abs(z)
    threshold = 2*alpha*lam/delta
    if z <= threshold:
        t = Fr(0)
    elif z < delta:
        t = (z-threshold)/(1-2*alpha*lam/delta**2)
    else:
        t = z
    t = min(t, bound)
    penalty = 2*t/delta-t*t/delta**2 if t < delta else Fr(1)
    return z*z/(2*alpha)-(t-z)**2/(2*alpha)-lam*penalty


def two_state_dp(weights, active):
    p = len(weights)
    dp = [[None, None] for _ in range(1 << p)]
    dp[0][0] = Fr(0)
    for mask in range(1, 1 << p):
        nodes = [i for i in range(p) if mask & (1 << i)]
        for j in nodes:
            prev = mask ^ (1 << j)
            violation = any((j, i) in active for i in nodes if i != j)
            score = sum((weights[i][j] for i in nodes if i != j), Fr(0))
            for oldflag in (0, 1):
                if dp[prev][oldflag] is None:
                    continue
                flag = int(oldflag or violation)
                candidate = dp[prev][oldflag] + score
                if dp[mask][flag] is None or candidate > dp[mask][flag]:
                    dp[mask][flag] = candidate
    return dp[-1][1]


def main(case="arrow"):
    assert case in ("arrow", "noise")
    alpha = beta = Fr(1,20)
    lam, delta = Fr(1,50), Fr(1,10)
    floor, ceiling, bound = Fr(1,10), Fr(4), Fr(2)
    radius = Fr(1,10**6)
    symbols = sy.symbols("b01 b10 o00 o01 o11")
    b01,b10,o00,o01,o11 = symbols
    b = sy.Matrix([[0,b01],[b10,0]])
    o = sy.Matrix([[o00,o01],[o01,o11]])
    s = sy.Matrix([[sy.Rational(101,100),sy.Rational(49,100)],
                   [sy.Rational(49,100),sy.Rational(123,100)]])
    inv = o.inv()
    gb = -s*(sy.eye(2)-b)*inv
    zb = b-sy.Rational(alpha.numerator,alpha.denominator)*gb
    u = sy.Matrix([[0,zb[0,1]],[0,0]]) if case == "arrow" else sy.zeros(2)
    res = (sy.eye(2)-u).T*s*(sy.eye(2)-u)
    go = (inv-inv*res*inv)/2
    zo = o-sy.Rational(beta.numerator,beta.denominator)*go
    f = sy.Matrix([u[0,1],0,zo[0,0],0 if case == "arrow" else zo[0,1],zo[1,1]])
    jac = f.jacobian(symbols)
    center = [Fr(49,101)+Fr(1,10**10),Fr(0),
              Fr(101,100)+Fr(2,10**11),Fr(0),
              Fr(123,100)-Fr(49,100)**2/Fr(101,100)-Fr(1,10**10)]
    if case == "noise":
        center = [Fr(0),Fr(0),Fr(101,100)+Fr(2,10**11),
                  Fr(49,100)+Fr(1,10**10),Fr(123,100)-Fr(1,10**10)]
    assert s.det()>0 and s[0,0]>0
    if case == "arrow":
        assert delta<center[0]<bound and center[1]==center[3]==0
    else:
        assert center[0]==center[1]==0 and center[3]>delta
    assert floor<center[2]-abs(center[3]) and center[2]+abs(center[3])<ceiling
    assert floor<center[4]-abs(center[3]) and center[4]+abs(center[3])<ceiling
    jnum = np.array(jac.subs(dict(zip(symbols, map(sy.Rational, center))))).astype(float)
    proposed_w = scipy.linalg.solve_discrete_lyapunov(jnum.T, np.eye(5))
    proposed_q = scipy.linalg.cholesky(proposed_w, lower=False)
    qmat = sy.Matrix([[sy.Rational(f"{v:.10f}") for v in row] for row in proposed_q])
    qinv = qmat.inv()
    radii = [radius*sqrt_upper(sum((rat(qinv[i,j])**2 for j in range(5)), Fr(0)))
             for i in range(5)]
    box = [IV(bounds=(IV(c-r).lo,IV(c+r).hi)) for c,r in zip(center,radii)]
    subs = dict(zip(symbols, box))
    packed = intervals([zb[0,1],zb[1,0],zo[0,0],zo[0,1],zo[1,1]],subs)
    arrow, reverse, noise0, noise_off, noise1 = packed
    assert alpha < delta**2/(2*lam) and beta < delta**2/(2*lam)
    if case == "arrow":
        assert Fr(arrow.lo,SCALE)>delta and Fr(arrow.hi,SCALE)<bound
    else:
        assert max(Fr(v.absmax(),SCALE) for v in (arrow,reverse))<2*alpha*lam/delta
    input_low = min(Fr(box[i].lo,SCALE) for i in (2,4))-Fr(box[3].absmax(),SCALE)
    input_high = max(Fr(box[i].hi,SCALE) for i in (2,4))+Fr(box[3].absmax(),SCALE)
    assert input_low>floor and input_high<ceiling
    if case == "arrow":
        assert Fr(noise_off.absmax(),SCALE)<beta*lam/delta
        output_off = Fr(0)
    else:
        assert Fr(noise_off.lo,SCALE)>delta
        output_off = Fr(noise_off.absmax(),SCALE)
    output_low = min(Fr(v.lo,SCALE) for v in (noise0,noise1))-output_off
    output_high = max(Fr(v.hi,SCALE) for v in (noise0,noise1))+output_off
    assert output_low>floor and output_high<ceiling
    lower_gain = Fr(arrow.lo,SCALE)**2/(2*alpha)-lam if case == "arrow" else Fr(0)
    upper_reverse_gain = gain(Fr(reverse.absmax(),SCALE),alpha,lam,delta,bound) if case == "arrow" else Fr(0)
    weights = [[Fr(0),lower_gain],[upper_reverse_gain,Fr(0)]]
    bad_score = two_state_dp(weights,{(0,1)} if case == "arrow" else set())
    gap = lower_gain-bad_score if bad_score is not None else None
    assert gap is None or gap>0
    # Cross-check the DP with exhaustive enumeration, using exact rational weights.
    bad_brute = max((sum((weights[a][b] for k,a in enumerate(order)
                        for b in order[k+1:]),Fr(0))
                    for order in itertools.permutations(range(2))
                    if case == "arrow" and order.index(0)>order.index(1)), default=None)
    assert bad_score == bad_brute
    vals = intervals(list(jac),subs)
    jiv = [vals[i*5:(i+1)*5] for i in range(5)]
    kiv = mm(mm(ivmatrix(qmat),jiv),ivmatrix(qinv))
    midpoint = sy.Matrix([[sy.Rational(f"{(v.lo+v.hi)/(2*SCALE):.12f}")
                           for v in row] for row in kiv])
    km = np.array(midpoint).astype(float)
    nominal_norm = float(np.linalg.norm(km,2))
    q0 = Fr(math.ceil(nominal_norm*10000)+1,10000)
    assert q0<1
    k0iv = ivmatrix(midpoint)
    gram = mm(transpose(k0iv),k0iv)
    pivots = ldl_positive([[IV(q0*q0 if i==j else 0)-gram[i][j]
                           for j in range(5)] for i in range(5)])
    epsilon = norm_upper([kiv[i][j]-k0iv[i][j] for i in range(5) for j in range(5)])
    qbound = q0+epsilon
    assert qbound<1
    fc = intervals(list(f),dict(zip(symbols,map(IV,center))))
    diff = [[fc[i]-IV(center[i])] for i in range(5)]
    residual = norm_upper([row[0] for row in mm(ivmatrix(qmat),diff)])
    assert residual+qbound*radius<=radius
    verified_error = residual/(1-qbound)
    # Exact check that the known representation lies in the certified ball;
    # not used by the interval certificate itself.
    known = [Fr(49,101),Fr(0),Fr(101,100),Fr(0),
             Fr(123,100)-Fr(49,100)**2/Fr(101,100)] if case == "arrow" else \
            [Fr(0),Fr(0),Fr(101,100),Fr(49,100),Fr(123,100)]
    witness_subs = dict(zip(symbols,map(sy.Rational,known)))
    assert all(sy.cancel(expr.subs(witness_subs)-sy.Rational(v))==0
               for expr,v in zip(f,known))
    exact_difference = qmat*sy.Matrix([sy.Rational(c-k) for c,k in zip(center,known)])
    exact_distance_squared = sum((rat(v)**2 for v in exact_difference), Fr(0))
    assert exact_distance_squared<=verified_error**2 and exact_distance_squared<=radius**2
    result = dict(status="verified: integer interval arithmetic with outward rounding",case=case,
        arithmetic=dict(denominator=str(SCALE),precision_bits=90,
             note="All final comparisons exact rational; floats only propose metric and q0."),
        covariance=[[str(rat(s[i,j])) for j in range(2)] for i in range(2)],
        exact_tunings={"alpha":str(alpha),"beta":str(beta),"lambda":str(lam),
             "delta":str(delta),"noise_floor":str(floor),"noise_ceiling":str(ceiling),
             "arrow_bound":str(bound)},
        center=[str(v) for v in center],metric_factor=[[str(qmat[i,j]) for j in range(5)] for i in range(5)],
        coordinate_convention=[str(v) for v in symbols],
        candidate_arrow_support=[[0,1]] if case == "arrow" else [],
        candidate_noise_pairs=[] if case == "arrow" else [[0,1]],
        radius=float(radius),coordinate_halfwidths=[float(v) for v in radii],
        residual_upper=float(residual),q0=float(q0),derivative_interval_error=float(epsilon),
        contraction_upper=float(qbound),ldl_pivot_lower=float(pivots),
        invariant_ball_slack=float(radius-residual-qbound*radius),
        fixed_point_distance_upper=float(verified_error),
        independent_exact_fixed_point_check=True,
        independent_actual_distance_upper=float(sqrt_upper(exact_distance_squared)),
        order_gap_lower=float(gap) if gap is not None else None,
        active_arrow_min=float(Fr(arrow.lo,SCALE)) if case == "arrow" else None,
        inactive_arrow_max=float(max(Fr(v.absmax(),SCALE) for v in (arrow,reverse))) if case == "noise" else None,
        arrow_threshold=float(2*alpha*lam/delta),
        active_noise_min=float(Fr(noise_off.lo,SCALE)) if case == "noise" else None,
        inactive_noise_max=float(Fr(noise_off.absmax(),SCALE)) if case == "arrow" else None,
        noise_threshold=float(beta*lam/delta),
        input_noise_spectrum_bounds=[float(input_low),float(input_high)],
        output_noise_spectrum_bounds=[float(output_low),float(output_high)],
        exact_bounds={"radius":str(radius),"residual_upper":str(residual),
             "q0":str(q0),"derivative_interval_error":str(epsilon),
             "contraction_upper":str(qbound),"ldl_pivot_lower":str(pivots),
             "invariant_ball_slack":str(radius-residual-qbound*radius),
             "fixed_point_distance_upper":str(verified_error),"order_gap_lower":str(gap) if gap is not None else None},
        notes=["Exact rational bounds are in exact_bounds; other decimal fields are rounded for display.",
               "The ambient derivative includes the reverse directed coefficient and the "
               "off-diagonal error covariance as inputs."])
    out = Path(__file__).resolve().parent / "results" / ("directed_edge.json" if case == "arrow" else "bidirected_edge.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","case","radius","residual_upper",
        "contraction_upper","invariant_ball_slack","fixed_point_distance_upper","order_gap_lower")},indent=2))


if __name__ == "__main__":
    main("arrow")
    main("noise")

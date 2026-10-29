"""Vectorized interval arithmetic with outward rounding (float64, numpy).

Rigor.  IEEE 754 binary64 +, -, *, / and sqrt are correctly rounded (round to nearest),
so the exact result of each operation on floats lies within one ulp of the computed
value; every result here is widened by one ulp outward with numpy.nextafter.  numpy
ufuncs are evaluated one operation at a time (no fused multiply-add, no extended
precision on x86-64), and BLAS is never used.  The natural logarithm is not taken from
the math library: `_log_enclose` evaluates ln(x) = e ln 2 + 2 atanh(z) with
x = m 2^e (frexp, exact), z = (m-1)/(m+1) (m-1 exact by Sterbenz), the atanh series in
interval arithmetic and a rigorous bound on its truncation error; ln 2 is enclosed by two
adjacent doubles checked against mpmath at import time.

An interval vector/matrix is an IA object holding arrays lo <= hi of the same shape.
"""
from __future__ import annotations

import numpy as np

_NINF = -np.inf
_PINF = np.inf


def dn(x):
    return np.nextafter(x, _NINF)


def up(x):
    return np.nextafter(x, _PINF)


class IA:
    __slots__ = ("lo", "hi")
    __array_priority__ = 1000

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=np.float64)
        if hi is None:
            hi = lo
        hi = np.asarray(hi, dtype=np.float64)
        if lo.shape != hi.shape:
            lo, hi = np.broadcast_arrays(lo, hi)
            lo, hi = lo.copy(), hi.copy()
        self.lo = lo
        self.hi = hi

    # ------------------------------------------------------------- basics
    @staticmethod
    def point(x):
        x = np.asarray(x, dtype=np.float64)
        return IA(x, x.copy())

    @property
    def shape(self):
        return self.lo.shape

    def __getitem__(self, k):
        return IA(self.lo[k], self.hi[k])

    def __setitem__(self, k, v):
        v = _as_ia(v)
        self.lo[k] = v.lo
        self.hi[k] = v.hi

    def copy(self):
        return IA(self.lo.copy(), self.hi.copy())

    def mid(self):
        return 0.5 * self.lo + 0.5 * self.hi

    def rad(self):
        return up(0.5 * (self.hi - self.lo))

    def width(self):
        return self.hi - self.lo

    def mag(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def contains_zero(self):
        return (self.lo <= 0) & (self.hi >= 0)

    def __repr__(self):
        return f"IA(lo={self.lo!r}, hi={self.hi!r})"

    # ------------------------------------------------------------- arithmetic
    def __neg__(self):
        return IA(-self.hi, -self.lo)

    def __add__(self, o):
        o = _as_ia(o)
        return IA(dn(self.lo + o.lo), up(self.hi + o.hi))

    __radd__ = __add__

    def __sub__(self, o):
        o = _as_ia(o)
        return IA(dn(self.lo - o.hi), up(self.hi - o.lo))

    def __rsub__(self, o):
        return _as_ia(o) - self

    def __mul__(self, o):
        o = _as_ia(o)
        a, b, c, d = self.lo, self.hi, o.lo, o.hi
        p1, p2, p3, p4 = a * c, a * d, b * c, b * d
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return IA(dn(lo), up(hi))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _as_ia(o)
        if np.any(o.contains_zero()):
            raise ZeroDivisionError("interval division by an interval containing 0")
        a, b, c, d = self.lo, self.hi, o.lo, o.hi
        p1, p2, p3, p4 = a / c, a / d, b / c, b / d
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return IA(dn(lo), up(hi))

    def __rtruediv__(self, o):
        return _as_ia(o) / self

    def sqr(self):
        a, b = self.lo, self.hi
        a2, b2 = a * a, b * b
        lo = np.where(a >= 0, a2, np.where(b <= 0, b2, 0.0))
        hi = np.maximum(a2, b2)
        lo = np.where(lo > 0, dn(lo), 0.0)
        return IA(lo, up(hi))

    def sqrt(self):
        if np.any(self.lo < 0):
            raise ValueError("sqrt of negative interval")
        return IA(np.maximum(dn(np.sqrt(self.lo)), 0.0), up(np.sqrt(self.hi)))

    def log(self):
        if np.any(self.lo <= 0):
            raise ValueError("log of nonpositive interval")
        lo, _ = _log_enclose(self.lo)
        _, hi = _log_enclose(self.hi)
        return IA(lo, hi)

    def abs(self):
        a, b = self.lo, self.hi
        lo = np.where(a >= 0, a, np.where(b <= 0, -b, 0.0))
        return IA(lo, np.maximum(-a, b))

    def hull(self, o):
        o = _as_ia(o)
        return IA(np.minimum(self.lo, o.lo), np.maximum(self.hi, o.hi))


def _as_ia(x):
    if isinstance(x, IA):
        return x
    return IA.point(x)


def stack(items, axis=-1):
    items = [_as_ia(x) for x in items]
    return IA(np.stack([x.lo for x in items], axis=axis), np.stack([x.hi for x in items], axis=axis))


def isum(items):
    out = None
    for x in items:
        out = _as_ia(x) if out is None else out + x
    return out


# ----------------------------------------------------------------- rigorous log
_LN2_LO = 0.6931471805599453   # nearest double to ln 2 (below ln 2)
_LN2_HI = float(np.nextafter(_LN2_LO, np.inf))
_SQRTH = 0.7071067811865476
_NTERMS = 14


def _check_ln2():
    import mpmath
    mpmath.mp.dps = 60
    ln2 = mpmath.log(2)
    assert mpmath.mpf(_LN2_LO) < ln2 < mpmath.mpf(_LN2_HI), "ln2 enclosure failed"


_check_ln2()


def _log_enclose(x):
    """Rigorous enclosure [lo, hi] of ln(x) for an array of positive finite doubles x."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)               # x = m 2^e exactly, m in [0.5, 1)
    small = m < _SQRTH
    m = np.where(small, m * 2.0, m)  # exact
    e = np.where(small, e - 1, e).astype(np.float64)
    # m in [sqrt(1/2), sqrt(2)) roughly; m - 1 exact by Sterbenz (m in [0.5, 2])
    num = IA.point(m - 1.0)
    den = IA.point(m) + 1.0
    z = num / den
    z2 = z.sqr()
    # 2 * sum_{k=0}^{N-1} z^{2k+1}/(2k+1), Horner in z2
    acc = IA.point(np.full(x.shape, 1.0 / (2 * _NTERMS - 1)))
    acc = IA(dn(acc.lo), up(acc.hi))
    for k in range(_NTERMS - 2, -1, -1):
        c = 1.0 / (2 * k + 1)
        acc = acc * z2 + IA(dn(np.full(x.shape, c)), up(np.full(x.shape, c)))
    s = acc * z * 2.0
    # truncation: |rem| <= 2 |z|^(2N+1) / ((2N+1)(1 - z^2)), |z| <= 0.1716
    zm = z.mag()
    zm = up(zm)
    rem = 2.0 * zm ** (2 * _NTERMS + 1) / ((2 * _NTERMS + 1) * (1.0 - 0.0295))
    rem = up(up(rem) * 1.0001)  # generous rounding margin for the power and division
    s = IA(dn(s.lo - rem), up(s.hi + rem))
    eln2 = IA.point(e) * IA(np.full(x.shape, _LN2_LO), np.full(x.shape, _LN2_HI))
    out = eln2 + s
    return out.lo, out.hi


# ----------------------------------------------------------------- 3x3 matrices
def matmul(A, B):
    """(..., n, m) @ (..., m, p) for IA arrays, summation in interval arithmetic."""
    A = _as_ia(A)
    B = _as_ia(B)
    m = A.shape[-1]
    out = None
    for k in range(m):
        term = A[..., :, k:k + 1] * B[..., k:k + 1, :]
        out = term if out is None else out + term
    return out


def transpose(A):
    return IA(np.swapaxes(A.lo, -1, -2), np.swapaxes(A.hi, -1, -2))


def det3(M):
    a = M
    t1 = a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
    t2 = a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
    t3 = a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
    return t1 - t2 + t3


def adj3(M):
    """Adjugate (transpose of the cofactor matrix) of a 3x3 interval matrix."""
    a = M
    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            # adj[i, j] = (-1)^{i+j} minor(j, i)
            r = [x for x in range(3) if x != j]
            c = [x for x in range(3) if x != i]
            mnr = a[..., r[0], c[0]] * a[..., r[1], c[1]] - a[..., r[0], c[1]] * a[..., r[1], c[0]]
            row.append(mnr if (i + j) % 2 == 0 else -mnr)
        rows.append(stack(row, axis=-1))
    return stack(rows, axis=-2)


def cholesky_pd(M):
    """Interval Cholesky of a symmetric 3x3 interval matrix.  Returns a boolean array: True
    where every pivot is certified positive, which certifies that EVERY symmetric matrix in
    the box is positive definite."""
    a = M
    d0 = a[..., 0, 0]
    ok = d0.lo > 0
    d0s = IA(np.where(ok, d0.lo, 1.0), np.where(ok, d0.hi, 1.0))
    l10 = a[..., 1, 0] / d0s.sqrt()
    l20 = a[..., 2, 0] / d0s.sqrt()
    d1 = a[..., 1, 1] - l10.sqr()
    ok &= d1.lo > 0
    d1s = IA(np.where(ok, d1.lo, 1.0), np.where(ok, d1.hi, 1.0))
    l21 = (a[..., 2, 1] - l20 * l10) / d1s.sqrt()
    d2 = a[..., 2, 2] - l20.sqr() - l21.sqr()
    ok &= d2.lo > 0
    return ok


def eye3(shape=()):
    E = np.zeros(shape + (3, 3))
    E[..., 0, 0] = E[..., 1, 1] = E[..., 2, 2] = 1.0
    return IA.point(E)

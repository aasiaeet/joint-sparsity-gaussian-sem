"""Shared definitions for the Verma graph on four vertices (main Section 7.2, Supplement S18).

Conventions follow the main article.  Parent-row
parametrization X = B^T X + eps, Sigma = Phi(B, Omega) = (I-B)^{-T} Omega (I-B)^{-1};
B[i, j] != 0 is the directed edge i -> j; Omega[i, j] != 0 (i < j) is the
bidirected edge i <-> j.  Vertices are 0..3 in code and 1..4 in names.

Contents
- enumeration of all acyclic mixed supports on four vertices (543 x 64);
- m-separation for ADMGs (paths, ancestor rule), all 24 CI statements;
- polynomial invariants evaluated over GF(p), p = 2^61 - 1, at random
  parameter points (exact arithmetic; Schwartz-Zippel): generic Jacobian rank,
  the 24 CI minors, and the 24 relabeled Verma cubics;
- the population loss and its pieces in float64.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
P = 4
PAIRS = [(i, j) for i in range(P) for j in range(i + 1, P)]          # 6
DIRECTED = [(i, j) for i in range(P) for j in range(P) if i != j]     # 12
MASTER_SEED = 20261022
STUDY_ID = 6          # second entry of every seed sequence

# Generating graph G*: 1->2->3->4 with 2<->4 (0-based below)
GSTAR_D = ((0, 1), (1, 2), (2, 3))
GSTAR_U = ((1, 3),)


# ----------------------------------------------------------------------------
# Supports
# ----------------------------------------------------------------------------

def is_acyclic(D):
    adj = {i: [j for (a, j) in D if a == i] for i in range(P)}
    state = [0] * P

    def dfs(v):
        state[v] = 1
        for w in adj[v]:
            if state[w] == 1 or (state[w] == 0 and not dfs(w)):
                return False
        state[v] = 2
        return True
    return all(state[v] == 2 or dfs(v) for v in range(P))


def edge_name(e, directed=True):
    i, j = e
    return f"{i+1}->{j+1}" if directed else f"{i+1}<->{j+1}"


@dataclass(frozen=True)
class Support:
    index: int
    D: tuple
    U: tuple

    @property
    def count(self):
        return len(self.D) + len(self.U)

    @property
    def name(self):
        parts = [edge_name(e) for e in self.D] + [edge_name(e, False) for e in self.U]
        return ",".join(parts) if parts else "empty"

    @property
    def bowed(self):
        und = {tuple(sorted(e)) for e in self.D}
        return any(u in und for u in self.U)

    def components(self):
        """Connected components of the bidirected graph (including singletons)."""
        parent = list(range(P))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j in self.U:
            parent[find(i)] = find(j)
        comps = {}
        for v in range(P):
            comps.setdefault(find(v), []).append(v)
        return [sorted(c) for c in comps.values()]

    @property
    def complete_components(self):
        Us = set(self.U)
        for c in self.components():
            for i, j in itertools.combinations(c, 2):
                if (i, j) not in Us:
                    return False
        return True


def enumerate_supports():
    Ds = []
    for m in range(1 << len(DIRECTED)):
        D = tuple(e for k, e in enumerate(DIRECTED) if m >> k & 1)
        if is_acyclic(D):
            Ds.append(D)
    Us = [tuple(pr for k, pr in enumerate(PAIRS) if m >> k & 1) for m in range(1 << len(PAIRS))]
    raw = sorted(((D, U) for D in Ds for U in Us),
                 key=lambda t: (len(t[0]) + len(t[1]), len(t[0]), t[0], t[1]))
    return [Support(k, D, U) for k, (D, U) in enumerate(raw)], len(Ds), len(Us)


SUPPORTS, N_DAG, N_BIDIR = enumerate_supports()
KEY_TO_INDEX = {(s.D, s.U): s.index for s in SUPPORTS}


def support_of(D, U):
    D = tuple(e for e in DIRECTED if e in set(D))
    U = tuple(pr for pr in PAIRS if pr in set(U))
    return SUPPORTS[KEY_TO_INDEX[(D, U)]]


GSTAR = support_of(GSTAR_D, GSTAR_U)


def immediate_subsupports(s):
    out = []
    for e in s.D:
        out.append(KEY_TO_INDEX[(tuple(x for x in s.D if x != e), s.U)])
    for e in s.U:
        out.append(KEY_TO_INDEX[(s.D, tuple(x for x in s.U if x != e))])
    return out


# ----------------------------------------------------------------------------
# m-separation
# ----------------------------------------------------------------------------

CI_STATEMENTS = []   # (i, j, S) with i < j, S subset of the other two vertices
for (i, j) in PAIRS:
    rest = [v for v in range(P) if v not in (i, j)]
    for r in range(len(rest) + 1):
        for S in itertools.combinations(rest, r):
            CI_STATEMENTS.append((i, j, tuple(S)))


def ci_name(st):
    i, j, S = st
    return f"{i+1}_||_{j+1}" + ("|" + "".join(str(v + 1) for v in S) if S else "")


def _edges_between(D, U, u, v):
    """Edges between u and v as (arrowhead_at_u, arrowhead_at_v)."""
    out = []
    if (u, v) in D:
        out.append((False, True))
    if (v, u) in D:
        out.append((True, False))
    if tuple(sorted((u, v))) in U:
        out.append((True, True))
    return out


def ancestors(D, S):
    an = set(S)
    changed = True
    while changed:
        changed = False
        for (a, b) in D:
            if b in an and a not in an:
                an.add(a)
                changed = True
    return an


def m_connected(D, U, i, j, S):
    """True if some path between i and j is m-connecting given S."""
    D, U = set(D), set(U)
    anS = ancestors(D, S)
    S = set(S)

    def extend(path_vertices, arrow_into_last):
        # arrow_into_last: arrowhead at the last vertex on the last edge
        last = path_vertices[-1]
        for w in range(P):
            if w in path_vertices:
                continue
            for (h_last, h_w) in _edges_between(D, U, last, w):
                if len(path_vertices) >= 2:
                    collider = arrow_into_last and h_last
                    if collider and last not in anS:
                        continue
                    if not collider and last in S:
                        continue
                if w == j:
                    return True
                if extend(path_vertices + [w], h_w):
                    return True
        return False
    return extend([i], False)


def msep_vector(s):
    """Tuple of booleans: statement k holds (m-separation) in support s."""
    return tuple(not m_connected(s.D, s.U, i, j, S) for (i, j, S) in CI_STATEMENTS)


# ----------------------------------------------------------------------------
# Exact algebra over GF(p)
# ----------------------------------------------------------------------------

PRIME = (1 << 61) - 1
IU = [(k, l) for k in range(P) for l in range(k, P)]   # 10 upper-triangular coordinates
VERMA_PERMS = list(itertools.permutations(range(P)))    # (a, b, c, d): a->b->c->d, b<->d


def _mm(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(P)) % PRIME for j in range(P)] for i in range(P)]


def _tr(X):
    return [[X[j][i] for j in range(P)] for i in range(P)]


def random_point_mod(s, rng):
    B = [[0] * P for _ in range(P)]
    for (i, j) in s.D:
        B[i][j] = int(rng.integers(1, PRIME))
    O = [[0] * P for _ in range(P)]
    for i in range(P):
        O[i][i] = int(rng.integers(1, PRIME))
    for (i, j) in s.U:
        O[i][j] = O[j][i] = int(rng.integers(1, PRIME))
    return B, O


def covariance_mod(B, O):
    I = [[int(i == j) for j in range(P)] for i in range(P)]
    B2 = _mm(B, B)
    B3 = _mm(B2, B)
    C = [[(I[i][j] + B[i][j] + B2[i][j] + B3[i][j]) % PRIME for j in range(P)] for i in range(P)]
    return C, _mm(_mm(_tr(C), O), C)


def jacobian_mod(s, C, Sig):
    cols = []
    for (i, j) in s.D:
        cols.append([(C[j][k] * Sig[i][l] + Sig[k][i] * C[j][l]) % PRIME for (k, l) in IU])
    for i in range(P):
        cols.append([(C[i][k] * C[i][l]) % PRIME for (k, l) in IU])
    for (i, j) in s.U:
        cols.append([(C[i][k] * C[j][l] + C[j][k] * C[i][l]) % PRIME for (k, l) in IU])
    return cols


def rank_mod(cols):
    M = [list(c) for c in cols]          # rows = columns of J; rank is the same
    rank = 0
    ncol = len(IU)
    for c in range(ncol):
        piv = next((r for r in range(rank, len(M)) if M[r][c] % PRIME), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        inv = pow(M[rank][c], PRIME - 2, PRIME)
        M[rank] = [(x * inv) % PRIME for x in M[rank]]
        for r in range(len(M)):
            if r != rank and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % PRIME for x, y in zip(M[r], M[rank])]
        rank += 1
        if rank == len(M):
            break
    return rank


def det_mod(M):
    n = len(M)
    if n == 1:
        return M[0][0] % PRIME
    if n == 2:
        return (M[0][0] * M[1][1] - M[0][1] * M[1][0]) % PRIME
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) % PRIME


def ci_minor(Sig, st):
    i, j, S = st
    rows = [i] + list(S)
    cols = [j] + list(S)
    return det_mod([[Sig[r][c] for c in cols] for r in rows])


def verma_poly(Sig, perm):
    """Verma cubic of the graph a->b->c->d, b<->d (perm = (a, b, c, d)):
    s_ac s_bb s_cd - s_ad s_bb s_cc - s_ac s_bc s_bd + s_ad s_bc^2."""
    a, b, c, d = perm
    s = Sig
    return (s[a][c] * s[b][b] * s[c][d] - s[a][d] * s[b][b] * s[c][c]
            - s[a][c] * s[b][c] * s[b][d] + s[a][d] * s[b][c] * s[b][c]) % PRIME


GSTAR_PERM = (0, 1, 2, 3)
GSTAR_CI = (0, 2, (1,))      # 1 _||_ 3 | 2


def algebraic_invariants(s, n_points=2):
    """Generic Jacobian rank and identically vanishing polynomials, over GF(p)."""
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, STUDY_ID, s.index, 3]))
    rank = 0
    ci_nz = [False] * len(CI_STATEMENTS)
    v_nz = [False] * len(VERMA_PERMS)
    aux_nz = False
    for _ in range(n_points):
        B, O = random_point_mod(s, rng)
        C, Sig = covariance_mod(B, O)
        rank = max(rank, rank_mod(jacobian_mod(s, C, Sig)))
        for k, st in enumerate(CI_STATEMENTS):
            ci_nz[k] |= ci_minor(Sig, st) != 0
        for k, pm in enumerate(VERMA_PERMS):
            v_nz[k] |= verma_poly(Sig, pm) != 0
        aux_nz |= (Sig[0][0] * Sig[1][1] * Sig[0][2]) % PRIME != 0
    ci_van = tuple(not x for x in ci_nz)
    v_van = tuple(not x for x in v_nz)
    return dict(rank=rank, ci_vanish=ci_van, verma_vanish=v_van, aux_nonvanishing=aux_nz)


def in_gstar_class(inv):
    """Test for V_H = V_{G*} (Supplement S18, "The constraint"):
    dim V_H = 8, f_CI and f_V vanish identically on M_H, and s11 s22 s13 does not."""
    k_ci = CI_STATEMENTS.index(GSTAR_CI)
    k_v = VERMA_PERMS.index(GSTAR_PERM)
    return bool(inv["rank"] == 8 and inv["ci_vanish"][k_ci] and inv["verma_vanish"][k_v]
                and inv["aux_nonvanishing"])


# ----------------------------------------------------------------------------
# Float model maps
# ----------------------------------------------------------------------------

def covariance(B, O):
    C = np.linalg.inv(np.eye(P) - B)
    return C.T @ O @ C


def loss(B, O, S):
    A = np.eye(P) - B
    R = A.T @ S @ A
    sign, ld = np.linalg.slogdet(O)
    if sign <= 0:
        return np.inf
    return float(0.5 * (ld + np.trace(np.linalg.solve(O, R))))


def saturated_value(S):
    return 0.5 * np.linalg.slogdet(S)[1] + P / 2

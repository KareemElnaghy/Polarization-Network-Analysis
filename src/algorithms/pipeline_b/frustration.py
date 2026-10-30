"""Weighted frustration of a signed graph: spectral start + greedy local search with restarts, and an exact ILP for validation.
F(s) = sum of |w_e| over edges violated by the partition s in {-1,+1}^n (positive edge across camps, negative edge inside).
Minimising F is NP-hard; the heuristic returns an upper bound on the optimum."""
import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.linalg import eigsh


def csr_from_edges(n: int, u, v, w) -> sp.csr_matrix:
    """Symmetric weighted adjacency (each undirected edge once in u,v,w)."""
    u, v, w = np.asarray(u), np.asarray(v), np.asarray(w, dtype=float)
    return sp.csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n))


@njit(cache=True)
def _objective(indptr, indices, data, s):
    """Sum over undirected edges of w*s_i*s_j (maximised by a good partition)."""
    t = 0.0
    for i in range(len(indptr) - 1):
        for k in range(indptr[i], indptr[i + 1]):
            t += data[k] * s[i] * s[indices[k]]
    return t / 2.0


@njit(cache=True)
def _local_search(indptr, indices, data, s, order):
    """Greedy single-node flips until no flip increases sum w*s*s (strictly monotone, so it terminates)."""
    n = len(s)
    h = np.zeros(n)
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            h[i] += data[k] * s[indices[k]]
    improved = True
    while improved:
        improved = False
        for i in order:
            if s[i] * h[i] < 0:
                a = s[i]
                for k in range(indptr[i], indptr[i + 1]):
                    h[indices[k]] -= 2.0 * a * data[k]
                s[i] = -a
                improved = True
    return s


@njit(cache=True)
def _anneal(indptr, indices, data, s, sweeps, t0, t1, seed):
    """Simulated annealing on single-node flips (geometric cooling t0 -> t1), then the caller polishes greedily."""
    np.random.seed(seed)
    n = len(s)
    h = np.zeros(n)
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            h[i] += data[k] * s[indices[k]]
    for sw in range(sweeps):
        T = t0 * (t1 / t0) ** (sw / max(sweeps - 1, 1))
        for _ in range(n):
            i = np.random.randint(n)
            d = -2.0 * s[i] * h[i]  # change in sum w*s*s if i flips
            if d > 0 or np.random.random() < np.exp(d / T):
                a = s[i]
                for k in range(indptr[i], indptr[i + 1]):
                    h[indices[k]] -= 2.0 * a * data[k]
                s[i] = -a
    return s


def frustration(W: sp.csr_matrix, restarts: int = 10, seed: int = 0, sa_sweeps: int | None = None) -> tuple[float, np.ndarray]:
    """Best (lowest) frustration found over one spectral start and `restarts - 1` random starts. Returns (F, s)."""
    n = W.shape[0]
    A = float(abs(W).sum()) / 2.0
    if n == 0 or W.nnz == 0:
        return 0.0, np.ones(n)
    rng = np.random.default_rng(seed)
    try:
        v = np.linalg.eigh(W.toarray())[1][:, -1] if n <= 400 else eigsh(W.astype(float), k=1, which="LA")[1][:, 0]
        starts = [np.where(v >= 0, 1.0, -1.0)]
    except Exception:  # eigensolver failure: random starts only
        starts = []
    starts += [rng.choice([-1.0, 1.0], n) for _ in range(max(restarts - len(starts), 0))]
    best_obj, best_s = -np.inf, None
    ip, ix, dt = W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float64)
    scale = float(np.abs(dt).mean())
    if sa_sweeps is None:  # fixed work budget of ~2e6 node-visits per restart: many sweeps on small graphs, 100 on big ones
        sa_sweeps = int(min(5000, max(100, 2_000_000 // n)))
    for r, s0 in enumerate(starts):
        s = s0.copy()
        if sa_sweeps and r > 0:  # the spectral start (r == 0) is only polished; SA starts from random states
            s = _anneal(ip, ix, dt, s, sa_sweeps, 1.5 * scale, 0.05 * scale, seed * 1000 + r)
        s = _local_search(ip, ix, dt, s, rng.permutation(n))
        obj = _objective(ip, ix, dt, s)
        if obj > best_obj:
            best_obj, best_s = obj, s
    return 0.5 * (A - best_obj), best_s


def exact_frustration(n: int, u, v, w, time_limit: float = 60.0, detail: bool = False):
    """Exact minimum by ILP (scipy HiGHS). Fine up to a few hundred nodes; used only to validate the heuristic.
    x_i in {0,1} is the camp; y_e >= 0 is 1 when edge e is violated. Rows are (coef y, coef x_u, coef x_v) >= lower bound."""
    u, v, w = np.asarray(u), np.asarray(v), np.asarray(w, dtype=float)
    m = len(w)
    pos = [((1, -1, 1), 0.0), ((1, 1, -1), 0.0)]      # y >= x_u - x_v ; y >= x_v - x_u
    neg = [((1, -1, -1), -1.0), ((1, 1, 1), 1.0)]     # y >= x_u + x_v - 1 ; y >= 1 - x_u - x_v
    rows, cols, vals, lb, r = [], [], [], [], 0
    for e in range(m):
        for (cy, cu, cv), low in (pos if w[e] > 0 else neg):
            rows += [r, r, r]; cols += [n + e, u[e], v[e]]; vals += [cy, cu, cv]; lb.append(low); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * m, n + m))
    res = milp(np.r_[np.zeros(n), np.abs(w)], constraints=LinearConstraint(A, lb, np.inf),
               integrality=np.r_[np.ones(n), np.zeros(m)], bounds=Bounds(np.zeros(n + m), np.ones(n + m)),
               options={"time_limit": time_limit})
    if detail:  # optimal: solver proved it; bound: best proven lower bound on F
        return {"F": float(res.fun), "optimal": res.status == 0, "bound": float(getattr(res, "mip_dual_bound", np.nan))}
    return float(res.fun)

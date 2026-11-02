import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

_TAU_LSE = 2e-4

def _metrics(X, iu0=None):
    n = X.shape[0]
    if iu0 is None:
        iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0]
    m = float(np.max(c))
    # top 500 / 100
    if len(c) >= 500:
        part = np.argpartition(-c, 500 - 1)[:500]
        top500 = c[part]
        # sort not needed for mean/lse but cheap
        # mean top100
        if len(top500) >= 100:
            p2 = np.argpartition(-top500, 100 - 1)[:100]
            mean100 = float(np.mean(top500[p2]))
        else:
            mean100 = float(np.mean(top500))
        # log-sum-exp of top500 with tau
        tau = _TAU_LSE
        # (top500 - m)/tau <= 0 stable
        e = np.exp((top500 - m) / tau)
        lse = float(m + tau * np.log(np.mean(e) + 1e-300))
    else:
        mean100 = float(np.mean(c))
        lse = m
    return m, mean100, lse, c

def _lex_less(a, b, eps=1e-12):
    # a,b are (max,mean,lse) tuples; True if a < b lexicographically
    if a[0] < b[0] - eps:
        return True
    if abs(a[0] - b[0]) <= eps:
        if a[1] < b[1] - eps:
            return True
        if abs(a[1] - b[1]) <= eps:
            if a[2] < b[2] - eps:
                return True
    return False

def _lp_min_t(X, I, J, c, radius, time_limit):
    n, d = X.shape
    p = len(I)
    nd = n * d
    nv = nd + 1
    # A_ub: Xj*dx_i + Xi*dx_j - t <= -c
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cd = np.arange(d)
    cols_i = (I[:, None] * d + cd).ravel()
    cols_j = (J[:, None] * d + cd).ravel()
    cols_t = np.full(p, nd)
    cols = np.concatenate([cols_i, cols_j, cols_t])
    # need interleaving per row: rows expects [row0 * (2d+1), row1*...]
    # Build vals in same per-row order: for each row r: X[J[r]], X[I[r]], -1
    vals_i = X[J].ravel()
    vals_j = X[I].ravel()
    # interleave: create (p, 2d+1) array then ravel
    M = np.empty((p, 2 * d + 1))
    M[:, :d] = X[J]
    M[:, d:2 * d] = X[I]
    M[:, 2 * d] = -1.0
    vals = M.ravel()
    # columns in matching order: for each row, cols_i block then cols_j then t
    C = np.empty((p, 2 * d + 1), dtype=int)
    C[:, :d] = I[:, None] * d + cd
    C[:, d:2 * d] = J[:, None] * d + cd
    C[:, 2 * d] = nd
    cols = C.ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    # A_eq tangent
    rows_eq = np.repeat(np.arange(n), d)
    cols_eq = np.arange(nd)
    vals_eq = X.ravel()
    A_eq = coo_matrix((vals_eq, (rows_eq, cols_eq)), shape=(n, nv)).tocsr()
    obj = np.zeros(nv)
    obj[-1] = 1.0
    b_ub = -c
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                  bounds=[(-radius, radius)] * nd + [(None, None)],
                  method="highs", options={"time_limit": float(time_limit)})
    return res

def _lp_min_sum(X, I, J, c, g, cur_max, radius, time_limit):
    n, d = X.shape
    p = len(I)
    nd = n * d
    rows = np.repeat(np.arange(p), 2 * d)
    C = np.empty((p, 2 * d), dtype=int)
    cd = np.arange(d)
    C[:, :d] = I[:, None] * d + cd
    C[:, d:] = J[:, None] * d + cd
    cols = C.ravel()
    M = np.empty((p, 2 * d))
    M[:, :d] = X[J]
    M[:, d:] = X[I]
    vals = M.ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nd)).tocsr()
    rows_eq = np.repeat(np.arange(n), d)
    cols_eq = np.arange(nd)
    vals_eq = X.ravel()
    A_eq = coo_matrix((vals_eq, (rows_eq, cols_eq)), shape=(n, nd)).tocsr()
    b_ub = cur_max - c  # >=0 feasible at 0
    res = linprog(g, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                  bounds=[(-radius, radius)] * nd,
                  method="highs", options={"time_limit": float(time_limit)})
    return res

def solve():
    t0 = time.time()
    HARD_END = t0 + 132.0  # stop optimisation by ~132s, return before 180s
    n, d = 82, 14
    nd = n * d
    # load best start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        Xcur = _norm(rng0.standard_normal((n, d)))
    else:
        cands.sort(key=lambda c: _max_cos(c))
        Xcur = cands[0].copy()
    Xcur = _norm(Xcur)
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    m0, mean0, lse0, _ = _metrics(Xcur, iu0)
    cur_tup = (m0, mean0, lse0)
    Xbest = Xcur.copy()
    best_tup = cur_tup
    best_max = m0

    radius = 5e-4
    radius_min = 1e-7
    radius_max = 1e-3
    it = 0
    tau_w = 2e-4

    while True:
        it += 1
        now = time.time()
        if now >= HARD_END - 2.5:
            break
        if radius < 1e-14:
            radius = 5e-4
        # evaluate current (recompute cheap G)
        G = Xcur @ Xcur.T
        c_all = G[I0, J0]
        m_cur = float(np.max(c_all))
        # metrics (avoid full sort: partition for top500)
        idx500 = np.argpartition(-c_all, 500 - 1)[:500]
        top500c = c_all[idx500]
        idx100 = np.argpartition(-top500c, 100 - 1)[:100]
        mean100 = float(np.mean(top500c[idx100]))
        e = np.exp((top500c - m_cur) / _TAU_LSE)
        lse500 = float(m_cur + _TAU_LSE * np.log(float(np.mean(e)) + 1e-300))
        cur_tup = (m_cur, mean100, lse500)

        # track best
        if m_cur < best_max - 1e-15:
            best_max = m_cur
            Xbest = Xcur.copy()
            best_tup = cur_tup

        # active set: wide 1200-2000 or within 2e-3
        slack = m_cur - c_all
        cnt_close = int(np.count_nonzero(slack < 2e-3))
        if cnt_close >= 1200 and cnt_close <= 2000:
            sel = np.where(slack < 2e-3)[0]
        elif cnt_close > 2000:
            sel = np.argpartition(-c_all, 2000 - 1)[:2000]
        else:
            # need 1500
            K = 1500
            if K > len(c_all):
                K = len(c_all)
            sel = np.argpartition(-c_all, K - 1)[:K]
        I = I0[sel]; J = J0[sel]; c_sel = c_all[sel]
        p = len(sel)

        improved = False
        # ---- LP1: minimise t ----
        tl = min(1.0, max(0.2, HARD_END - 2.5 - time.time()))
        if tl >= 0.2 and p > 0:
            try:
                res = _lp_min_t(Xcur, I, J, c_sel, float(radius), tl)
            except Exception:
                res = None
            if res is not None and getattr(res, "status", 1) == 0 and res.x is not None:
                dx = np.asarray(res.x[:-1]).reshape(n, d)
                if np.all(np.isfinite(dx)):
                    Y = _norm(Xcur + dx)
                    if np.all(np.isfinite(Y)):
                        mY, meanY, lseY, _ = _metrics(Y, iu0)
                        tupY = (mY, meanY, lseY)
                        if _lex_less(tupY, cur_tup):
                            Xcur = Y
                            cur_tup = tupY
                            m_cur = mY
                            if mY < best_max - 1e-15:
                                best_max = mY
                                Xbest = Y.copy()
                                best_tup = tupY
                            radius = min(radius * 1.3, radius_max)
                            improved = True
                            # update slack-dependent vars for second LP? keep going to also try sum LP from new point next iter
                            # still attempt sum LP in same iteration from new point? skip to next iter for time
                            continue
        if time.time() >= HARD_END - 2.5:
            break
        # ---- LP2: minimise weighted sum of top-500 subject to <= cur max ----
        # build weights for top500 (softmax)
        # map global pair index -> weight; only top500 get nonzero
        # use same active set for constraints, objective from top500 subset
        try:
            # weights
            w500 = np.exp((top500c - m_cur) / tau_w)
            w500 = w500 / (np.mean(w500) + 1e-300)  # normalize to O(1)
            # build g
            g = np.zeros(nd)
            # need I/J for idx500 pairs
            I500 = I0[idx500]; J500 = J0[idx500]
            # accumulate: vectorized via loop (500 iters cheap)
            Xc = Xcur
            for k in range(len(idx500)):
                w = float(w500[k])
                i = int(I500[k]); j = int(J500[k])
                g[i * d:(i + 1) * d] += w * Xc[j]
                g[j * d:(j + 1) * d] += w * Xc[i]
            ng = float(np.linalg.norm(g))
            if np.isfinite(ng) and ng > 1e-18:
                g = g / ng  # normalize scale for solver stability
                # scale RHS uses current m_cur
                tl2 = min(1.0, max(0.2, HARD_END - 2.5 - time.time()))
                if tl2 >= 0.2:
                    res2 = _lp_min_sum(Xcur, I, J, c_sel, g, m_cur, float(radius), tl2)
                    if res2 is not None and getattr(res2, "status", 1) == 0 and res2.x is not None:
                        dx2 = np.asarray(res2.x).reshape(n, d)
                        if np.all(np.isfinite(dx2)):
                            Y2 = _norm(Xcur + dx2)
                            if np.all(np.isfinite(Y2)):
                                mY2, meanY2, lseY2, _ = _metrics(Y2, iu0)
                                tupY2 = (mY2, meanY2, lseY2)
                                if _lex_less(tupY2, cur_tup):
                                    Xcur = Y2
                                    cur_tup = tupY2
                                    if mY2 < best_max - 1e-15:
                                        best_max = mY2
                                        Xbest = Y2.copy()
                                        best_tup = tupY2
                                    radius = min(radius * 1.3, radius_max)
                                    improved = True
        except Exception:
            pass

        if not improved:
            radius *= 0.5
            if radius < radius_min:
                # reset to small-mid to keep moving shell without topology shock
                radius = 5e-4
                # optionally revert to best to avoid drift
                # keep current (lexicographically not worse?) but anchor occasionally
                if it % 20 == 0:
                    Xcur = Xbest.copy()
        # safety: cap iterations
        if it > 300:
            break
    Xbest = _norm(Xbest)
    return Xbest

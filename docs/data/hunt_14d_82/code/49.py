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

_TAU = 2e-4

def _metrics(X, iu):
    G = X @ X.T
    c = G[iu]
    m = float(np.max(c))
    idx500 = np.argpartition(-c, 500 - 1)[:500]
    t500 = c[idx500]
    idx100 = np.argpartition(-t500, 100 - 1)[:100]
    mean100 = float(np.mean(t500[idx100]))
    e = np.exp((t500 - m) / _TAU)
    lse = float(m + _TAU * np.log(float(np.mean(e)) + 1e-300))
    return m, mean100, lse

def _lex_less(a, b, eps=1e-12):
    if a[0] < b[0] - eps:
        return True
    if abs(a[0] - b[0]) <= eps:
        if a[1] < b[1] - eps:
            return True
        if abs(a[1] - b[1]) <= eps:
            if a[2] < b[2] - eps:
                return True
    return False

def _lp_t(X, I, J, c, radius, tl):
    n, d = X.shape
    p = len(I)
    nd = n * d
    nv = nd + 1
    cd = np.arange(d)
    C = np.empty((p, 2 * d + 1), dtype=int)
    C[:, :d] = I[:, None] * d + cd
    C[:, d:2 * d] = J[:, None] * d + cd
    C[:, 2 * d] = nd
    M = np.empty((p, 2 * d + 1))
    M[:, :d] = X[J]
    M[:, d:2 * d] = X[I]
    M[:, 2 * d] = -1.0
    rows = np.repeat(np.arange(p), 2 * d + 1)
    A_ub = coo_matrix((M.ravel(), (rows, C.ravel())), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                  bounds=[(-radius, radius)] * nd + [(None, None)],
                  method="highs", options={"time_limit": float(tl)})
    return res

def _lp_sum(X, I, J, c, g, cur_max, radius, tl):
    n, d = X.shape
    p = len(I)
    nd = n * d
    cd = np.arange(d)
    C = np.empty((p, 2 * d), dtype=int)
    C[:, :d] = I[:, None] * d + cd
    C[:, d:] = J[:, None] * d + cd
    M = np.empty((p, 2 * d))
    M[:, :d] = X[J]
    M[:, d:] = X[I]
    rows = np.repeat(np.arange(p), 2 * d)
    A_ub = coo_matrix((M.ravel(), (rows, C.ravel())), shape=(p, nd)).tocsr()
    A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nd)).tocsr()
    b_ub = cur_max - c
    res = linprog(g, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                  bounds=[(-radius, radius)] * nd,
                  method="highs", options={"time_limit": float(tl)})
    return res

def solve():
    t0 = time.time()
    HARD = t0 + 130.0
    n, d = 82, 14
    nd = n * d
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
    iu = np.triu_indices(n, 1)
    I0 = iu[0]
    J0 = iu[1]
    try:
        m0, me0, ls0 = _metrics(Xcur, iu)
    except Exception:
        m0 = _max_cos(Xcur); me0 = m0; ls0 = m0
    cur_tup = (m0, me0, ls0)
    Xbest = Xcur.copy()
    best_tup = cur_tup
    best_max = m0
    radius = 5e-4
    rmin = 1e-7
    rmax = 1e-3
    it = 0
    try:
        while it < 250:
            it += 1
            if time.time() >= HARD - 3.0:
                break
            G = Xcur @ Xcur.T
            c_all = G[I0, J0]
            m_cur = float(np.max(c_all))
            idx500 = np.argpartition(-c_all, 500 - 1)[:500]
            top500c = c_all[idx500]
            idx100 = np.argpartition(-top500c, 100 - 1)[:100]
            mean100 = float(np.mean(top500c[idx100]))
            e = np.exp((top500c - m_cur) / _TAU)
            lse500 = float(m_cur + _TAU * np.log(float(np.mean(e)) + 1e-300))
            cur_tup = (m_cur, mean100, lse500)
            if m_cur < best_max - 1e-15:
                best_max = m_cur
                Xbest = Xcur.copy()
                best_tup = cur_tup
            slack = m_cur - c_all
            cnt_close = int(np.count_nonzero(slack < 2e-3))
            if 1200 <= cnt_close <= 2000:
                sel = np.where(slack < 2e-3)[0]
            elif cnt_close > 2000:
                sel = np.argpartition(-c_all, 2000 - 1)[:2000]
            else:
                K = 1500
                sel = np.argpartition(-c_all, K - 1)[:K]
            I = I0[sel]; J = J0[sel]; c_sel = c_all[sel]
            improved = False
            # LP1: min t
            tl = min(1.0, HARD - 3.0 - time.time())
            if tl >= 0.15:
                try:
                    res = _lp_t(Xcur, I, J, c_sel, float(radius), tl)
                except Exception:
                    res = None
                if res is not None and getattr(res, "status", 1) == 0 and res.x is not None:
                    try:
                        dx = np.asarray(res.x[:-1]).reshape(n, d)
                        if np.all(np.isfinite(dx)):
                            Y = _norm(Xcur + dx)
                            mY, meY, lsY = _metrics(Y, iu)
                            tupY = (mY, meY, lsY)
                            if _lex_less(tupY, cur_tup):
                                Xcur = Y
                                cur_tup = tupY
                                if mY < best_max - 1e-15:
                                    best_max = mY
                                    Xbest = Y.copy()
                                    best_tup = tupY
                                radius = min(radius * 1.3, rmax)
                                improved = True
                                continue
                    except Exception:
                        pass
            if time.time() >= HARD - 3.0:
                break
            # LP2: min weighted sum top-500 with no-exceed-best
            try:
                w500 = np.exp((top500c - m_cur) / _TAU)
                w500 = w500 / (float(np.mean(w500)) + 1e-300)
                I500 = I0[idx500]; J500 = J0[idx500]
                g = np.zeros(nd)
                for k in range(len(idx500)):
                    w = float(w500[k])
                    ii = int(I500[k]); jj = int(J500[k])
                    g[ii * d:(ii + 1) * d] += w * Xcur[jj]
                    g[jj * d:(jj + 1) * d] += w * Xcur[ii]
                ng = float(np.linalg.norm(g))
                if np.isfinite(ng) and ng > 1e-18:
                    g = g / ng
                    tl2 = min(1.0, HARD - 3.0 - time.time())
                    if tl2 >= 0.15:
                        res2 = _lp_sum(Xcur, I, J, c_sel, g, m_cur, float(radius), tl2)
                        if res2 is not None and getattr(res2, "status", 1) == 0 and res2.x is not None:
                            dx2 = np.asarray(res2.x).reshape(n, d)
                            if np.all(np.isfinite(dx2)):
                                Y2 = _norm(Xcur + dx2)
                                mY2, meY2, lsY2 = _metrics(Y2, iu)
                                tupY2 = (mY2, meY2, lsY2)
                                if _lex_less(tupY2, cur_tup):
                                    Xcur = Y2
                                    cur_tup = tupY2
                                    if mY2 < best_max - 1e-15:
                                        best_max = mY2
                                        Xbest = Y2.copy()
                                        best_tup = tupY2
                                    radius = min(radius * 1.3, rmax)
                                    improved = True
            except Exception:
                pass
            if not improved:
                radius *= 0.5
                if radius < rmin:
                    radius = 5e-4
                    if it % 20 == 0:
                        Xcur = Xbest.copy()
    except Exception:
        pass
    try:
        Xbest = _norm(Xbest)
    except Exception:
        pass
    return Xbest

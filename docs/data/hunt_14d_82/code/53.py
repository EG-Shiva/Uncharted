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
    # top500
    k500 = 500 if len(c) >= 500 else len(c)
    part = np.argpartition(-c, k500 - 1)[:k500]
    top500 = c[part]
    k100 = 100 if k500 >= 100 else k500
    p2 = np.argpartition(-top500, k100 - 1)[:k100]
    mean100 = float(np.mean(top500[p2]))
    e = np.exp((top500 - m) / _TAU)
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

def solve():
    t0 = time.time()
    OPT_END = t0 + 128.0
    n, d = 82, 14
    nd = n * d
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        Xcur = _norm(rng0.standard_normal((n, d)))
    else:
        cands.sort(key=_max_cos)
        Xcur = cands[0].copy()
    Xcur = _norm(Xcur)
    iu = np.triu_indices(n, 1)
    I0 = iu[0]; J0 = iu[1]
    c_all_tmp = (Xcur @ Xcur.T)[I0, J0]
    m0 = float(np.max(c_all_tmp))
    _, mean0, lse0 = _metrics(Xcur, iu)
    cur_tup = (m0, mean0, lse0)
    Xbest = Xcur.copy()
    best_tup = cur_tup
    best_max = m0

    radius = 5e-4
    radius_min = 2e-7
    radius_max = 1e-3
    cd = np.arange(d)
    it = 0
    tau_w = 2e-4

    while True:
        it += 1
        if time.time() >= OPT_END - 1.5:
            break
        if it > 400:
            break
        # current metrics
        G = Xcur @ Xcur.T
        c_all = G[I0, J0]
        m_cur = float(np.max(c_all))
        # fast top500 stats
        idx500 = np.argpartition(-c_all, 499)[:500]
        top500c = c_all[idx500]
        idx100 = np.argpartition(-top500c, 99)[:100]
        mean100 = float(np.mean(top500c[idx100]))
        e = np.exp((top500c - m_cur) / _TAU)
        lse500 = float(m_cur + _TAU * np.log(float(np.mean(e)) + 1e-300))
        cur_tup = (m_cur, mean100, lse500)
        if m_cur < best_max - 1e-15:
            best_max = m_cur
            Xbest = Xcur.copy()
            best_tup = cur_tup

        # wide active set 1200-2000 or within 2e-3
        slack = m_cur - c_all
        cnt_close = int(np.count_nonzero(slack < 2e-3))
        if 1200 <= cnt_close <= 2000:
            sel = np.where(slack < 2e-3)[0]
        elif cnt_close > 2000:
            sel = np.argpartition(-c_all, 1999)[:2000]
        else:
            K = 1500
            sel = np.argpartition(-c_all, K - 1)[:K]
        Ia = I0[sel]; Ja = J0[sel]; ca = c_all[sel]
        p = len(sel)

        improved = False
        # --- LP1 minimax t ---
        remain = OPT_END - 1.5 - time.time()
        if remain >= 0.25 and p > 0:
            tl = min(0.9, max(0.25, remain))
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                C = np.empty((p, 2 * d + 1), dtype=int)
                C[:, :d] = Ia[:, None] * d + cd
                C[:, d:2 * d] = Ja[:, None] * d + cd
                C[:, 2 * d] = nd
                M = np.empty((p, 2 * d + 1))
                M[:, :d] = Xcur[Ja]
                M[:, d:2 * d] = Xcur[Ia]
                M[:, 2 * d] = -1.0
                A_ub = coo_matrix((M.ravel(), (rows, C.ravel())), shape=(p, nd + 1)).tocsr()
                A_eq = coo_matrix((Xcur.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nd + 1)).tocsr()
                obj = np.zeros(nd + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-ca, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-float(radius), float(radius))] * nd + [(None, None)],
                              method="highs", options={"time_limit": float(tl)})
                if getattr(res, "status", 1) == 0 and res.x is not None:
                    dx = np.asarray(res.x[:-1], dtype=float).reshape(n, d)
                    if np.all(np.isfinite(dx)):
                        Y = _norm(Xcur + dx)
                        mY, meanY, lseY = _metrics(Y, iu)
                        tupY = (mY, meanY, lseY)
                        if _lex_less(tupY, cur_tup):
                            Xcur = Y
                            cur_tup = tupY
                            if mY < best_max - 1e-15:
                                best_max = mY
                                Xbest = Y.copy()
                                best_tup = tupY
                            radius = min(radius * 1.3, radius_max)
                            improved = True
                            continue
            except Exception:
                pass
        if time.time() >= OPT_END - 1.5:
            break
        # --- LP2 weighted tail sum with cap ---
        try:
            remain2 = OPT_END - 1.5 - time.time()
            if remain2 >= 0.25:
                w500 = np.exp((top500c - m_cur) / tau_w)
                w500 = w500 / (float(np.mean(w500)) + 1e-300)
                g = np.zeros(nd)
                I500 = I0[idx500]; J500 = J0[idx500]
                Xc = Xcur
                for k in range(500):
                    w = float(w500[k])
                    ii = int(I500[k]); jj = int(J500[k])
                    g[ii * d:(ii + 1) * d] += w * Xc[jj]
                    g[jj * d:(jj + 1) * d] += w * Xc[ii]
                ng = float(np.linalg.norm(g))
                if np.isfinite(ng) and ng > 1e-18:
                    g = g / ng
                    tl2 = min(0.9, max(0.25, remain2))
                    C2 = np.empty((p, 2 * d), dtype=int)
                    C2[:, :d] = Ia[:, None] * d + cd
                    C2[:, d:] = Ja[:, None] * d + cd
                    M2 = np.empty((p, 2 * d))
                    M2[:, :d] = Xcur[Ja]
                    M2[:, d:] = Xcur[Ia]
                    rows2 = np.repeat(np.arange(p), 2 * d)
                    A_ub2 = coo_matrix((M2.ravel(), (rows2, C2.ravel())), shape=(p, nd)).tocsr()
                    A_eq2 = coo_matrix((Xcur.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nd)).tocsr()
                    b_ub2 = m_cur - ca
                    res2 = linprog(g, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=np.zeros(n),
                                   bounds=[(-float(radius), float(radius))] * nd,
                                   method="highs", options={"time_limit": float(tl2)})
                    if getattr(res2, "status", 1) == 0 and res2.x is not None:
                        dx2 = np.asarray(res2.x, dtype=float).reshape(n, d)
                        if np.all(np.isfinite(dx2)):
                            Y2 = _norm(Xcur + dx2)
                            mY2, meanY2, lseY2 = _metrics(Y2, iu)
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
                radius = 5e-4
                if it % 20 == 0:
                    Xcur = Xbest.copy()
    Xbest = _norm(np.asarray(Xbest, dtype=float))
    # safety finite
    Xbest[~np.isfinite(Xbest)] = 0.0
    Xbest = _norm(Xbest)
    return Xbest

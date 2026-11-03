import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _viol_sum(x, t):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    v = g - t
    v[v < 0] = 0.0
    # sum over upper triangle only
    iu = np.triu_indices(x.shape[0], 1)
    return float(v[iu].sum()), float(g.max())

def _feas_lp_step(x, t, r, K):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    iu0 = np.triu_indices(n, 1)
    c_all = G[iu0]
    # most violated K pairs (largest cosine)
    P = c_all.size
    K = min(K, P)
    if K < P:
        part = np.argpartition(-c_all, K - 1)[:K]
        # sort for determinism
        o = np.argsort(-c_all[part])
        sel_idx = part[o]
    else:
        sel_idx = np.argsort(-c_all)
    I = iu0[0][sel_idx]
    J = iu0[1][sel_idx]
    c_sel = c_all[sel_idx]
    p = K
    nv = n * d + p
    # A_ub: <u_i,x_j>+<x_i,u_j> - s_k <= t - c
    rows = np.repeat(np.arange(p), 2 * d + 1)
    ucols_I = (I[:, None] * d + np.arange(d)).ravel()
    ucols_J = (J[:, None] * d + np.arange(d)).ravel()
    scols = np.repeat(np.arange(n * d, n * d + p), 1)
    # interleave per row: need cols in same order as rows repeat
    cols = np.empty(p * (2 * d + 1), dtype=np.int64)
    vals = np.empty(p * (2 * d + 1), dtype=float)
    # x[J] and x[I]
    XJ = x[J]  # p,d
    XI = x[I]
    for k in range(p):
        base = k * (2 * d + 1)
        cols[base:base + d] = I[k] * d + np.arange(d)
        vals[base:base + d] = XJ[k]
        cols[base + d:base + 2 * d] = J[k] * d + np.arange(d)
        vals[base + d:base + 2 * d] = XI[k]
        cols[base + 2 * d] = n * d + k
        vals[base + 2 * d] = -1.0
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    b_ub = t - c_sel
    # A_eq: <x_i,u_i>=0
    erows = np.repeat(np.arange(n), d)
    ecols = np.arange(n * d)
    edata = x.ravel()
    A_eq = coo_matrix((edata, (erows, ecols)), shape=(n, nv)).tocsr()
    b_eq = np.zeros(n)
    obj = np.zeros(nv)
    obj[n * d:] = 1.0
    bounds = [(-r, r)] * (n * d) + [(0.0, None)] * p
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return res, I, J, c_sel

def _lp_polish(x, best, t_end):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = float(best)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap constraints for speed
        if p > 1200:
            # keep most tight
            cs = c[sel]
            o = np.argsort(-cs)[:1200]
            I, J = I[o], J[o]
            cs = cs[o]
            c_sel = cs
            p = 1200
        else:
            c_sel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    # load incumbent
    xb = None
    best_val = 1e9
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        rng = np.random.default_rng(0)
        x_best = rng.standard_normal((102, 12))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = _max_cos(x_best)
    incumbent = float(best_val)
    n, d = 102, 12

    deltas = [1e-6, 3e-6, 1e-5, 3e-5]
    K = 350
    found = False
    # target-feasibility search
    for di, delta in enumerate(deltas):
        if time.time() > T_END - 14.0:
            break
        t = best_val - delta
        # start from current best
        x = x_best.copy()
        S_old, m_old = _viol_sum(x, t)
        if m_old < best_val - 1e-12:
            x_best = x.copy()
            best_val = float(m_old)
            found = True
            break
        r = 0.004
        # adapt K slightly: 200-500
        Kcur = 300 + 50 * (di % 2)
        if Kcur < 200: Kcur = 200
        if Kcur > 500: Kcur = 500
        for it in range(28):
            if time.time() > T_END - 14.0:
                break
            try:
                res, I, J, c_sel = _feas_lp_step(x, t, r, Kcur)
            except Exception:
                r *= 0.5
                if r < 1e-9:
                    break
                continue
            if res.status != 0:
                r *= 0.5
                if r < 1e-9:
                    break
                continue
            u = res.x[:n * d].reshape(n, d)
            y = x + u
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y /= nr
            S_new, m_new = _viol_sum(y, t)
            # feasibility success: below incumbent
            if m_new < best_val - 5e-13:
                x_best = y.copy()
                best_val = float(m_new)
                found = True
                break
            # accept if slack improved
            if S_new < S_old - 1e-15:
                # progress -> accept and enlarge
                x = y
                S_old = S_new
                m_old = m_new
                r = min(r * 1.3, 0.05)
                # if close to feasible, tighten K to most violated
                # keep as is
            else:
                # stall -> shrink, reject
                r *= 0.5
                if r < 5e-7:
                    break
                # if no progress for a while, continue with smaller r
                continue
        if found:
            break
        # if this target made partial slack progress, keep its x if max improved at all
        # (even if not below incumbent, keep best max)
        cur_max = _max_cos(x)
        if cur_max < best_val:
            x_best = x.copy()
            best_val = float(cur_max)
            found = True
            break

    # final ordinary LP polish under strict deadline
    try:
        x_best, best_val = _lp_polish(x_best, best_val, T_END)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

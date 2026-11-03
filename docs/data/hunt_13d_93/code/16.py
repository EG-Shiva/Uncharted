import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_attempt(x, radius, iu, cap=4000):
    n, d = x.shape
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    width = max(20.0 * radius, 5e-9)
    thresh = m - width
    sel = c_all > thresh
    cnt = int(sel.sum())
    if cnt == 0:
        return x, m, False
    if cnt > cap:
        idx = np.argpartition(-c_all, cap)[:cap]
        mask = np.zeros_like(sel)
        mask[idx] = True
        sel = mask
        cnt = cap
    I = iu[0][sel]
    J = iu[1][sel]
    p = cnt
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cI = np.arange(d)
    cols = np.concatenate([I[:, None] * d + cI, J[:, None] * d + cI, np.full((p, 1), n * d)], axis=1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    except Exception:
        return x, m, False
    obj = np.zeros(nv)
    obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return x, m, False
    if res.status != 0:
        return x, m, False
    dx = res.x[:-1].reshape(n, d)
    # predicted t
    # actual
    y = x + dx
    y = _norm(y)
    v = _max_cos(y)
    return y, v, True

def _pair_line_search(x, best_val):
    # try to push the current worst pair apart with tiny steps
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    ii = k // n
    jj = k % n
    if ii == jj:
        return x, best_val, False
    xi = x[ii]
    xj = x[jj]
    di = xj - (xj @ xi) * xi
    dj = xi - (xi @ xj) * xj
    ni = float(np.linalg.norm(di))
    nj = float(np.linalg.norm(dj))
    if ni < 1e-14 or nj < 1e-14:
        return x, best_val, False
    di /= ni
    dj /= nj
    # try geometrically decreasing eps, both / single moves
    for eps in (2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8, 5e-9):
        for mode in (0, 1, 2):
            y = x.copy()
            if mode == 0:
                y[ii] = xi - eps * di
                y[jj] = xj - eps * dj
            elif mode == 1:
                y[ii] = xi - eps * di
            else:
                y[jj] = xj - eps * dj
            y[ii] /= np.linalg.norm(y[ii])
            y[jj] /= np.linalg.norm(y[jj])
            v = _max_cos(y)
            if v < best_val - 1e-14:
                return y, v, True
    return x, best_val, False

def solve():
    np.random.seed(0)
    t0 = time.time()
    LIMIT = 170.0
    n, d = 93, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        rng = np.random.RandomState(0)
        best = _norm(rng.randn(n, d))
        best_val = _max_cos(best)
    x = best.copy()
    cur_val = best_val
    iu = np.triu_indices(n, 1)
    rng = np.random.RandomState(1)
    radius = 2e-5
    # micro trust-region descent, monotonic
    # reserve last ~25s for kicks
    phase1_end = t0 + 130.0
    stall = 0
    it = 0
    while time.time() < phase1_end and time.time() - t0 < LIMIT - 30:
        it += 1
        y, v, ok = _lp_attempt(x, radius, iu)
        if ok and v < cur_val - 1e-15:
            x = y
            cur_val = v
            if v < best_val:
                best = y.copy()
                best_val = v
            radius = min(radius * 1.3, 5e-5)
            stall = 0
            continue
        # LP did not improve: try tiny pair search at same scale
        y2, v2, ok2 = _pair_line_search(x, cur_val)
        if ok2 and v2 < cur_val:
            x = y2
            cur_val = v2
            if v2 < best_val:
                best = y2.copy()
                best_val = v2
            stall = 0
            # keep radius small
            radius = min(radius, 5e-6)
            continue
        # shrink
        radius *= 0.5
        stall += 1
        if radius < 5e-10:
            # re-inflate; tiny random nudge of worst points to change active set
            radius = 3e-6
            if stall > 12:
                # small isotropic jitter then re-project, keep only if not too bad
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                worst = np.argsort(rm)[-6:]
                yj = x.copy()
                yj[worst] += 1e-6 * rng.randn(len(worst), d)
                yj = _norm(yj)
                vj = _max_cos(yj)
                # accept as new iterate even if slightly worse to escape flat LP
                if vj < cur_val + 5e-9:
                    x = yj
                    cur_val = vj
                stall = 0
        if stall > 4 and radius < 1e-7:
            radius = min(radius * 2.0, 5e-6)
    # phase 2: small kicks + micro LP to try different topology
    kick_deadline = t0 + LIMIT - 4.0
    trial = 0
    while time.time() < kick_deadline:
        trial += 1
        remain = kick_deadline - time.time()
        if remain < 5:
            break
        # alternate kick sizes; very small because bottleneck is tiny
        sig = [0.002, 0.0008, 0.004, 0.0004, 0.0012][trial % 5]
        yk = best + sig * rng.randn(n, d)
        yk = _norm(yk)
        # short micro-LP run (few iterations)
        r2 = 1e-5
        cur2 = _max_cos(yk)
        xk = yk
        inner_end = time.time() + min(6.0, remain - 1.0)
        while time.time() < inner_end:
            yn, vn, ok = _lp_attempt(xk, r2, iu, cap=2500)
            if ok and vn < cur2 - 1e-15:
                xk = yn
                cur2 = vn
                r2 = min(r2 * 1.3, 3e-5)
            else:
                r2 *= 0.5
                if r2 < 1e-9:
                    break
                # pair try
                yp, vp, okp = _pair_line_search(xk, cur2)
                if okp:
                    xk = yp
                    cur2 = vp
                else:
                    if r2 < 2e-8:
                        break
        if cur2 < best_val:
            best = xk.copy()
            best_val = cur2
            x = xk.copy()
            cur_val = cur2
    best = _norm(best)
    return best

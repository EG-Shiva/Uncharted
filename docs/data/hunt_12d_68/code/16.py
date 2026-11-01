import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.3:
            break
        c = (x @ x.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_global(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def best_hole_vec(x_rest, rng, ncand=20000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def single_fg_factory(Xr, beta):
    # Xr: (m,d) fixed, optimize z in R^d
    def fg(z):
        r = max(np.linalg.norm(z), 1e-12)
        y = z / r
        s = Xr @ y
        m = float(s.max())
        e = np.exp(beta * (s - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    return fg

def opt_single_point(Xr, y_start, beta, maxiter=120):
    fg = single_fg_factory(Xr, beta)
    z0 = y_start.copy()
    try:
        r = minimize(fg, z0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter*2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        return y_start / (np.linalg.norm(y_start) + 1e-300)

def sweep_single_points(x, rng, k=8, ncand=20000):
    n, d = x.shape
    pm = per_max(x)
    order = np.argsort(-pm)[:k]
    for idx in order:
        mask = np.ones(n, dtype=bool); mask[idx] = False
        Xr = x[mask]
        cur = x[idx].copy()
        cur_max = float((Xr @ cur).max())
        # candidate starts
        hole = best_hole_vec(Xr, rng, ncand=ncand)
        starts = [cur, hole]
        # one perturbed hole
        pv = hole + 0.05 * rng.standard_normal(d)
        pv /= np.linalg.norm(pv)
        starts.append(pv)
        best_y, best_m = cur, cur_max
        for s in starts:
            for beta in (120.0, 450.0):
                y = opt_single_point(Xr, s, beta, maxiter=100)
                m = float((Xr @ y).max())
                if m < best_m:
                    best_m, best_y = m, y
        if best_m < cur_max - 1e-12:
            x[idx] = best_y
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x_init = rng0.standard_normal((68, 12))
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    else:
        bv = 1e9; x_init = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bv:
                bv, x_init = v, c.copy()
    x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    x_best = x_init.copy()
    best_val = max_cos(x_best)
    n, d = x_best.shape
    rng = np.random.default_rng(1234567)
    explore_end = t0 + TLIM - 48.0
    trial = 0
    # quick single-point descent first (exploits isolated max pair)
    try:
        if time.time() < explore_end:
            x = sweep_single_points(x_best.copy(), rng, k=12, ncand=30000)
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x, v
    except Exception:
        pass
    sigmas = [0.004, 0.012, 0.03, 0.08, 0.18, 0.35]
    while time.time() < explore_end and trial < 400:
        trial += 1
        u = rng.random()
        x = x_best.copy()
        if u < 0.38:
            # worst-hole + single-point style kick then global smooth
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            pm = G.max(axis=1)
            order = np.argsort(-pm)
            k = int(rng.integers(1, 4))
            for q in order[:k]:
                mask = np.ones(n, dtype=bool); mask[q] = False
                hole = best_hole_vec(x[mask], rng, ncand=12000)
                # mix hole with current to keep some memory
                alpha = float(rng.uniform(0.7, 1.0))
                v = alpha * hole + (1 - alpha) * rng.standard_normal(d)
                v /= (np.linalg.norm(v) + 1e-12)
                x[q] = v
            # shake neighbours of worst
            wi = int(order[0])
            nbrs = np.argsort(-(x @ x[wi]))[1:7]
            x[nbrs] += 0.02 * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.55:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.72:
            c0 = int(rng.integers(0, n))
            dots = x @ x[c0]
            order = np.argsort(-dots)
            k = int(rng.integers(3, 12))
            if rng.random() < 0.4:
                idx = rng.choice(n, size=k, replace=False)
            else:
                idx = order[:k]
            for q in idx:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                x[q] = v
            x += 0.004 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            uu = rng.standard_normal(d); uu /= np.linalg.norm(uu)
            proj = x @ uu
            thr = float(rng.uniform(0.0, 0.6))
            mask = proj > thr
            if mask.sum() < 2 or mask.sum() > n - 2:
                x += 0.04 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                vv = rng.standard_normal(d); vv -= (vv @ uu) * uu; vv /= (np.linalg.norm(vv) + 1e-12)
                amp = float(rng.uniform(0.1, 0.5))
                x[mask] += amp * vv
                x += 0.004 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        b1 = float(rng.uniform(60, 140))
        b2 = float(rng.uniform(250, 550))
        m1 = int(rng.integers(60, 140))
        m2 = int(rng.integers(60, 150))
        try:
            x = run_global(x, b1, m1)
        except Exception:
            continue
        if time.time() > explore_end:
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
            break
        # interleave single-point sweep half the time
        if rng.random() < 0.5:
            try:
                x = sweep_single_points(x, rng, k=6, ncand=8000)
            except Exception:
                pass
        try:
            x = run_global(x, b2, m2)
        except Exception:
            pass
        v = max_cos(x)
        if v < best_val - 1e-12:
            if v < best_val - 4e-7 and (explore_end - time.time()) > 5.0:
                try:
                    x = sweep_single_points(x, rng, k=10, ncand=12000)
                    v = max_cos(x)
                except Exception:
                    pass
                if (explore_end - time.time()) > 5.0:
                    x = lp_polish(x, min(3.0, explore_end - time.time() - 0.3))
                    v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
        elif v < best_val + 0.002 and (trial % 6 == 0) and (time.time() < explore_end - 4.0):
            y = lp_polish(x, 1.2)
            v2 = max_cos(y)
            if v2 < best_val:
                x_best, best_val = y.copy(), v2
    # final convergence: alternate single sweeps + high beta + LP
    remain = TLIM - (time.time() - t0)
    if remain > 5.0:
        try:
            for _ in range(3):
                if TLIM - (time.time() - t0) < 8.0:
                    break
                x_best = sweep_single_points(x_best, rng, k=14, ncand=25000)
        except Exception:
            pass
        try:
            x_best = run_global(x_best, 550.0, 160)
        except Exception:
            pass
        remain = TLIM - (time.time() - t0)
        if remain > 4.0:
            x_best = lp_polish(x_best, max(2.0, remain - 1.5))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard against NaN/zero
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best

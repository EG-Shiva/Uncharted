import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.25:
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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

def worst_index(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return int(np.argmax(G.max(axis=1)))

def best_hole(x_rest, rng, ncand=8000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def make_fg(n, d, IU0, IU1, beta):
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
        # symmetric P
        # build dense P via scatter
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    IU1 = (IU0[1], IU0[0])
    fg = make_fg(n, d, IU0, IU1, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

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
        bv = 1e9
        x_init = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bv:
                bv = v
                x_init = c.copy()
    x_init = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    best_val = max_cos(x_init)
    x_best = x_init.copy()
    n, d = x_best.shape
    rng = np.random.default_rng(20261031)
    explore_end = t0 + TLIM - 30.0
    trial = 0
    sigmas = [0.005, 0.015, 0.04, 0.10, 0.22, 0.40]
    while time.time() < explore_end and trial < 500:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        if u < 0.30:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.55:
            # block re-init: nearest neighbourhood of random center
            c0 = int(rng.integers(0, n))
            dots = x @ x[c0]
            order = np.argsort(-dots)
            k = int(rng.integers(3, 14))
            # sometimes random indices instead
            if rng.random() < 0.35:
                idx = rng.choice(n, size=k, replace=False)
            else:
                idx = order[:k]
            for q in idx:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                x[q] = v
            # shake rest slightly
            x += 0.005 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.70:
            # cap shift: move a spherical cap together
            uu = rng.standard_normal(d); uu /= np.linalg.norm(uu)
            proj = x @ uu
            thr = float(rng.uniform(0.1, 0.65))
            mask = proj > thr
            if mask.sum() < 2 or mask.sum() > n - 2:
                x += 0.05 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                vv = rng.standard_normal(d); vv -= (vv @ uu) * uu; vv /= (np.linalg.norm(vv)+1e-12)
                amp = float(rng.uniform(0.1, 0.6))
                x[mask] += amp * vv
                x += 0.005 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.85:
            # worst-hole reinsertion + neighbour shake
            wi = worst_index(x)
            mask = np.ones(n, dtype=bool); mask[wi] = False
            hole = best_hole(x[mask], rng, ncand=8000)
            x[wi] = hole
            G = x @ x.T
            nbrs = np.argsort(-G[wi])[:6]
            x[nbrs] += 0.03 * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # kick worst few
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(rng.integers(2, 6))
            idx = order[:k]
            if rng.random() < 0.5:
                for q in idx:
                    v = rng.standard_normal(d); v /= np.linalg.norm(v)
                    x[q] = v
            else:
                x[idx] += 0.45 * rng.standard_normal((len(idx), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # two-stage L-BFGS with random sharpness
        b1 = float(rng.uniform(50, 130))
        b2 = float(rng.uniform(250, 600))
        m1 = int(rng.integers(80, 180))
        m2 = int(rng.integers(80, 200))
        x = run_lbfgs(x, b1, m1)
        if time.time() > explore_end:
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
            break
        x = run_lbfgs(x, b2, m2)
        v = max_cos(x)
        if v < best_val - 1e-12:
            # squeeze promising basin if clearly better in smooth sense
            if v < best_val - 3e-7 and (explore_end - time.time()) > 4.0:
                x = lp_polish(x, min(4.0, explore_end - time.time() - 0.3))
                v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
        elif v < best_val + 0.003:
            if (trial % 5 == 0) and (time.time() < explore_end - 4.0):
                y = lp_polish(x, 1.5)
                v2 = max_cos(y)
                if v2 < best_val:
                    x_best, best_val = y.copy(), v2
        if time.time() > explore_end:
            break
    remain = TLIM - (time.time() - t0)
    if remain > 3.0:
        # final high-sharpness tune then LP
        try:
            x_best = run_lbfgs(x_best, 600.0, 150)
        except Exception:
            pass
        remain = TLIM - (time.time() - t0)
        if remain > 3.0:
            x_best = lp_polish(x_best, min(remain - 1.0, 27.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

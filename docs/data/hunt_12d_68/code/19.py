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

def soft_loads(x, beta=150.0):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    W = np.exp(beta * (G - m))
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=1), m

def best_hole(x_rest, rng, ncand=10000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(s))]

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.25:
            break
        if deadline is not None and time.time() > deadline - 0.6:
            break
        if radius < 1e-15:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        win = max(20.0 * radius, 5e-7)
        sel = c > m - win
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        # cap number of constraints for speed
        if int(sel.sum()) > 1600:
            # keep tightest 1600
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:1600]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            # accept, grow a bit
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
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

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def max_sgd(x, iters, lr0=0.02, m0=0.008, m1=2e-4):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(iters):
        t = s / max(iters - 1, 1)
        margin = m0 * (m1 / m0) ** t
        lr = lr0 * (1.0 - 0.85 * t)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G[iu] > m - margin
        if not np.any(sel):
            continue
        I = iu[0][sel]; J = iu[1][sel]
        Grad = np.zeros_like(x)
        np.add.at(Grad, I, x[J])
        np.add.at(Grad, J, x[I])
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        x = x - (lr / rms) * Tg
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
    rng = np.random.default_rng(777123)
    explore_end = t0 + TLIM - 32.0
    trial = 0
    sigmas = [0.004, 0.012, 0.03, 0.08, 0.18, 0.35]
    while time.time() < explore_end and trial < 400:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        if u < 0.22:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.40:
            k = int(rng.integers(3, 13))
            if rng.random() < 0.6:
                # load-biased choice for part of block
                ld, _ = soft_loads(x, beta=120.0)
                order = np.argsort(-ld)
                kk = min(k, n)
                # take top 2*kk then subsample
                pool = order[:min(n, 2 * kk + 4)]
                idx = rng.choice(pool, size=kk, replace=False)
            else:
                idx = rng.choice(n, size=k, replace=False)
            for q in idx:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                x[q] = v
            x += 0.004 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.55:
            # delete high-load points -> holes
            ld, _ = soft_loads(x, beta=200.0)
            order = np.argsort(-ld)
            k = int(rng.integers(1, 4))
            idx = order[:k]
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            rest = x[mask]
            for q in idx:
                hole = best_hole(rest, rng, ncand=6000 + int(rng.integers(0, 8000)))
                x[q] = hole
                rest = np.vstack([rest, hole[None, :]])
            # shake neighbours of reinserted
            x += 0.006 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.68:
            uu = rng.standard_normal(d); uu /= np.linalg.norm(uu)
            proj = x @ uu
            thr = float(rng.uniform(0.05, 0.6))
            mask = proj > thr
            if mask.sum() < 2 or mask.sum() > n - 2:
                x += 0.05 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                vv = rng.standard_normal(d); vv -= (vv @ uu) * uu
                vv /= (np.linalg.norm(vv) + 1e-12)
                amp = float(rng.uniform(0.08, 0.55))
                x[mask] += amp * vv
                x += 0.005 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.84:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            wi = int(np.argmax(per))
            mask = np.ones(n, dtype=bool); mask[wi] = False
            hole = best_hole(x[mask], rng, ncand=10000)
            x[wi] = hole
            # shake its new neighbours
            G2 = x @ x.T
            nbrs = np.argsort(-G2[wi])[1:7]
            x[nbrs] += 0.03 * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
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
                x[idx] += 0.5 * rng.standard_normal((len(idx), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        b1 = float(rng.uniform(60, 150))
        b2 = float(rng.uniform(280, 650))
        m1 = int(rng.integers(70, 150))
        m2 = int(rng.integers(70, 170))
        x = run_lbfgs(x, b1, m1)
        if time.time() > explore_end:
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
            break
        # occasional subgradient between stages to tie exact max
        if rng.random() < 0.35:
            x = max_sgd(x, iters=60, lr0=0.015, m0=0.005, m1=3e-4)
        x = run_lbfgs(x, b2, m2)
        v = max_cos(x)
        if v < best_val - 1e-12:
            if v < best_val - 2e-7 and (explore_end - time.time()) > 5.0:
                x = lp_polish(x, min(3.5, explore_end - time.time() - 0.3), deadline=explore_end)
                v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
        elif v < best_val + 0.0025:
            if (trial % 4 == 0) and (time.time() < explore_end - 3.0):
                y = lp_polish(x, 1.2, deadline=explore_end)
                v2 = max_cos(y)
                if v2 < best_val:
                    x_best, best_val = y.copy(), v2
        if time.time() > explore_end:
            break

    remain = TLIM - (time.time() - t0)
    if remain > 4.0:
        try:
            x_best = run_lbfgs(x_best, 700.0, 130)
        except Exception:
            pass
        try:
            x_best = max_sgd(x_best, iters=120, lr0=0.012, m0=0.004, m1=1e-4)
        except Exception:
            pass
        try:
            x_best = run_lbfgs(x_best, 1200.0, 120)
        except Exception:
            pass
        remain = TLIM - (time.time() - t0)
        if remain > 3.0:
            x_best = lp_polish(x_best, min(remain - 1.0, 28.0), deadline=t0 + TLIM - 0.8)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

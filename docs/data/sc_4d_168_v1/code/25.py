import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(x, seconds, t0=None):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    if t0 is None:
        t0 = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - t0 > seconds:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(25.0 * radius, 2e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5000:
            idx = np.argsort(-c)
            keep = np.zeros_like(sel)
            keep[idx[:5000]] = True
            sel = sel & keep.astype(bool)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = _norm(x + dx)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _gd_anneal(X, steps, b0, b1, lr0, lr1, noise0, rng):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(max(b0, 1e-9)); lb1 = np.log(max(b1, 1e-9))
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / max(steps, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if S <= 0:
            S = 1.0
        Grad = (W @ X) * (2.0 / S)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if noise0 > 0:
            amp = noise0 * (1.0 - f)
            if amp > 0:
                Nz = rng.normal(size=X.shape)
                Nz -= np.sum(Nz * X, axis=1, keepdims=True) * X
                Gt = Gt + amp * Nz
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        c1 = 1.0 - b1a ** t
        c2 = 1.0 - b2a ** t
        mhat = m / c1
        vhat = vv / c2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _refine_single(x0, Xref, steps=140, beta0=50.0, lr0=0.07):
    x = x0 / (np.linalg.norm(x0) + 1e-12)
    beta = beta0
    lr = lr0
    for _ in range(steps):
        dd = Xref @ x
        dm = float(dd.max())
        w = np.exp(beta * (dd - dm))
        s = float(w.sum()) + 1e-12
        g = (w @ Xref) / s
        g = g - float(np.dot(g, x)) * x
        gn = float(np.linalg.norm(g))
        if gn > 1e-12:
            g = g / gn
        else:
            break
        x = x - lr * g
        nn = float(np.linalg.norm(x))
        if nn == 0:
            break
        x = x / nn
        lr *= 0.993
        beta = min(beta * 1.008, 260.0)
    return x / (np.linalg.norm(x) + 1e-12)

def _reinsert(X, idx_remove, rng, n_cand=6000):
    n, d = X.shape
    idx_remove = list(map(int, np.asarray(idx_remove).tolist()))
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx_remove] = False
    cur = X[keep_mask].copy()
    new_pts = []
    for _ in idx_remove:
        cand = rng.normal(size=(n_cand, d))
        cand = _norm(cand)
        D = cand @ cur.T
        worst = D.max(axis=1)
        top = np.argsort(worst)[:4]
        best_pt = None
        best_val = 1e9
        for ti in top:
            pt = _refine_single(cand[int(ti)], cur, steps=130, beta0=50.0, lr0=0.07)
            val = float((cur @ pt).max())
            if val < best_val:
                best_val = val
                best_pt = pt
        new_pts.append(best_pt)
        cur = np.vstack([cur, best_pt.reshape(1, d)])
    Y = X.copy()
    Y[idx_remove] = np.array(new_pts)
    return _norm(Y)

def _degrees(X, tol=0.004):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = np.sum(G > m - tol, axis=1)
    return deg, m, G

def solve():
    t0 = time.time()
    TIME_LIMIT = 176.0
    SEARCH_END = 148.0
    rng = np.random.default_rng(11)
    n, d = 168, 4
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape == (n, d):
            xb = _norm(xb)
            if np.all(np.isfinite(xb)):
                cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (n, d):
            xr = _norm(xr)
            if np.all(np.isfinite(xr)):
                cands.append(xr)
    except Exception:
        pass
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(n, d)))
        best = _max_cos(bestX)
    else:
        vals = [_max_cos(c) for c in cands]
        bestX = cands[int(np.argmin(vals))].copy()
        best = float(min(vals))
    it = 0
    while True:
        now = time.time() - t0
        if now > SEARCH_END:
            break
        if now > TIME_LIMIT - 32.0:
            break
        it += 1
        mode = it % 12
        try:
            if mode == 0:
                Xtry = _norm(bestX + 0.008 * rng.normal(size=(n, d)))
                kw = dict(steps=800, b0=30.0, b1=420.0, lr0=0.010, lr1=0.0007, noise0=0.0)
            elif mode == 1:
                Xtry = _norm(bestX + 0.025 * rng.normal(size=(n, d)))
                kw = dict(steps=1200, b0=25.0, b1=380.0, lr0=0.012, lr1=0.0007, noise0=0.0)
            elif mode == 2:
                Xtry = _norm(bestX + 0.06 * rng.normal(size=(n, d)))
                deg, m, G = _degrees(Xtry, tol=0.004)
                k = 4
                victims = np.argsort(-deg)[:k].tolist()
                Xtry = _reinsert(Xtry, victims, rng, n_cand=6000)
                Xtry = _norm(Xtry + 0.005 * rng.normal(size=(n, d)))
                kw = dict(steps=1500, b0=20.0, b1=420.0, lr0=0.014, lr1=0.0006, noise0=0.001)
            elif mode == 3:
                Xtry = bestX.copy()
                G = Xtry @ Xtry.T; np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                neigh = set(np.argsort(-G[ij[0]])[:5].tolist()) | set(np.argsort(-G[ij[1]])[:5].tolist())
                sel = list(neigh)[:8]
                Xtry = _reinsert(Xtry, sel, rng, n_cand=6000)
                kw = dict(steps=1600, b0=22.0, b1=480.0, lr0=0.012, lr1=0.0006, noise0=0.001)
            elif mode == 4:
                deg, m, G = _degrees(bestX, tol=1e-4)
                ratt = np.argsort(deg)[:2].tolist()
                over = np.argsort(-deg)[:6].tolist()
                sel = list(set(ratt + over))[:7]
                Xtry = _reinsert(bestX, sel, rng, n_cand=7000)
                Xtry = _norm(Xtry + 0.008 * rng.normal(size=(n, d)))
                kw = dict(steps=1700, b0=18.0, b1=400.0, lr0=0.014, lr1=0.0006, noise0=0.002)
            elif mode == 5:
                Xtry = _norm(bestX + 0.15 * rng.normal(size=(n, d)))
                k = 5
                idx = rng.choice(n, size=k, replace=False).tolist()
                Xtry[idx] = _norm(rng.normal(size=(k, d)))
                kw = dict(steps=1800, b0=12.0, b1=320.0, lr0=0.016, lr1=0.0008, noise0=0.003)
            elif mode == 6:
                Xtry = bestX.copy()
                deg, _, _ = _degrees(Xtry, tol=0.004)
                hi = np.argsort(-deg)[:24]
                Xtry[hi] = _norm(Xtry[hi] + 0.12 * rng.normal(size=(len(hi), d)))
                kw = dict(steps=1400, b0=25.0, b1=500.0, lr0=0.010, lr1=0.0005, noise0=0.0)
            elif mode == 7:
                Xtry = _norm(bestX + 0.30 * rng.normal(size=(n, d)))
                k = 6
                idx = rng.choice(n, size=k, replace=False).tolist()
                Xtry = _reinsert(Xtry, idx, rng, n_cand=5000)
                kw = dict(steps=1800, b0=10.0, b1=300.0, lr0=0.016, lr1=0.0008, noise0=0.002)
            elif mode == 8:
                Xtry = _norm(bestX + 0.045 * rng.normal(size=(n, d)))
                k = 2 + (it % 3)
                idx = rng.choice(n, size=k, replace=False).tolist()
                Xtry = _reinsert(Xtry, idx, rng, n_cand=5000)
                kw = dict(steps=1300, b0=28.0, b1=450.0, lr0=0.011, lr1=0.0006, noise0=0.0)
            elif mode == 9:
                Xtry = _norm(bestX + 0.10 * rng.normal(size=(n, d)))
                kw = dict(steps=1300, b0=15.0, b1=350.0, lr0=0.013, lr1=0.0007, noise0=0.004)
            elif mode == 10:
                if it % 36 == 10:
                    Xtry = _norm(rng.normal(size=(n, d)))
                    kw = dict(steps=2200, b0=8.0, b1=300.0, lr0=0.016, lr1=0.0008, noise0=0.0)
                else:
                    Xtry = _norm(bestX + 0.035 * rng.normal(size=(n, d)))
                    kw = dict(steps=1200, b0=30.0, b1=520.0, lr0=0.009, lr1=0.0005, noise0=0.0)
            else:
                Xtry = _norm(bestX + 0.055 * rng.normal(size=(n, d)))
                deg, _, _ = _degrees(Xtry, tol=0.003)
                victims = np.argsort(-deg)[:3].tolist()
                victims += rng.choice(n, size=2, replace=False).tolist()
                victims = list(set(victims))[:5]
                Xtry = _reinsert(Xtry, victims, rng, n_cand=5000)
                kw = dict(steps=1500, b0=20.0, b1=420.0, lr0=0.013, lr1=0.0006, noise0=0.001)
        except Exception:
            continue
        try:
            Xopt = _gd_anneal(Xtry, kw["steps"], kw["b0"], kw["b1"], kw["lr0"], kw["lr1"], kw["noise0"], rng)
        except Exception:
            continue
        try:
            coarse = _max_cos(Xopt)
        except Exception:
            continue
        if coarse < best:
            best = coarse
            bestX = Xopt.copy()
            rem = TIME_LIMIT - (time.time() - t0) - 8.0
            if rem > 2.5:
                try:
                    Xs = _lp_polish(bestX, min(3.5, rem), t0=time.time())
                    vs = _max_cos(Xs)
                    if vs < best:
                        best = vs
                        bestX = Xs
                except Exception:
                    pass
        elif coarse < best + 0.006:
            rem = SEARCH_END - (time.time() - t0)
            if rem > 4.0:
                try:
                    Xs = _lp_polish(Xopt, 1.2, t0=time.time())
                    vs = _max_cos(Xs)
                    if vs < best:
                        best = vs
                        bestX = Xs.copy()
                except Exception:
                    pass
        if time.time() - t0 > SEARCH_END:
            break
    try:
        rem = TIME_LIMIT - (time.time() - t0) - 1.0
        if rem > 2.0:
            Xs = _lp_polish(bestX, min(26.0, rem), t0=time.time())
            vs = _max_cos(Xs)
            if vs < _max_cos(bestX):
                bestX = Xs
    except Exception:
        pass
    bestX = _norm(bestX)
    return bestX

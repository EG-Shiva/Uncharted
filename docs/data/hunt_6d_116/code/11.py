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
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt

def _smooth(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m_ad = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for it in range(1, iters + 1):
        frac = it / iters
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = G.max()
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - s * X
        b1p *= b1
        b2p *= b2
        m_ad *= b1
        m_ad += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m_ad / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_hole(y, others, iters=250, a0=150.0, a1=2500.0, lr0=0.05, lr1=0.005):
    y = y / np.linalg.norm(y)
    for it in range(1, iters + 1):
        frac = it / iters
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        dots = others @ y
        m = dots.max()
        w = np.exp(alpha * (dots - m))
        sw = w.sum()
        if sw == 0 or not np.isfinite(sw):
            break
        w /= sw
        grad = w @ others
        gt = grad - (grad @ y) * y
        y = y - lr * gt
        ny = np.linalg.norm(y)
        if ny == 0 or not np.isfinite(ny):
            break
        y /= ny
    return y

def _best_hole_cand(others, rng, m_try=8000):
    cand = rng.standard_normal((m_try, 6))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[int(np.argmin(worst))]

def _lp_polish(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        G = X @ X.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) > 1500:
            order = np.argsort(c_all[idx])[-1500:]
            idx = idx[order]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        Xb = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        Xb = None
    try:
        Xr = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        Xr = None
    cands = []
    if Xb is not None and Xb.shape == (116, 6):
        cands.append(_norm(Xb))
    if Xr is not None and Xr.shape == (116, 6):
        cands.append(_norm(Xr))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best_val = float(vals[bi])
    cur = Xbest.copy()
    cur_val = best_val
    n, d = 116, 6
    explore_end = t0 + 132.0
    it = 0
    while time.time() < explore_end:
        it += 1
        remain = explore_end - time.time()
        if remain < 2.0:
            break
        progress = 1.0 - remain / 132.0
        Temp = 5e-4 * (1.0 - progress) + 1e-5
        mode = it % 6
        Y = cur.copy()
        try:
            if mode == 0:
                # rattler hole reinsert
                _, cnt = _contacts(cur, tol=1e-7)
                ratt = np.where(cnt <= 1)[0]
                if len(ratt) == 0:
                    ratt = np.argsort(cnt)[:5]
                k = min(len(ratt), 3)
                pick = rng.choice(ratt, k, replace=False) if len(ratt) >= k else ratt
                for r in pick:
                    others = np.delete(Y, r, axis=0)
                    y0 = _best_hole_cand(others, rng, m_try=8000)
                    Y[r] = _refine_hole(y0, others, iters=250)
                Y = Y + 0.002 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, iters=350, a0=120.0, a1=1400.0, lr0=0.008, lr1=0.001)
                v = _max_cos(Y)
                if v < best_val + 0.002 and remain > 8:
                    Y = _lp_polish(Y, seconds=1.0)
            elif mode == 1:
                # worst point to largest hole
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                w = int(np.argmax(per))
                others = np.delete(Y, w, axis=0)
                y0 = _best_hole_cand(others, rng, m_try=12000)
                Y[w] = _refine_hole(y0, others, iters=300)
                Y = _norm(Y)
                Y = _smooth(Y, iters=250, a0=200.0, a1=1800.0, lr0=0.005, lr1=0.0008)
                v = _max_cos(Y)
                if v < best_val + 0.0015 and remain > 8:
                    Y = _lp_polish(Y, seconds=1.0)
            elif mode == 2:
                # jammed kick
                _, cnt = _contacts(cur, tol=1e-7)
                order = np.argsort(-cnt)
                k = 8
                idx = order[:k]
                Y[idx] = Y[idx] + 0.12 * rng.standard_normal((k, d))
                Y = Y + 0.008 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, iters=450, a0=80.0, a1=1200.0, lr0=0.010, lr1=0.001)
                v = _max_cos(Y)
                if v < best_val + 0.002 and remain > 8:
                    Y = _lp_polish(Y, seconds=0.8)
            elif mode == 3:
                # global shake
                sigma = float(rng.uniform(0.02, 0.06))
                Y = Y + sigma * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, iters=400, a0=100.0, a1=1600.0, lr0=0.009, lr1=0.001)
                v = _max_cos(Y)
                if v < best_val + 0.001 and remain > 10:
                    Y = _lp_polish(Y, seconds=0.8)
            elif mode == 4:
                # random subset teleport to holes
                k = int(rng.integers(4, 11))
                pick = rng.choice(n, k, replace=False)
                for r in pick:
                    others = np.delete(Y, r, axis=0)
                    y0 = _best_hole_cand(others, rng, m_try=4000)
                    Y[r] = _refine_hole(y0, others, iters=150)
                Y = _norm(Y)
                Y = _smooth(Y, iters=350, a0=150.0, a1=1500.0, lr0=0.007, lr1=0.001)
                v = _max_cos(Y)
            else:
                # fine smooth with tiny noise
                Y = Y + 0.004 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, iters=500, a0=300.0, a1=2500.0, lr0=0.004, lr1=0.0005)
                v = _max_cos(Y)
        except Exception:
            continue
        try:
            v = _max_cos(Y)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            Xbest = Y.copy()
            best_val = v
            cur = Y.copy()
            cur_val = v
        elif v < cur_val:
            cur = Y.copy()
            cur_val = v
        else:
            # SA acceptance, reset if drifted too far
            if cur_val > best_val + 0.005:
                cur = Xbest.copy()
                cur_val = best_val
            elif rng.random() < np.exp(-(v - cur_val) / Temp):
                cur = Y.copy()
                cur_val = v
    # final squeeze: smooth + long LP on best
    try:
        if time.time() - t0 < 170:
            Xbest = _smooth(Xbest, iters=600, a0=250.0, a1=2500.0, lr0=0.003, lr1=0.0004)
            nv2 = _max_cos(Xbest)
            if nv2 < best_val:
                best_val = nv2
    except Exception:
        pass
    try:
        left = 178.0 - (time.time() - t0)
        if left > 5:
            Xp = _lp_polish(Xbest, seconds=min(left - 1.0, 28.0))
            vp = _max_cos(Xp)
            if vp < best_val:
                Xbest = Xp
                best_val = vp
    except Exception:
        pass
    Xbest = _norm(Xbest)
    return Xbest

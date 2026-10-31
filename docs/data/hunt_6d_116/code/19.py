import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt, G

def _smooth(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    b1p, b2p = 1.0, 1.0
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
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = _norm(X)
    return X

def _deep_hole(others, rng, m_try=4000, refine=60):
    cand = rng.standard_normal((m_try, 6))
    cand = _norm(cand)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    y = cand[int(np.argmin(worst))].copy()
    step = 0.05
    for _ in range(refine):
        d = others @ y
        j = int(np.argmax(d))
        g = others[j] - (others[j] @ y) * y
        n = float(np.linalg.norm(g))
        if n < 1e-15:
            break
        g /= n
        y = y - step * g
        ny = float(np.linalg.norm(y))
        if ny == 0:
            break
        y /= ny
        step *= 0.97
    ny = float(np.linalg.norm(y))
    if ny == 0 or not np.all(np.isfinite(y)):
        y = cand[0] / np.linalg.norm(cand[0])
    else:
        y = y / ny
    return y

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    best_x = X.copy()
    radius = 2e-3
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
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c_all > thresh)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1500:
            order = np.argsort(c_all[idx])[-1500:]
            idx = idx[order]
        sel = np.zeros_like(c_all, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]
        J = iu[1][sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _maxcos(y)
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
    np.random.seed(0)
    rng = np.random.default_rng(12345)
    def clean(a):
        a = np.asarray(a, dtype=float)
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        return _norm(a)
    cands = []
    try:
        c = clean(np.load("best.npy"))
        if c is not None:
            cands.append(c)
    except Exception:
        pass
    try:
        c = clean(np.load("record.npy"))
        if c is not None:
            cands.append(c)
    except Exception:
        pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = x_best.copy()
    cur_val = best_val
    n, d = 116, 6
    explore_end = t0 + 132.0
    temp0, temp1 = 0.004, 0.00012
    it = 0
    while time.time() < explore_end:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(132.0, 1e-9)
        temp = temp0 * (temp1 / temp0) ** min(max(frac, 0.0), 1.0)
        _, cnt, G = _contacts(cur_x if (it % 4 != 0) else x_best, tol=1e-7)
        base = cur_x.copy() if (it % 4 != 0) else x_best.copy()
        # periodic restart from best to avoid drift
        if it % 25 == 0:
            base = x_best.copy()
            cnt = _contacts(base, tol=1e-7)[1]
        Y = base.copy()
        mode = it % 7
        if mode == 0:
            sigma = 0.015 + 0.015 * rng.random()
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=800, a0=50.0, a1=500.0, lr0=0.012, lr1=0.002)
        elif mode == 1:
            order = np.argsort(-cnt)
            k = 8
            idx = order[:k]
            Y[idx] = Y[idx] + 0.14 * rng.standard_normal((k, d))
            Y = Y + 0.007 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=900, a0=50.0, a1=550.0, lr0=0.012, lr1=0.002)
        elif mode == 2:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:6]
            else:
                ratt = ratt[:max(6, len(ratt))]
            Y[ratt] = rng.standard_normal((len(ratt), d))
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=1200, a0=60.0, a1=650.0, lr0=0.015, lr1=0.002)
        elif mode == 3:
            ratt = np.where(cnt <= 2)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:5]
            perm = rng.permutation(ratt)[:6]
            for r in perm:
                others = np.delete(Y, r, axis=0)
                Y[r] = _deep_hole(others, rng, m_try=3500, refine=50)
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=1200, a0=60.0, a1=650.0, lr0=0.015, lr1=0.002)
        elif mode == 4:
            # worst-pair surgery: find top worst pairs
            Gc = Y @ Y.T
            np.fill_diagonal(Gc, -2.0)
            for _ in range(5):
                a, b = np.unravel_index(int(np.argmax(Gc)), Gc.shape)
                Y[a] = Y[a] + 0.06 * rng.standard_normal(d) + 0.04 * (Y[a] - Y[b])
                Y[b] = Y[b] + 0.06 * rng.standard_normal(d) + 0.04 * (Y[b] - Y[a])
                Gc[a, b] = Gc[b, a] = -2.0
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=900, a0=55.0, a1=600.0, lr0=0.013, lr1=0.002)
        elif mode == 5:
            sigma = 0.05 + 0.05 * rng.random()
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=850, a0=50.0, a1=500.0, lr0=0.012, lr1=0.002)
        else:
            # jammed hub hole-reinsertion
            hub = int(np.argmax(cnt))
            others = np.delete(Y, hub, axis=0)
            Y[hub] = _deep_hole(others, rng, m_try=3500, refine=50)
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, iters=1100, a0=60.0, a1=650.0, lr0=0.014, lr1=0.002)
        v = _maxcos(Y)
        # selective LP refinement
        if v < cur_val + 0.004 or v < best_val + 0.001:
            if time.time() < explore_end - 1.2:
                Y2 = _lp_polish(Y, seconds=0.9)
                v2 = _maxcos(Y2)
                Y, v = Y2, v2
        if v < best_val - 1e-12:
            x_best = Y.copy()
            best_val = v
            cur_x = Y.copy()
            cur_val = v
        else:
            # metropolis move for cur to explore new basin
            if v <= cur_val or rng.random() < np.exp((cur_val - v) / max(temp, 1e-9)):
                cur_x = Y.copy()
                cur_val = v
        if time.time() >= explore_end:
            break
    try:
        remain = 170.0 - (time.time() - t0)
        if remain > 10:
            x_best = _smooth(x_best, iters=500, a0=500.0, a1=1300.0, lr0=0.004, lr1=0.0005)
            remain = 170.0 - (time.time() - t0)
        if remain > 6:
            sec = max(5.0, min(35.0, remain - 1.0))
            x_best = _lp_polish(x_best, seconds=float(sec))
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

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

def _per_point(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pm = G.max(axis=1)
    return pm, G

def _smooth_adam(X, iters, a0, a1, lr0, lr1):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for it in range(1, iters + 1):
        frac = it / float(iters)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Grad = W @ X
        proj = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - proj * X
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

def _deep_hole(others, rng, m_try=3000, refine=50):
    cand = rng.standard_normal((m_try, 6))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-15
    dots = cand @ others.T
    worst = dots.max(axis=1)
    y = cand[int(np.argmin(worst))].copy()
    step = 0.04
    for _ in range(refine):
        d = others @ y
        j = int(np.argmax(d))
        g = others[j] - (others[j] @ y) * y
        n = float(np.linalg.norm(g))
        if n < 1e-12:
            break
        g /= n
        y = y - step * g
        y /= np.linalg.norm(y) + 1e-15
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bestX = X.copy()
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
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if len(idx) > 1600:
            order = np.argsort(c[idx])[-1600:]
            idx = idx[order]
        if len(idx) == 0:
            break
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        p = len(I)
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
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def solve():
    t0 = time.time()
    BUDGET = 172.0
    def _clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    cands = []
    try:
        cands.append(_clean(np.load("best.npy")))
    except Exception:
        pass
    try:
        cands.append(_clean(np.load("record.npy")))
    except Exception:
        pass
    cands = [c for c in cands if c is not None]
    if not cands:
        rng0 = np.random.default_rng(0)
        cands.append(_norm(rng0.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = x_best.copy()
    cur_val = best_val
    rng = np.random.default_rng(1234567)
    explore_end = t0 + BUDGET - 30.0
    it = 0
    temp0, temp1 = 0.0035, 0.00012
    while time.time() < explore_end:
        it += 1
        frac = min(1.0, (time.time() - t0) / max(explore_end - t0, 1e-9))
        temp = temp0 + (temp1 - temp0) * frac
        trial = cur_x.copy()
        pm, Gfull = _per_point(trial)
        mcur = float(pm.max())
        cnt = (Gfull > mcur - 1e-7).sum(axis=1)
        order_worst = np.argsort(-pm)
        order_cnt = np.argsort(-cnt)
        op = rng.random()
        if op < 0.24:
            sigma = float(rng.uniform(0.015, 0.10))
            trial = trial + sigma * rng.standard_normal(trial.shape)
            trial = _norm(trial)
            trial = _smooth_adam(trial, iters=int(rng.integers(300, 550)),
                                 a0=25, a1=180, lr0=0.025, lr1=0.004)
        elif op < 0.48:
            k = int(rng.integers(5, 12))
            idx = order_cnt[:k].copy()
            for ii in range(k):
                if rng.random() < 0.25:
                    idx[ii] = int(rng.integers(0, 116))
            sigma = float(rng.uniform(0.08, 0.30))
            trial[idx] += sigma * rng.standard_normal((k, 6))
            trial = _norm(trial)
            trial = _smooth_adam(trial, iters=int(rng.integers(350, 600)),
                                 a0=25, a1=220, lr0=0.03, lr1=0.003)
        elif op < 0.68:
            nr = int(rng.integers(1, 4))
            worst_idx = order_worst[:nr].copy()
            for ii in range(nr):
                if rng.random() < 0.25:
                    worst_idx[ii] = int(rng.integers(0, 116))
            for wi in worst_idx:
                mask = np.ones(116, dtype=bool)
                mask[int(wi)] = False
                others = trial[mask]
                y = _deep_hole(others, rng, m_try=3500, refine=55)
                y = y + 0.004 * rng.standard_normal(6)
                y /= np.linalg.norm(y) + 1e-15
                trial[int(wi)] = y
            trial = _norm(trial)
            trial = _smooth_adam(trial, iters=int(rng.integers(350, 550)),
                                 a0=35, a1=240, lr0=0.02, lr1=0.003)
        elif op < 0.86:
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(116, 1)
            c = G[iu]
            top = np.argsort(-c)[:24]
            involved = np.unique(np.concatenate([iu[0][top], iu[1][top]]))
            if len(involved) > 14:
                involved = rng.choice(involved, size=14, replace=False)
            sigma = float(rng.uniform(0.06, 0.22))
            trial[involved] += sigma * rng.standard_normal((len(involved), 6))
            trial = _norm(trial)
            trial = _smooth_adam(trial, iters=int(rng.integers(350, 600)),
                                 a0=30, a1=220, lr0=0.028, lr1=0.003)
        else:
            k = int(rng.integers(8, 20))
            idx = rng.choice(116, size=k, replace=False)
            sigma = float(rng.uniform(0.10, 0.35))
            trial[idx] += sigma * rng.standard_normal((k, 6))
            trial = _norm(trial)
            trial = _smooth_adam(trial, iters=int(rng.integers(400, 650)),
                                 a0=20, a1=200, lr0=0.03, lr1=0.003)
        v = _maxcos(trial)
        if v < best_val + 0.0025 and (it % 20 == 0) and (time.time() < explore_end - 2.0):
            trial = _lp_polish(trial, 1.0)
            v = _maxcos(trial)
        if v < best_val - 1e-12:
            best_val = v
            x_best = trial.copy()
            cur_x = trial.copy()
            cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(temp, 1e-12)):
            if v < cur_val + 0.03:
                cur_x = trial.copy()
                cur_val = v
        if it % 100 == 0 and time.time() < explore_end:
            cur_x = x_best.copy()
            cur_val = best_val
        if time.time() >= explore_end:
            break
    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 5:
        x_best = _lp_polish(x_best, remain)
    else:
        x_best = _norm(x_best)
    x_best = _norm(x_best)
    return x_best

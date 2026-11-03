import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _smooth(X, steps, p0, p1, lr0=0.04, lr1=0.01):
    n, d = X.shape
    x = X
    for s in range(steps):
        frac = s / max(1, steps - 1)
        # log_interp p
        lp0 = np.log(p0); lp1 = np.log(p1)
        p = float(np.exp(lp0 + (lp1 - lp0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        # ReLU
        np.maximum(G, 0.0, out=G)
        m = float(np.max(G))
        if m <= 1e-12:
            break
        # weights (G/m)^(p-1)
        # avoid divide issues
        Gn = G / m
        # power; clip tiny to 0 to save? just power
        W = Gn ** (p - 1.0)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        mn = float(np.mean(np.linalg.norm(grad, axis=1))) + 1e-12
        step = lr / mn
        x = x - step * grad
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def _void_points(Xkeep, k, rng, n_cand=2500):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cn = np.linalg.norm(cand, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    cand /= cn
    dots = cand @ Xkeep.T
    vals = np.max(dots, axis=1)
    chosen = []
    # greedy diverse farthest
    for _ in range(k):
        if chosen:
            S = np.stack(chosen)
            d2 = cand @ S.T
            d2m = np.max(d2, axis=1)
            score = np.maximum(vals, d2m)
        else:
            score = vals
        bi = int(np.argmin(score))
        chosen.append(cand[bi].copy())
        vals[bi] = np.inf
    return np.stack(chosen)

def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < budget and radius > 1e-14 and it < 60:
        it += 1
        if time.time() - t0 + 0.5 > budget and radius < 8e-4:
            break
        Gf = x @ x.T
        c = Gf[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 2e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c, -2500)[-2500]
            sel = c >= kth
            # subsample if still too many
            cnt2 = int(np.sum(sel))
            if cnt2 > 3000:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:3000]
                mask = np.zeros_like(c, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    bestX = None; bestV = np.inf
    for c in cands:
        v = _max_cos(c)
        if v < bestV:
            bestV = v; bestX = c.copy()
    # quick baseline smooth of random to have diverse elite
    # elite hopping
    curX = bestX.copy(); curV = bestV
    T0 = 8e-04
    it = 0
    # reserve for LP
    LP_RESERVE = 28.0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < LP_RESERVE + 2.0:
            break
        it += 1
        frac = 1.0 - rem / BUDGET  # 0->1
        Temp = T0 * (1.0 - 0.85 * frac) + 1e-06
        mode = it % 10
        y = None
        try:
            if mode in (0, 1):
                sig = 0.015 if mode == 0 else 0.05
                y = curX + rng.standard_normal((n, d)) * sig
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=280, p0=18, p1=120, lr0=0.05, lr1=0.01)
            elif mode == 2:
                sig = 0.22
                y = curX + rng.standard_normal((n, d)) * sig
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=350, p0=15, p1=150, lr0=0.05, lr1=0.008)
            elif mode in (3, 4, 6, 8):
                k = 3 if mode in (3, 8) else (5 if mode == 4 else 8)
                rm, _ = _row_max(curX)
                worst = np.argsort(-rm)[:k]
                # sometimes pick random among top-20 to diversify
                if mode == 8 and rng.random() < 0.5:
                    pool = np.argsort(-rm)[:20]
                    worst = rng.choice(pool, size=k, replace=False)
                mask = np.ones(n, dtype=bool); mask[worst] = False
                Xkeep = curX[mask]
                voids = _void_points(Xkeep, k, rng, n_cand=2500)
                y = curX.copy()
                y[worst] = voids
                # small shake of neighbours of worst to break cage
                if mode == 4:
                    y = y + rng.standard_normal((n, d)) * 0.012
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    # restore voids? keep approx
                y = _smooth(y, steps=320, p0=20, p1=160, lr0=0.045, lr1=0.008)
            elif mode == 5:
                # fresh random with strong anneal (new basin, balanced)
                y = rng.standard_normal((n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=500, p0=12, p1=140, lr0=0.06, lr1=0.01)
            elif mode == 7:
                # crossover: mix best and cur
                y = curX.copy()
                # swap random block with best
                idx = rng.choice(n, size=n // 6, replace=False)
                y[idx] = bestX[idx]
                y = y + rng.standard_normal((n, d)) * 0.03
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=300, p0=18, p1=130, lr0=0.05, lr1=0.01)
            else:
                # mode 9: shake best directly
                sig = 0.10
                y = bestX + rng.standard_normal((n, d)) * sig
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=320, p0=16, p1=150, lr0=0.05, lr1=0.008)
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < curV or rng.random() < np.exp((curV - v) / max(Temp, 1e-09)):
            curX, curV = y, v
        if v < bestV - 1e-12:
            bestX, bestV = y.copy(), v
        # opportunistic: if cur stuck, restart cur from best
        if it % 25 == 0 and curV > bestV + 0.002:
            curX, curV = bestX.copy(), bestV
    # final sharpen with high-p smooth before LP
    try:
        bestX = _smooth(bestX.copy(), steps=250, p0=60, p1=300, lr0=0.02, lr1=0.005)
        bestV = _max_cos(bestX)
    except Exception:
        pass
    rem = BUDGET - (time.time() - t0)
    lp_time = max(4.0, min(26.0, rem - 1.0))
    if rem > 5.0:
        try:
            px = _lp_polish(bestX, budget=lp_time)
            pv = _max_cos(px)
            if pv < bestV and np.all(np.isfinite(px)):
                bestX, bestV = px, pv
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # tiny shrink guard: ensure finite
    bestX = np.where(np.isfinite(bestX), bestX, rng.standard_normal((n, d)))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

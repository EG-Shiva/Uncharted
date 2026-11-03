import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(x, steps, T0, T1, lr0, lr1, mu):
    n, d = x.shape
    vel = np.zeros_like(x)
    # geometric schedules
    lt0 = np.log(T0); lt1 = np.log(T1)
    lrlt0 = np.log(lr0); lrlt1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(lrlt0 + (lrlt1 - lrlt0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        # softmax weights
        W = np.exp((G - m) / T)
        Z = float(W.sum())
        if not np.isfinite(Z) or Z == 0:
            break
        grad = (W @ x) / Z
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        # clip to avoid blow up
        vel = mu * vel + gt
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    HARD_END = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        # try reshape / truncate
        tmp = rng.standard_normal((99, 13))
        m = min(xb.shape[0], 99)
        tmp[:m, :min(xb.shape[1], 13)] = xb[:m, :min(xb.shape[1], 13)]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = xb.copy()
    best_val = _max_cos(best)
    trial = 0
    # main basin-hopping loop
    while time.time() < HARD_END - 8:
        trial += 1
        mode = trial % 5
        y = best.copy()
        if mode == 0:
            sig = 0.06
            y = y + sig * rng.standard_normal(y.shape)
        elif mode == 1:
            sig = 0.15
            y = y + sig * rng.standard_normal(y.shape)
        elif mode == 2:
            k = 2 + (trial % 6)  # 2..7
            idx = rng.choice(99, size=k, replace=False)
            r = rng.standard_normal((k, 13))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
            y = y + 0.02 * rng.standard_normal(y.shape)
        elif mode == 3:
            sig = 0.30
            y = y + sig * rng.standard_normal(y.shape)
        else:
            # single overloaded-point style: move a random block strongly
            k = 10
            idx = rng.choice(99, size=k, replace=False)
            y[idx] = y[idx] + 0.5 * rng.standard_normal((k, 13))
            y = y + 0.05 * rng.standard_normal(y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth repulsion annealing: coarse then fine
        y = _anneal(y, steps=2200, T0=0.03, T1=0.002, lr0=0.20, lr1=0.03, mu=0.85)
        if time.time() > HARD_END - 8:
            v = _max_cos(y)
            if v < best_val:
                best, best_val = y, v
            break
        y = _anneal(y, steps=1200, T0=0.005, T1=0.0008, lr0=0.05, lr1=0.012, mu=0.90)
        v = _max_cos(y)
        # screen then LP polish
        if v < best_val + 0.015:
            # leave at least ~20s for final polish
            remain = HARD_END - time.time()
            if remain < 10:
                if v < best_val:
                    best, best_val = y, v
                break
            short = 3.0 if remain > 30 else 1.5
            y2 = _lp_polish(y, seconds=short, t_end=HARD_END - 6)
            v2 = _max_cos(y2)
            if v2 < best_val - 1e-12:
                best, best_val = y2.copy(), v2
                # continue exploiting new basin
            # opportunistic: if not improved but very close, keep diversity rarely
        if trial > 60:
            break
    # final intensive polish of incumbent
    rem = HARD_END - time.time()
    if rem > 4:
        try:
            best = _lp_polish(best, seconds=min(rem - 1.0, 25.0), t_end=HARD_END - 1.0)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # tiny shrink guard + NaN guard
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best

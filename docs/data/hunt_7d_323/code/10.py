import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_anneal(X, steps, T0, T1, e0, e1):
    X = X.copy()
    # log interpolation factors
    lt0, lt1 = np.log(T0), np.log(T1)
    le0, le1 = np.log(e0), np.log(e1)
    n = X.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = np.exp(lt0 + (lt1 - lt0) * f)
        eta = np.exp(le0 + (le1 - le0) * f)
        G = X @ X.T
        # max off diag
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights, subtract max for stability (<=0)
        # use in-place exp where possible
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        r = E.sum(axis=1)
        # avoid div0
        r = np.maximum(r, 1e-30)
        A = (E @ X) / r[:, None]
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        X = X - eta * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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

def solve():
    import time as _time
    t0 = _time.time()
    TIME_LIMIT = 172.0
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    try:
        xr = np.load("record.npy")
        if max_cos(xr / np.linalg.norm(xr, axis=1, keepdims=True)) < max_cos(xb / np.linalg.norm(xb, axis=1, keepdims=True)):
            xb = xr
    except Exception:
        pass
    xb = np.asarray(xb, dtype=float)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    # enforce shape (323,7) if needed
    assert xb.shape == (323, 7)
    best = max_cos(xb)
    Xbest = xb.copy()
    rng = np.random.default_rng(0)
    iu0 = np.triu_indices(n, 1)

    def find_hole(Xrest, m):
        C = rng.standard_normal(size=(m, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Xrest.T
        smax = S.max(axis=1)
        return C[np.argmin(smax)]

    trial = 0
    # reserve ~28s for final LP
    while _time.time() - t0 < TIME_LIMIT - 28:
        trial += 1
        Gfull = Xbest @ Xbest.T
        np.fill_diagonal(Gfull, -2.0)
        # degrees for victim selection
        thresh = best - 0.02
        deg = (Gfull > thresh).sum(axis=1)
        mode = trial % 5
        Xt = Xbest.copy()
        if mode == 0:
            Xt += rng.standard_normal(size=Xt.shape) * 0.006
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, e0, e1 = 600, 0.02, 0.002, 0.04, 0.008
        elif mode == 1:
            idx = rng.choice(n, size=25, replace=False)
            Xt[idx] += rng.standard_normal(size=(25, d)) * 0.06
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, e0, e1 = 700, 0.025, 0.0025, 0.05, 0.008
        elif mode == 2 or mode == 3:
            # hub -> hole relocation
            # pick victim among top degrees
            top = np.argsort(deg)[-12:]
            v = int(rng.choice(top))
            mask = np.ones(n, dtype=bool); mask[v] = False
            Xrest = Xbest[mask]
            hole = find_hole(Xrest, 3000)
            hole = hole + rng.standard_normal(size=d) * 0.01
            hole /= np.linalg.norm(hole)
            Xt[v] = hole
            # jitter neighbors of victim slightly to break jam
            # neighbors = closest 10 to victim in old config
            dists = Xbest @ Xbest[v]
            nbrs = np.argsort(dists)[-12:-1]
            Xt[nbrs] += rng.standard_normal(size=(len(nbrs), d)) * 0.02
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            if mode == 2:
                steps, T0, T1, e0, e1 = 800, 0.03, 0.002, 0.06, 0.008
            else:
                steps, T0, T1, e0, e1 = 800, 0.015, 0.0015, 0.04, 0.006
        else:
            # worst-pair split
            # get top 20 worst pairs
            flat = Gfull[iu0]
            k = min(20, flat.size)
            part = np.argpartition(-flat, k - 1)[:k]
            pick = int(rng.choice(part))
            i, j = int(iu0[0][pick]), int(iu0[1][pick])
            dij = Xbest[i] - Xbest[j]
            nd = np.linalg.norm(dij)
            if nd < 1e-12:
                dij = rng.standard_normal(size=d); nd = np.linalg.norm(dij)
            dij /= nd
            Xt[i] += dij * 0.03 + rng.standard_normal(size=d) * 0.01
            Xt[j] -= dij * 0.03 + rng.standard_normal(size=d) * 0.01
            # small global noise
            Xt += rng.standard_normal(size=Xt.shape) * 0.004
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, e0, e1 = 600, 0.02, 0.002, 0.05, 0.008
        # anneal
        Xt = smooth_anneal(Xt, steps, T0, T1, e0, e1)
        v = max_cos(Xt)
        if v < best - 1e-12:
            best = v
            Xbest = Xt
        # timeout check
        if _time.time() - t0 > TIME_LIMIT - 28:
            break
        # stop if many trials
        if trial > 120:
            break

    remain = TIME_LIMIT - (_time.time() - t0)
    remain = max(5.0, min(28.0, remain - 2.0))
    try:
        Xbest = lp_polish(Xbest, remain)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

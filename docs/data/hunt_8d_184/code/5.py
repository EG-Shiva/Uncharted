import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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


def _e8():
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(8)
                    v[i] = s1
                    v[j] = s2
                    pts.append(v)
    for mask in range(256):
        v = np.empty(8)
        nminus = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -0.5
                nminus += 1
            else:
                v[k] = 0.5
        if nminus % 2 == 0:
            pts.append(v.copy())
    pts = np.array(pts) / np.sqrt(2.0)
    return pts


def _smooth_anneal(x, iters, beta0, beta1, lr0, lr1, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lbeta0 = np.log(beta0)
    lbeta1 = np.log(beta1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for t in range(1, iters + 1):
        frac = (t - 1) / (iters - 1) if iters > 1 else 1.0
        beta = float(np.exp(lbeta0 + (lbeta1 - lbeta0) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        s = float(E.sum())
        if not np.isfinite(s) or s == 0:
            continue
        W = E / s
        Gr = W @ x
        Gt = Gr - (np.sum(Gr * x, axis=1, keepdims=True)) * x
        rms = float(np.sqrt(np.mean(Gt * Gt))) + 1e-12
        Gn = Gt / rms
        v = mu * v + (1.0 - mu) * Gn
        denom = 1.0 - mu ** t
        vh = v / denom if denom > 1e-12 else v
        x = x - lr * vh
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    rng = np.random.default_rng(20261103)
    t0 = time.time()
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=np.float64))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=np.float64))
    if not cands:
        c = rng.normal(size=(184, 8))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        return c
    best = None
    best_val = 2.0
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    e8 = _e8()
    n, d = best.shape
    # fixed budget of trials, time-guarded
    MAX_TRIALS = 80
    for trial in range(MAX_TRIALS):
        if time.time() - t0 > 132:
            break
        y = best.copy()
        from_e8 = (trial % 5 == 4)
        if from_e8:
            idx = rng.choice(e8.shape[0], n, replace=False)
            y = e8[idx].copy()
            sig = float(rng.uniform(0.01, 0.04))
            y = y + rng.normal(size=y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            G = y @ y.T
            np.fill_diagonal(G, -np.inf)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            r = float(rng.random())
            if r < 0.6:
                nrep = 1
            elif r < 0.85:
                nrep = 2
            else:
                nrep = 3
            if float(rng.random()) < 0.75:
                ncand = 3000
                cand_pool = rng.normal(size=(ncand, 8))
                cand_pool /= np.linalg.norm(cand_pool, axis=1, keepdims=True)
                for k in range(nrep):
                    idxr = int(order[k])
                    S = cand_pool @ y.T
                    S[:, idxr] = -np.inf
                    # also ignore other to-be-replaced not yet replaced: keep simple
                    scores = S.max(axis=1)
                    bi = int(np.argmin(scores))
                    y[idxr] = cand_pool[bi]
            sig = float(rng.choice(np.array([0.008, 0.015, 0.03, 0.06])))
            y = y + rng.normal(size=y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        beta0 = float(rng.uniform(15, 35))
        beta1 = float(rng.uniform(250, 500))
        iters = int(rng.integers(1300, 2200))
        lr0 = float(rng.uniform(0.006, 0.015))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        mu = float(rng.uniform(0.85, 0.93))
        y = _smooth_anneal(y, iters, beta0, beta1, lr0, lr1, mu)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best = y.copy()
    # final trust-region LP polish with remaining time
    rem = 172 - (time.time() - t0)
    if rem > 5:
        rem = min(rem, 32)
        best = _lp_polish(best, rem)
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

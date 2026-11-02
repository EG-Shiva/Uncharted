import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    return np.nan_to_num(x)

def _maxcos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(y, steps, T0=0.02, T1=0.002, lr=0.008):
    n, d = y.shape
    m1 = np.zeros((n, d))
    v1 = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        gm = G.max()
        W = np.exp((G - gm) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5 + 1e-30
        grad = (W @ y) / S
        # tangent projection
        grad = grad - (np.sum(grad * y, axis=1, keepdims=True)) * y
        t += 1
        m1 = b1 * m1 + (1 - b1) * grad
        v1 = b2 * v1 + (1 - b2) * (grad * grad)
        mhat = m1 / (1 - b1 ** t)
        vhat = v1 / (1 - b2 ** t)
        step = mhat / (np.sqrt(vhat) + eps)
        eff = lr * (1.0 - 0.5 * frac)
        y = y - eff * step
        y = _proj(y)
    return y

def _perturb(x, rng, sigma):
    n, d = x.shape
    N = rng.normal(size=(n, d))
    N -= (np.sum(N * x, axis=1, keepdims=True)) * x
    y = x + sigma * N
    return _proj(y)

def _reinsert_one(y, rng, R=400):
    n, d = y.shape
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    idx = int(np.argmax(G.max(axis=1)))
    cand = rng.normal(size=(R, d))
    cand = _proj(cand)
    others = np.delete(y, idx, axis=0)  # (n-1,d)
    dots = cand @ others.T  # R x (n-1)
    best = int(np.argmin(dots.max(axis=1)))
    y2 = y.copy()
    y2[idx] = cand[best]
    return y2

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _maxcos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > 2500:
            k = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[k], J[k]
            csv = c[sel][k]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    try:
        base = np.load("best.npy")
    except Exception:
        base = np.load("record.npy")
    base = _proj(np.array(base, dtype=float))
    # ensure shape (54,21)
    if base.shape != (54, 21):
        # try to reshape/crop defensively
        tmp = np.zeros((54, 21))
        m = min(base.shape[0], 54)
        k = min(base.shape[1], 21)
        tmp[:m, :k] = base[:m, :k]
        # fill rest random
        if m < 54:
            tmp[m:] = _proj(rng.normal(size=(54 - m, 21)))
        base = _proj(tmp)
    best = base.copy()
    best_val = _maxcos(best)
    sigmas = [0.04, 0.10, 0.22, 0.35, 0.07, 0.15, 0.28, 0.03]
    trial = 0
    # fast basin hopping, reserve ~40s for final LP
    while time.time() - T0 < LIMIT - 45:
        trial += 1
        sig = sigmas[(trial - 1) % len(sigmas)]
        y = _perturb(best, rng, sig)
        # main quench
        y = _quench(y, steps=700, T0=0.02, T1=0.0025, lr=0.008)
        # occasional reinsert + short re-quench to change topology
        if trial % 2 == 0:
            y = _reinsert_one(y, rng, R=300)
            y = _quench(y, steps=400, T0=0.008, T1=0.0015, lr=0.006)
        v = _maxcos(y)
        if v < best_val:
            best, best_val = y, v
        # also try perturbing original base sometimes for diversity
        if trial % 7 == 0 and time.time() - T0 < LIMIT - 60:
            y2 = _perturb(base, rng, 0.3)
            y2 = _quench(y2, steps=700, T0=0.025, T1=0.002, lr=0.009)
            v2 = _maxcos(y2)
            if v2 < best_val:
                best, best_val = y2, v2
        if trial > 60:
            break
        if time.time() - T0 > LIMIT - 50:
            break
    remain = LIMIT - (time.time() - T0) - 2.0
    if remain > 5:
        best = _lp_polish(best, min(remain, 40.0))
    best = _proj(best)
    return best

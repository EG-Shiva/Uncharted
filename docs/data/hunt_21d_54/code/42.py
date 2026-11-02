import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    x = np.nan_to_num(x)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _quench(y, steps, T0, T1, lr0, mu, rng, noise0=0.0):
    n, d = y.shape
    vel = np.zeros((n, d))
    lT0 = np.log(max(T0, 1e-12))
    lT1 = np.log(max(T1, 1e-12))
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = lr0 * (1.0 - 0.6 * frac)
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ y) / rs
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        # clip huge grads
        vel = mu * vel + grad
        y = y - lr * vel
        if noise0 > 0:
            ns = noise0 * (1 - frac)
            nz = rng.normal(size=(n, d))
            nz -= (nz * y).sum(axis=1, keepdims=True) * y
            y = y + ns * nz
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
    return y

def _void_replace(y, rng, K=2000):
    n, d = y.shape
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    iw = int(np.argmax(rowmax))
    # sometimes second worst for diversity
    if rng.random() < 0.3:
        order = np.argsort(-rowmax)
        iw = int(order[int(rng.integers(0, min(5, n)))])
    C = rng.normal(size=(K, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    # max dot with all except iw
    Mask = np.ones(n, dtype=bool); Mask[iw] = False
    Ym = y[Mask]
    D = C @ Ym.T
    scores = D.max(axis=1)
    ib = int(np.argmin(scores))
    y2 = y.copy()
    y2[iw] = C[ib]
    return y2

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 30:
        it += 1
        if time.time() - start > seconds - 0.5:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            idx = np.argpartition(-c[sel], 1500)[:1500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 1500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
    n, d = 54, 21
    try:
        b0 = _proj(np.load("best.npy").astype(float))
    except Exception:
        b0 = _proj(rng.normal(size=(n, d)))
    if b0.shape != (n, d):
        b0 = _proj(rng.normal(size=(n, d)))
    try:
        r0 = _proj(np.load("record.npy").astype(float))
    except Exception:
        r0 = b0.copy()
    bv = _mc(b0); rv = _mc(r0)
    best = b0.copy() if bv <= rv else r0.copy()
    bv = min(bv, rv)

    def left():
        return LIMIT - (time.time() - T0)

    # basin hopping with cheap quenches; keep LP out of the loop
    it = 0
    # precompute quench budgets
    while left() > 22:
        it += 1
        u = rng.random()
        if it <= 3:
            # early: fresh random quenches for diversity
            y = _proj(rng.normal(size=(n, d)))
        elif u < 0.55:
            y = _proj(rng.normal(size=(n, d)))
        elif u < 0.80:
            # large global shake of incumbent (collective topology break)
            s = float(rng.uniform(0.15, 0.6))
            y = _proj(best + s * rng.normal(size=(n, d)))
        else:
            s = float(rng.uniform(0.05, 0.25))
            y = _proj(best + s * rng.normal(size=(n, d)))
            # void reinsertion on top
            if rng.random() < 0.7:
                y = _void_replace(y, rng, K=2000)
        # choose quench schedule by iteration
        if left() < 40:
            steps1, steps2 = 1200, 800
        else:
            steps1, steps2 = 1600, 900
        # stage 1: hot break
        T0a = float(rng.uniform(0.015, 0.03)) if (it % 2 == 0) else 0.02
        y = _quench(y, steps=steps1, T0=T0a, T1=0.003, lr0=0.30, mu=0.85, rng=rng, noise0=0.002 if (it % 3 == 0) else 0.0)
        if left() < 20:
            v = _mc(y)
            if v < bv:
                best, bv = y.copy(), v
            break
        # stage 2: cold converge
        y = _quench(y, steps=steps2, T0=0.003, T1=0.0006, lr0=0.22, mu=0.88, rng=rng, noise0=0.0)
        v = _mc(y)
        if v < bv:
            best, bv = y.copy(), v
            # opportunistic void + short re-quench on improvement
            if left() > 30 and rng.random() < 0.5:
                y2 = _void_replace(best, rng, K=1500)
                y2 = _quench(y2, steps=800, T0=0.004, T1=0.0006, lr0=0.22, mu=0.88, rng=rng)
                v2 = _mc(y2)
                if v2 < bv:
                    best, bv = y2.copy(), v2
        # safety: time check
        if left() <= 22:
            break

    # short final LP squeeze on best (bounded)
    if left() > 5:
        try:
            polished = _lp_polish(best, min(left() - 1.0, 11.0))
            vp = _mc(polished)
            if vp < bv:
                best, bv = polished, vp
            # if LP failed to improve, keep quench best
        except Exception:
            pass
    best = _proj(best)
    return best

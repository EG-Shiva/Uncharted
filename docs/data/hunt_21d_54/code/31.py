import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    return np.nan_to_num(x)

def _quench(y, steps, lr0, T0, T1, mu=0.85):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1.0 - frac) + T1 * frac
        if T < 1e-4:
            T = 1e-4
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * vel
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > 2200:
            idx = np.argpartition(-c[sel], 2200)[:2200]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2200
        else:
            csv = c[sel]
        if p == 0:
            break
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
                          method="highs",
                          options={"time_limit": 2.0, "presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 162.0
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            rv = _max_cos(r0)
        else:
            rv = 1e9
    except Exception:
        rv = 1e9
        r0 = b0
    best = b0.copy()
    bv = _max_cos(best)
    if rv < bv:
        best, bv = r0.copy(), rv
    n, d = 54, 21

    def tleft():
        return LIMIT - (time.time() - T0)

    # fast initial quench to ensure local convergence without LP
    if tleft() > 40:
        cand = _quench(best.copy(), steps=1500, lr0=0.25, T0=0.015, T1=0.001)
        v = _max_cos(cand)
        if v < bv:
            best, bv = cand, v

    strengths = [0.01, 0.03, 0.07, 0.15, 0.30, 0.05, 0.12, 0.02]
    kicks = [3, 6, 10, 5, 8, 4]
    it = 0
    # reserve ~14s for final LP
    while tleft() > 22:
        it += 1
        base = best
        mode = it % 3
        y = base.copy()
        if mode == 0:
            # worst-point reinsertion: best of many random
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            iw = int(order[int(rng.integers(0, 4))])
            K = 1200
            C = rng.normal(size=(K, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            D = C @ base.T
            # ignore self column by setting it low
            D[:, iw] = -2.0
            worst = D.max(axis=1)
            ib = int(np.argmin(worst))
            y[iw] = C[ib]
            s = strengths[(it - 1) % len(strengths)]
            # slight jitter others tangent
            noise = rng.normal(size=(n, d)) * (s * 0.15)
            noise -= (noise * y).sum(axis=1, keepdims=True) * y
            y = y + noise
            y = _proj(y)
            steps = 1400
            lr0 = 0.3
            TT0, TT1 = 0.012, 0.0008
        else:
            s = strengths[(it - 1) % len(strengths)]
            nk = kicks[(it - 1) % len(kicks)]
            # pick nk targets: mix worst + random
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            targets = set(order[:2].tolist())
            while len(targets) < nk:
                targets.add(int(rng.integers(0, n)))
            for i in list(targets):
                v = rng.normal(size=d)
                v -= (v @ y[i]) * y[i]
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                v /= nv
                ang = s * (0.5 + rng.random())
                y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
            noise = rng.normal(size=(n, d)) * (s * 0.1)
            noise -= (noise * y).sum(axis=1, keepdims=True) * y
            y = y + noise
            y = _proj(y)
            steps = 1200
            lr0 = 0.30
            TT0, TT1 = 0.015, 0.001
        y = _quench(y, steps=steps, lr0=lr0, T0=TT0, T1=TT1, mu=0.85)
        v = _max_cos(y)
        if v < bv - 1e-12:
            best, bv = y, v
        # safety: abort if almost out of time
        if tleft() < 23:
            break

    if tleft() > 4:
        budget = min(12.0, tleft() - 1.0)
        if budget > 2:
            best = _lp_polish(best, budget)
    best = _proj(best)
    return best

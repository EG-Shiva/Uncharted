import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    return np.nan_to_num(x)

def _quench(y, steps, lr0, t0, t1, mu=0.9):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = t0 * (1 - frac) + t1 * frac
        if T < 1e-12:
            T = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -15, 0, out=E)
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

def _lp_polish(x, seconds, t_end):
    x = _proj(x)
    best = _max_cos(x)
    n, d = x.shape
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
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
        if time.time() >= t_end:
            break
    return x

def solve():
    T0 = time.time()
    HARD = 165.0  # hard internal deadline
    def tl():
        return HARD - (time.time() - T0)
    rng = np.random.default_rng(0)
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
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0
    except Exception:
        pass
    n, d = 54, 21
    best = b0.copy()
    bv = _max_cos(best)
    # parameters
    strengths = [0.02, 0.05, 0.12, 0.25, 0.03, 0.08, 0.18, 0.01, 0.06, 0.15]
    it = 0
    # reserve ~18s for final LP
    while tl() > 22 and it < 500:
        it += 1
        s = strengths[(it - 1) % len(strengths)]
        do_reinsert = (it % 4 == 0)
        if do_reinsert:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            pick = int(rng.integers(0, 3))
            iw = int(order[pick])
            y = best.copy()
            K = 1200
            C = rng.normal(size=(K, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            mask = np.ones(n, dtype=bool)
            mask[iw] = False
            D = C @ y[mask].T
            scores = D.max(axis=1)
            ib = int(np.argmin(scores))
            y[iw] = C[ib]
            # small tangent jitter of others
            nz = rng.normal(size=(n, d)) * (s * 0.15)
            nz -= (nz * y).sum(axis=1, keepdims=True) * y
            y = _proj(y + nz)
            y = _quench(y, steps=500, lr0=0.30, t0=0.012, t1=0.0012, mu=0.88)
        else:
            y = best.copy()
            nz = rng.normal(size=(n, d))
            nz -= (nz * y).sum(axis=1, keepdims=True) * y
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            ang = s * (0.5 + rng.random(n)[:, None])
            # rotate each point a bit: cos*y + sin*nz approx, then also raw add
            y = y + nz * ang
            y = _proj(y)
            y = _quench(y, steps=600, lr0=0.32, t0=0.010, t1=0.0008, mu=0.9)
        v = _max_cos(y)
        if v < bv:
            best, bv = y, v
        if tl() <= 22:
            break
    # final squeeze: quench best once more briefly then LP
    if tl() > 6:
        # one medium quench
        if tl() > 25:
            c = _quench(best.copy(), steps=800, lr0=0.25, t0=0.006, t1=0.0005, mu=0.9)
            cv = _max_cos(c)
            if cv < bv:
                best, bv = c, cv
        t_end = time.time() + min(16.0, tl() - 2.0)
        if time.time() < t_end:
            c2 = _lp_polish(best, 60, t_end)
            v2 = _max_cos(c2)
            if v2 < bv:
                best, bv = c2, v2
    best = _proj(best)
    return best

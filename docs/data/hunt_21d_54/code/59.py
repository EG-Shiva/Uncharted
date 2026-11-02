import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _descend(y, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / max(T, 1e-12)
        np.clip(E, -15.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        scale = float(W.sum()) / n + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= scale
        vel = mu * vel + grad
        y = y - lr * vel
        # renormalize
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
    return y

def _lp_polish(x, seconds, r0=2e-3):
    x = _proj(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 900:
            idx = np.argsort(c)[-900:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        csv = c[sel]
        p = len(I)
        nv = n * d + 1
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
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
    t0 = time.time()
    LIMIT = 175.0
    rng = np.random.RandomState(12345)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = None
    try:
        r0 = np.load("record.npy")
    except Exception:
        r0 = None
    cands = []
    if b0 is not None:
        try:
            b0 = _proj(np.asarray(b0, dtype=float))
            if b0.shape == (54, 21):
                cands.append(b0)
        except Exception:
            pass
    if r0 is not None:
        try:
            r0 = _proj(np.asarray(r0, dtype=float))
            if r0.shape == (54, 21):
                cands.append(r0)
        except Exception:
            pass
    if not cands:
        cands.append(_proj(rng.randn(54, 21)))
    best = min(cands, key=_max_cos).copy()
    bv = _max_cos(best)
    n, d = 54, 21
    sigmas = [0.05, 0.02, 0.01, 0.08, 0.03, 0.005, 0.12, 0.04]

    def rowmax(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    trial = 0
    # hopping phase: no LP, only fast quenches
    while time.time() - t0 < LIMIT - 30.0:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 4
        cand = best.copy()
        if mode == 0:
            cand = cand + sigma * rng.randn(n, d)
            cand = _proj(cand)
            cand = _descend(cand, steps=1400, T0=0.018, T1=0.0012, lr0=0.08, lr1=0.012, mu=0.85)
        elif mode == 1:
            rm, _ = rowmax(best)
            order = np.argsort(-rm)[:3]
            cand[order] = cand[order] + sigma * 2.0 * rng.randn(len(order), d)
            cand = cand + (sigma * 0.2) * rng.randn(n, d)
            cand = _proj(cand)
            cand = _descend(cand, steps=1200, T0=0.012, T1=0.0008, lr0=0.06, lr1=0.01, mu=0.86)
        elif mode == 2:
            rm, _ = rowmax(best)
            idx = int(np.argmax(rm))
            if trial % 8 == 0:
                order = np.argsort(-rm)
                idx = int(order[int(rng.randint(0, 5))])
            keep = np.delete(best, idx, axis=0)
            K = 4000
            S = rng.randn(K, d)
            S = _proj(S)
            dots = S @ keep.T
            md = dots.max(axis=1)
            hole = S[int(np.argmin(md))]
            cand[idx] = hole + 0.005 * rng.randn(d)
            # tiny shake others to open hole
            cand = cand + (sigma * 0.15) * rng.randn(n, d)
            cand = _proj(cand)
            cand = _descend(cand, steps=1200, T0=0.010, T1=0.0007, lr0=0.05, lr1=0.008, mu=0.85)
        else:
            k = 8
            sel = rng.choice(n, k, replace=False)
            cand[sel] = cand[sel] + sigma * 1.5 * rng.randn(k, d)
            cand = _proj(cand)
            T0 = 0.015 if (trial % 2 == 0) else 0.025
            cand = _descend(cand, steps=1300, T0=T0, T1=0.001, lr0=0.07, lr1=0.01, mu=0.84)
        v = _max_cos(cand)
        if v < bv - 1e-12:
            bv = v
            best = cand.copy()
        # safety: stop if very little time for final polish
        if time.time() - t0 > LIMIT - 31.0:
            break

    remain = LIMIT - (time.time() - t0)
    if remain > 3.0:
        best = _lp_polish(best, seconds=max(3.0, remain - 1.0), r0=2e-3)
    best = _proj(best)
    return best

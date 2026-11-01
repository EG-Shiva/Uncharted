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
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = iu_c > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(iu_c[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_opt(X, steps, K0, K1, lr0):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lK0 = np.log(K0)
    lK1 = np.log(K1)
    for s in range(steps):
        frac = s / max(1, steps - 1) if steps > 1 else 0.0
        K = float(np.exp(lK0 + (lK1 - lK0) * frac))
        lr = lr0 * (0.2 + 0.8 * 0.5 * (1.0 + np.cos(np.pi * frac)))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(K * (G - m))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        m1h = m1 / c1
        m2h = m2 / c2
        step = m1h / (np.sqrt(m2h) + eps)
        X = X - lr * step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _find_holes(Xref, k, rng, n_samples=30000):
    C = rng.normal(size=(n_samples, 5))
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ Xref.T
    s_min = D.max(axis=1)
    order = np.argsort(s_min)
    chosen = []
    for idx in order:
        c = C[idx]
        if len(chosen) == 0:
            chosen.append(c)
        else:
            Ch = np.stack(chosen)
            if np.all(Ch @ c < 0.90):
                chosen.append(c)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for idx in order:
            if len(chosen) >= k:
                break
            chosen.append(C[idx])
    return np.stack(chosen[:k])

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(an)):
                    v = _max_cos(an)
                    if v < best_val:
                        best_val = v
                        best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    def row_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 8.0:
            break
        # allocate time for this trial
        lp_t = 1.0
        if rem < 25:
            lp_t = max(0.5, rem - 7.0)
            lp_t = min(lp_t, 0.8)
        steps = 1200
        if rem < 25:
            steps = 600
        mode = trial % 7
        y = best.copy()
        if mode == 0:
            y += 0.02 * rng.normal(size=(n, d))
            sm_steps, K0, K1, lr0 = steps, 12.0, 250.0, 0.03
        elif mode == 1:
            k = 2
            rm, G = row_max(best)
            victims = np.argsort(-rm)[:k]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            holes = _find_holes(best[keep], k, rng)
            y[victims] = holes + 0.01 * rng.normal(size=(k, d))
            y += 0.005 * rng.normal(size=(n, d))
            sm_steps, K0, K1, lr0 = steps, 12.0, 300.0, 0.03
        elif mode == 2:
            k = 4
            rm, G = row_max(best)
            victims = np.argsort(-rm)[:k]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            holes = _find_holes(best[keep], k, rng)
            y[victims] = holes + 0.015 * rng.normal(size=(k, d))
            y += 0.005 * rng.normal(size=(n, d))
            sm_steps, K0, K1, lr0 = steps, 10.0, 300.0, 0.035
        elif mode == 3:
            y = rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            sm_steps, K0, K1, lr0 = 1500 if rem > 30 else steps, 8.0, 250.0, 0.04
        elif mode == 4:
            subset = rng.choice(n, 25, replace=False)
            y[subset] += 0.06 * rng.normal(size=(len(subset), d))
            sm_steps, K0, K1, lr0 = 800 if rem > 30 else 500, 15.0, 250.0, 0.025
        elif mode == 5:
            rm, G = row_max(best)
            # high contact count via threshold
            m = float(G.max())
            cnt = (G >= m - 1e-9).sum(axis=1)
            order = np.argsort(-cnt - 0.5 * rm)
            k = 3
            victims = order[:k]
            # half holes half random
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            holes = _find_holes(best[keep], k, rng, n_samples=20000)
            y[victims] = holes + 0.01 * rng.normal(size=(k, d))
            # also shake neighbours of victims
            neigh = np.where(G[victims[0]] > m - 0.03)[0]
            if len(neigh) > 0:
                y[neigh] += 0.03 * rng.normal(size=(len(neigh), d))
            sm_steps, K0, K1, lr0 = steps, 12.0, 300.0, 0.03
        else:
            y += 0.07 * rng.normal(size=(n, d))
            sm_steps, K0, K1, lr0 = 900 if rem > 30 else 500, 10.0, 200.0, 0.035
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        # smooth stage
        try:
            y = _smooth_opt(y, sm_steps, K0, K1, lr0)
        except Exception:
            pass
        # LP stage
        pt = min(lp_t, rem - 6.5)
        if pt < 0.4:
            break
        try:
            y = _lp_polish(y, pt)
        except Exception:
            pass
        try:
            v = _max_cos(y)
        except Exception:
            trial += 1
            continue
        if v < best_val and np.isfinite(v):
            best = y
            best_val = v
        trial += 1

    rem = deadline - time.time()
    if rem > 1.5:
        try:
            best = _lp_polish(best, rem - 0.7)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

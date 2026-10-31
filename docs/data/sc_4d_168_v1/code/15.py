import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
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

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 136.0
    rng = np.random.RandomState(12345)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 168, 4
    if xb.shape != (168, 4):
        xb = rng.randn(168, 4)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    def refine(x_init, steps, mu=0.9, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def opt_hole(X_keep, n_sample=6000, iters=120):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        s = S[np.argmin(m)].copy()
        # local optimisation of single point: minimise LSE
        lr0, lr1 = 0.08, 0.005
        b0, b1 = 30.0, 250.0
        for it in range(iters):
            f = it / max(1, iters - 1)
            beta = b0 * (b1 / b0) ** f
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = X_keep @ s
            cm = float(np.max(dots))
            w = np.exp(np.clip(beta * (dots - cm), -20.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ X_keep
            gt = g - np.dot(g, s) * s
            s = s - lr * gt
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4)
                ns = np.linalg.norm(s)
            s /= ns
        return s

    def sample_hole(X_keep, n_sample=8000):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        return S[np.argmin(m)]

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.5:
            break
        x = x_best.copy()
        typ = trial % 8
        if typ == 0:
            x = x + 0.018 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 650
        elif typ == 1:
            x = x + 0.060 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 900
        elif typ == 2:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.008, axis=1)
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 12) % n]
            nbr = np.argsort(-G[i0], kind="stable")[:8]
            x[nbr] = x[nbr] + 0.15 * rng.randn(len(nbr), d)
            x[i0] = x[i0] + 0.15 * rng.randn(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 950
        elif typ == 3:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.010, axis=1)
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 10) % n]
            k = 4 + (trial % 3)  # 4..6
            nbrs = np.argsort(-G[i0], kind="stable")[:k]
            cluster = np.concatenate([[i0], nbrs])
            keep = np.ones(n, dtype=bool)
            keep[cluster] = False
            cur = x[keep]
            new_pts = []
            for _ in range(len(cluster)):
                h = opt_hole(cur, n_sample=5000, iters=100)
                new_pts.append(h)
                cur = np.vstack([cur, h[None, :]])
            x[cluster] = np.array(new_pts)
            steps = 1050
        elif typ == 4:
            idx = rng.choice(n, 20, replace=False)
            x[idx] = x[idx] + 0.20 * rng.randn(len(idx), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 850
        elif typ == 5:
            if trial % 16 == 5:
                x = rng.randn(n, d)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                steps = 1700
            else:
                x = x + 0.040 * rng.randn(n, d)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                steps = 750
        elif typ == 6:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 1e-4, axis=1)
            loose = np.where(deg <= 4)[0]
            if len(loose) == 0:
                loose = np.argsort(deg)[:4]
            # move 2 loose points to holes
            nl = min(2, len(loose))
            picks = rng.choice(loose, nl, replace=False)
            for li in picks:
                X_keep = np.delete(x, li, axis=0)
                h = opt_hole(X_keep, n_sample=5000, iters=120)
                x[li] = h
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            # jitter neighbours of moved points
            for li in picks:
                nbr = np.argsort(-G2[li], kind="stable")[:5]
                x[nbr] = x[nbr] + 0.04 * rng.randn(len(nbr), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 900
        else:
            # contact-edge break: pick tight pairs, push apart
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            # find tight pair indices
            iu0 = np.triu_indices(n, 1)
            c = G[iu0]
            order = np.argsort(-c, kind="stable")
            npair = 3 + (trial % 3)
            for k in range(npair):
                idx = order[(rng.randint(0, 60) + k) % len(c)]
                i, j = iu0[0][idx], iu0[1][idx]
                diff = x[i] - x[j]
                diff -= np.dot(diff, x[i]) * x[i]
                nd = np.linalg.norm(diff)
                if nd > 1e-12:
                    diff /= nd
                    x[i] = x[i] + 0.06 * diff
                    x[j] = x[j] - 0.06 * diff
            # fill largest hole with small jitter of random point
            ri = rng.randint(0, n)
            X_keep = np.delete(x, ri, axis=0)
            h = sample_hole(X_keep, n_sample=6000)
            x[ri] = 0.85 * x[ri] + 0.15 * h
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 950

        x = refine(x, steps)
        v = _max_cos(x)
        if v < best_val - 2e-12:
            # opportunistic quick LP squeeze if promising
            if v < best_val - 1e-6 and time.time() < SEARCH_END - 4:
                try:
                    xq = _lp_polish(x, min(time.time() + 1.2, SEARCH_END))
                    vq = _max_cos(xq)
                    if vq < v:
                        x, v = xq, vq
                except Exception:
                    pass
            if v < best_val:
                best_val = v
                x_best = x.copy()

    # final tighten + polish
    try:
        if time.time() < HARD_END - 3:
            x_best = refine(x_best, 350, beta0=150.0, beta1=900.0, lr0=0.004, lr1=0.0003)
            v = _max_cos(x_best)
            if v < best_val:
                best_val = v
    except Exception:
        pass
    remain = min(HARD_END - 1.5, time.time() + 26.0)
    if remain > time.time() + 1.0:
        try:
            x_best = _lp_polish(x_best, remain)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

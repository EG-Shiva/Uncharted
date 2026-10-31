import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, deadline):
    # trust-region LP from parent, time-bounded by deadline (absolute time)
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
    SEARCH_END = t0 + 138.0
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # load starting point
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(168, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = xb.shape
    best_val = _max_cos(xb)
    x_best = xb.copy()

    def refine(x_init, steps, mu=0.9, beta0=30.0, beta1=400.0,
               lr0=0.012, lr1=0.0008):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if time.time() >= SEARCH_END and steps > 200:
                # allow early abort of long refinement
                if s > 200:
                    break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            # stable weights exp(beta*(G-cmax))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            # momentum (scale-preserving)
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def best_hole(X_keep, n_sample=12000):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        return S[np.argmin(m)]

    trial = 0
    # quick evaluation helper
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 2:
            break
        x = x_best.copy()
        typ = trial % 7

        if typ == 0:  # small global shake
            x = x + 0.020 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 700
        elif typ == 1:  # medium global shake
            x = x + 0.070 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 1000
        elif typ == 2:  # local shake of jammed cluster
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.008, axis=1)
            order = np.argsort(-deg, kind="stable")
            k = rng.randint(0, 12)
            i0 = order[k % n]
            nbr = np.argsort(-G[i0], kind="stable")[:9]
            x[nbr] = x[nbr] + 0.16 * rng.randn(len(nbr), d)
            x[i0] = x[i0] + 0.16 * rng.randn(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 1000
        elif typ == 3:  # cluster delete + hole reinsert (topology change)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.010, axis=1)
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 10) % n]
            nbrs = np.argsort(-G[i0], kind="stable")[:4]
            cluster = np.concatenate([[i0], nbrs])
            keep = np.ones(n, dtype=bool)
            keep[cluster] = False
            cur = x[keep]
            new_pts = []
            for _ in range(len(cluster)):
                h = best_hole(cur, n_sample=12000)
                new_pts.append(h)
                cur = np.vstack([cur, h[None, :]])
            x[cluster] = np.array(new_pts)
            steps = 1100
        elif typ == 4:  # random subset shake
            idx = rng.choice(n, 18, replace=False)
            x[idx] = x[idx] + 0.22 * rng.randn(len(idx), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 900
        elif typ == 5:  # fresh random restart (occasional)
            if trial % 14 == 5:
                x = rng.randn(n, d)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                steps = 1800
            else:
                x = x + 0.045 * rng.randn(n, d)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                steps = 800
        else:  # rattler / loose-point to hole + neighbour jitter
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 1e-4, axis=1)
            loose = np.where(deg <= 4)[0]
            if len(loose) == 0:
                loose = np.argsort(deg)[:3]
            li = loose[rng.randint(0, len(loose))]
            # move loose point to largest hole
            X_keep = np.delete(x, li, axis=0)
            h = best_hole(X_keep, n_sample=15000)
            x[li] = h
            # jitter its new neighbours slightly
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            nbr = np.argsort(-G2[li], kind="stable")[:5]
            x[nbr] = x[nbr] + 0.05 * rng.randn(len(nbr), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 900

        x = refine(x, steps)
        v = _max_cos(x)
        if v < best_val - 5e-13:
            best_val = v
            x_best = x.copy()

    # final polish with remaining time
    remain_deadline = min(HARD_END - 2.0, time.time() + 24.0)
    if remain_deadline > time.time() + 1.0:
        # one more short anneal at high beta to tighten before LP
        # cheap: 300 steps high beta
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        x_best = refine(x_best, 350, beta0=150.0, beta1=800.0,
                        lr0=0.004, lr1=0.0003)
        v = _max_cos(x_best)
        if v < best_val:
            best_val = v
        x_best = _lp_polish(x_best, remain_deadline)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

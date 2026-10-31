import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 142.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (168, 4):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if len(cands) == 0:
        x_best = rng.randn(168, 4)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        x_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(x_best)
    n, d = 168, 4

    def refine(x_init, steps, mu=0.9, beta0=30.0, beta1=400.0, lr0=0.012, lr1=0.0008):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if s > 200 and time.time() >= SEARCH_END:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def greedy_hole(X_keep):
        # best of random candidates maximizing min distance (min cosine)
        S = rng.randn(9000, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        return S[int(np.argmin(m))]

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.5:
            break
        x = x_best.copy()
        typ = trial % 9
        if typ == 0:
            x = x + 0.020 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 700, dict(beta0=30., beta1=350., lr0=0.012, lr1=0.0008)
        elif typ == 1:
            x = x + 0.070 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1000, dict(beta0=25., beta1=400., lr0=0.012, lr1=0.0007)
        elif typ == 2:
            x = x + 0.22 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            x[idx] = rng.randn(k, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1400, dict(beta0=20., beta1=450., lr0=0.014, lr1=0.0006)
        elif typ == 3:
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.008, axis=1)
            order = np.argsort(-deg, kind="stable")
            k = int(rng.randint(0, 12))
            i0 = order[k % n]
            nbr = np.argsort(-G[i0], kind="stable")[:9]
            x[nbr] += 0.16 * rng.randn(len(nbr), d)
            x[i0] += 0.16 * rng.randn(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1100, dict(beta0=30., beta1=380., lr0=0.011, lr1=0.0007)
        elif typ == 4:
            # jammed-cluster deletion + greedy reinsert
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.005, axis=1)
            k = 4 + (trial % 5)  # 4..8
            idx = np.argsort(-deg, kind="stable")[:k]
            # sequential greedy reinsert
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                Xk = x[mask]
                x[j] = greedy_hole(Xk)
            x = x + 0.02 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1300, dict(beta0=25., beta1=420., lr0=0.012, lr1=0.0006)
        elif typ == 5:
            # rattler / low-contact relocation
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.005, axis=1)
            idx_lo = np.argsort(deg, kind="stable")[:4]
            for j in idx_lo:
                mask = np.ones(n, dtype=bool); mask[j] = False
                x[j] = greedy_hole(x[mask])
            # also shake one jammed point
            idx_hi = np.argsort(-deg, kind="stable")[0]
            x[idx_hi] += 0.10 * rng.randn(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1200, dict(beta0=30., beta1=400., lr0=0.010, lr1=0.0007)
        elif typ == 6:
            x = rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1800, dict(beta0=20., beta1=500., lr0=0.015, lr1=0.0006)
        elif typ == 7:
            x = x + 0.035 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # reinsert 2 random points at holes
            idx = rng.choice(n, size=2, replace=False)
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                x[j] = greedy_hole(x[mask])
            steps, kw = 1100, dict(beta0=35., beta1=350., lr0=0.012, lr1=0.0008)
        else:
            x = x + 0.10 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, kw = 1200, dict(beta0=22., beta1=380., lr0=0.013, lr1=0.0007)

        try:
            xo = refine(x, steps, **kw)
        except Exception:
            continue
        v = _max_cos(xo)
        improved = False
        if v < best_val:
            best_val = v
            x_best = xo.copy()
            improved = True
        # LP polish promising basins
        do_lp = False
        lp_budget = 0.0
        if improved:
            do_lp = True; lp_budget = 2.0
        elif v < best_val + 0.002 and (trial % 4 == 0):
            do_lp = True; lp_budget = 1.0
        elif v < best_val + 0.006 and (trial % 12 == 0):
            do_lp = True; lp_budget = 0.8
        if do_lp and time.time() < SEARCH_END - 0.5:
            try:
                xp = _lp_polish(xo, min(time.time() + lp_budget, SEARCH_END + 4.0, HARD_END - 6.0))
                vp = _max_cos(xp)
                if vp < best_val:
                    best_val = vp
                    x_best = xp.copy()
            except Exception:
                pass
    # final polish
    rem_end = min(HARD_END - 1.0, time.time() + max(5.0, HARD_END - time.time() - 1.0))
    if time.time() < HARD_END - 2.0:
        try:
            x_best = _lp_polish(x_best, rem_end)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

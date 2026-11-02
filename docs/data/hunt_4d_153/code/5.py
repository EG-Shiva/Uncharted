import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        # max over off-diag
        np.fill_diagonal(G, -2.0)
        # G now has -2 diag
        c_full = G[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        # need original cos for b_ub: c[sel] is cosine
        c_sel = c_full[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d+k, J*d+k, last
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = cols_I[0::d] if False else 0  # placeholder, overwritten below
        # build cols vectorized
        base = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        cols = base
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_descent(x0, steps, T0, T1, lr0, lr1, mu, rng, deadline):
    import time as _t
    X = x0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        if (s & 63) == 0 and _t.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights, max is 1
        # (G-m)/T <= 0
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)
        # S >= 1 by construction (max entry 1 per row? not necessarily per row, but global max ensures at least 2 rows have 1)
        # guard tiny rows (isolated): keep S floor
        S = np.maximum(S, 1e-12)
        Avg = (W @ X) / S
        dot = (Avg * X).sum(axis=1, keepdims=True)
        Tang = Avg - dot * X
        # clip huge?
        V = mu * V + Tang
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    import time
    import numpy as np
    t0 = time.time()
    BUDGET = 172.0
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (153, 4):
        # try to fix: flatten/reshape or pad
        rng0 = np.random.default_rng(0)
        tmp = rng0.normal(size=(153, 4))
        try:
            flat = xb.ravel()
            tmp.ravel()[:min(flat.size, tmp.size)] = flat[:min(flat.size, tmp.size)]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    bestX = xb.copy()
    rng = np.random.default_rng(7)
    n, d = 153, 4

    # helper to get contact counts for targeted moves
    def contact_counts(X, thr_offset):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        thr = m - thr_offset
        C = (G > thr).sum(axis=1)
        return G, m, C

    trial = 0
    # reserve final polish
    while time.time() - t0 < BUDGET - 9.0:
        trial += 1
        mode = trial % 4
        Gbest, m_best, Cbest = contact_counts(bestX, 0.008)
        Xp = bestX.copy()
        if mode == 1:
            sigma = 0.02 + 0.01 * (rng.random())
            Xp = Xp + sigma * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1 = 0.02, 0.0012
            lr0, lr1 = 0.35, 0.04
            steps = 3000
            mu = 0.85
        elif mode == 2:
            # targeted reset of jammed points
            k = int(rng.integers(4, 11))
            # probability proportional to counts
            probs = Cbest.astype(np.float64) + 0.3
            probs /= probs.sum()
            idx = rng.choice(n, size=k, replace=False, p=probs)
            for j in idx:
                Xp[j] = rng.normal(size=(d,))
            # small noise to rest
            Xp = Xp + 0.012 * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1 = 0.035, 0.0015
            lr0, lr1 = 0.45, 0.05
            steps = 3500
            mu = 0.88
        elif mode == 3:
            sigma = 0.08 + 0.08 * rng.random()
            Xp = Xp + sigma * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1 = 0.05, 0.002
            lr0, lr1 = 0.5, 0.06
            steps = 3500
            mu = 0.85
        else:
            # rattler + void reinsertion: find most isolated and most crowded
            k = int(rng.integers(2, 6))
            # rattlers: smallest max-similarity
            Gtmp = bestX @ bestX.T
            np.fill_diagonal(Gtmp, -2.0)
            rowmax = Gtmp.max(axis=1)
            order = np.argsort(rowmax)  # smallest first = most isolated
            # replace most crowded (largest rowmax) with voids near isolated?
            crowded = np.argsort(-rowmax)[:k]
            for j in crowded:
                Xp[j] = rng.normal(size=(d,))
            Xp = Xp + 0.02 * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1 = 0.03, 0.001
            lr0, lr1 = 0.4, 0.04
            steps = 3000
            mu = 0.86
        # jitter annealing params deterministically
        T0 *= 1.0 + 0.2 * (rng.random() - 0.5)
        T1 *= 1.0 + 0.4 * (rng.random() - 0.5)
        T1 = float(np.clip(T1, 0.0006, 0.004))
        T0 = float(np.clip(T0, 0.012, 0.07))
        inner_deadline = t0 + BUDGET - 7.0
        Xd = _smooth_descent(Xp, steps, T0, T1, lr0, lr1, mu, rng, inner_deadline)
        # quick LP polish with short budget
        lp_budget = 2.2
        # if current descent already much worse, still polish briefly (may snap to new basin)
        lp_deadline = min(time.time() + lp_budget, t0 + BUDGET - 6.0)
        try:
            Xp2 = _lp_polish(Xd, lp_deadline)
        except Exception:
            Xp2 = Xd
        v = _max_cos(Xp2)
        if v < best_val - 1e-12 and np.all(np.isfinite(Xp2)):
            best_val = v
            bestX = Xp2.copy()
            # extra greedy: if improved a lot, intensify around it next trial (no extra work)
        # early exit if time low
        if time.time() - t0 > BUDGET - 9.0:
            break
    # final intensive polish on best
    try:
        final_deadline = t0 + BUDGET - 0.5
        bestX = _lp_polish(bestX, final_deadline)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

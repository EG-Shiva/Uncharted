import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def smooth_refine(X, steps, T0, T1, lr0, lr1, mom=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    if steps <= 1:
        return X
    logT0 = np.log(T0)
    logT1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1)
        T = np.exp(logT0 * (1.0 - frac) + logT1 * frac)
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        E = (G - m) / T
        np.clip(E, -50.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            continue
        E /= S
        Gr = E @ X
        Gr *= 2.0
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gr -= dot * X
        V *= mom
        V -= lr * Gr
        X += V
        nr = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        nr[nr == 0] = 1.0
        X /= nr
    return X

def greedy_spread(Xkeep, k, rng, n_try=4000):
    d = Xkeep.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # score to keepers
    s_keep = np.max(cand @ Xkeep.T, axis=1)
    chosen = []
    # greedy sequential for diversity
    best_idx = np.argsort(s_keep)[:min(64, n_try)]
    # we will iteratively pick
    sel = []
    # precompute cand-cand not needed; iterative
    cur_max = s_keep.copy()
    # mask to avoid repick
    used = np.zeros(n_try, dtype=bool)
    out = []
    for t in range(k):
        # pick unused minimal cur_max
        idx = int(np.argmin(np.where(used, np.inf, cur_max)))
        used[idx] = True
        q = cand[idx].copy()
        out.append(q)
        # update cur_max with similarity to q
        sim = cand @ q
        # only for unused
        np.maximum(cur_max, sim, out=cur_max)
    return np.array(out)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    bestX = None
    bestV = np.inf
    for path in ("best.npy", "record.npy"):
        try:
            c = np.load(path)
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (102, 12):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < bestV:
                bestV = v
                bestX = c.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.randn(102, 12)
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)

    n, d = 102, 12
    # exploration until ~132s
    trial = 0
    DEADLINE_EXPL = t0 + 130.0
    while time.time() < DEADLINE_EXPL:
        trial += 1
        # time check for at least one more trial + final
        if time.time() - t0 > 128:
            break
        r = rng.rand()
        Xt = bestX.copy()
        if r < 0.50:
            # cluster cavity: delete overloaded neighborhood, respread
            try:
                G = bestX @ bestX.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                cnt = (G > m - 0.005).sum(axis=1)
                order = np.argsort(-cnt)
                # pick center among top overloaded, sometimes random
                if rng.rand() < 0.7:
                    ci = int(order[rng.randint(0, 20)])
                else:
                    ci = int(rng.randint(0, n))
                # cluster = k nearest to center (largest dot)
                dots = bestX @ bestX[ci]
                dots[ci] = 2.0
                k = int(rng.randint(4, 13))  # 4..12
                # include center + k-1 nearest
                nn = np.argsort(-dots)[:k-1]
                kill = np.zeros(n, dtype=bool)
                kill[ci] = True
                kill[nn] = True
                keep_idx = np.where(~kill)[0]
                Xkeep = Xt[~kill]
                newpts = greedy_spread(Xkeep, int(kill.sum()), rng, n_try=4000)
                Xt[kill] = newpts
                steps, T0p, T1p, lr0, lr1, mom = 450, 0.02, 0.001, 0.03, 0.002, 0.88
                if rng.rand() < 0.3:
                    steps = 650
            except Exception:
                continue
        elif r < 0.78:
            # heavy kick to random subset
            k = int(rng.randint(6, 24))
            idxs = rng.choice(n, size=k, replace=False)
            sigma = 10 ** rng.uniform(-1.3, -0.3)
            Xt[idxs] += sigma * rng.randn(k, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0p, T1p, lr0, lr1, mom = 400, 0.015, 0.001, 0.025, 0.002, 0.87
        else:
            # global shake
            sigma = 10 ** rng.uniform(-2.0, -0.6)
            Xt = bestX + sigma * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0p, T1p, lr0, lr1, mom = 350, 0.012, 0.0009, 0.02, 0.0015, 0.90

        try:
            Xt = smooth_refine(Xt, steps, T0p, T1p, lr0, lr1, mom)
            # short cold quench
            Xt = smooth_refine(Xt, 120, 0.0007, 0.00035, 0.0012, 0.0006, 0.90)
        except Exception:
            continue
        if not np.all(np.isfinite(Xt)):
            continue
        v = max_cos(Xt)
        if v < bestV - 1e-12:
            bestV = v
            bestX = Xt.copy()

    # final cool of best
    try:
        if time.time() - t0 < 150:
            Xt = smooth_refine(bestX.copy(), 600, 0.006, 0.0004, 0.006, 0.0004, 0.90)
            v = max_cos(Xt)
            if v < bestV and np.all(np.isfinite(Xt)):
                bestV = v
                bestX = Xt
    except Exception:
        pass

    remaining = 177.0 - (time.time() - t0)
    if remaining > 7:
        try:
            bestX = lp_polish(bestX, min(remaining - 2.0, 22.0))
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

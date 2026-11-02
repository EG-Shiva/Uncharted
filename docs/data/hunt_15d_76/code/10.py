import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_opt(X, steps=1000, T0=0.02, T1=0.002, lr0=0.05, lr1=0.005):
    X = X.copy()
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        D = (G - m) / T
        np.clip(D, -20.0, 0.0, out=D)
        np.exp(D, out=D)
        np.fill_diagonal(D, 0.0)
        Grad = D @ X
        dot = np.einsum('ij,ij->i', Grad, X)[:, None]
        Gt = Grad - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        X -= lr * (Gt / nr)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPL_END = t0 + 145.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    base = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if v.shape == (76, 15) and np.all(np.isfinite(v)):
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    base = v.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((76, 15))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        best_val = _max_cos(base)
    best = base.copy()
    n, d = best.shape

    trial = 0
    sigmas = [0.02, 0.05, 0.10, 0.20]
    while time.time() < EXPL_END:
        trial += 1
        mode = trial % 3
        Xt = None
        if mode == 0:
            sigma = sigmas[trial % len(sigmas)]
            Xt = best + sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        elif mode == 1:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            row_max = G.max(axis=1)
            k = int(rng.integers(3, 12))
            idx = np.argsort(row_max)[:k]
            Xt = best.copy()
            Xt[idx] = rng.standard_normal((k, d))
            Xt[idx] /= np.linalg.norm(Xt[idx], axis=1, keepdims=True)
            # tiny shake of rest to break symmetry
            Xt += 0.005 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        else:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            c0 = int(rng.integers(n))
            order = np.argsort(-G[c0])
            nn = int(rng.integers(6, 16))
            neigh = order[1:1 + nn]
            Xt = best.copy()
            Xt[neigh] += 0.22 * rng.standard_normal((nn, d))
            Xt[c0] += 0.05 * rng.standard_normal(d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)

        # annealed smoothing, vary schedule slightly per trial
        T0 = float(rng.choice([0.015, 0.025, 0.04]))
        T1 = float(rng.choice([0.0015, 0.003, 0.005]))
        steps = 900 if (trial % 2 == 0) else 1300
        Xs = _smooth_opt(Xt, steps=steps, T0=T0, T1=T1, lr0=0.06, lr1=0.004)
        v = _max_cos(Xs)
        # decide whether to spend LP time
        thresh = best_val + (0.008 if mode != 1 else 0.015)
        if v < thresh and time.time() < EXPL_END - 2.0:
            Xs2 = _lp_polish(Xs, min(time.time() + 2.2, EXPL_END))
            v2 = _max_cos(Xs2)
            if v2 < v:
                Xs, v = Xs2, v2
        if v < best_val:
            best, best_val = Xs, v
            if time.time() > EXPL_END - 5:
                pass

    # final squeeze of the best
    if time.time() < HARD_END - 1.0:
        best = _lp_polish(best, HARD_END - 0.8)
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

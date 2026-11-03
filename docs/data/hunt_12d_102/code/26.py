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
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        T = np.exp(lT0 * (1.0 - frac) + lT1 * frac)
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        # weights exp((G-m)/T)
        E = G - m
        E /= T
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
        # cap constraints for speed
        if p > 2500:
            # keep most tight
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            ss = sel.copy()
            # rebuild c sel values
            csel = c[sel][idx]
            p = 2500
        else:
            csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
    X0 = None
    R0 = None
    try:
        X0 = np.load("best.npy")
    except Exception:
        X0 = None
    try:
        R0 = np.load("record.npy")
    except Exception:
        R0 = None
    bestX = None
    bestV = np.inf
    for c in (X0, R0):
        if c is None:
            continue
        try:
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

    trial = 0
    # fast hopping, no LP inside loop
    while time.time() - t0 < 150:
        trial += 1
        r = rng.rand()
        if r < 0.40:
            # global Gaussian kick, varied scale
            sigma = 10 ** rng.uniform(-2.4, -0.6)
            Xt = bestX + sigma * rng.randn(102, 12)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 450
            T0p, T1p = 0.014, 0.0010
            lr0, lr1 = 0.028, 0.002
            mom = 0.86 + rng.rand() * 0.07
        elif r < 0.65:
            # random subset re-randomization: strong topology change
            Xt = bestX.copy()
            k = int(rng.randint(3, 16))
            idxs = rng.choice(102, size=k, replace=False)
            # re-randomize subset completely
            Rn = rng.randn(k, 12)
            Rn /= np.linalg.norm(Rn, axis=1, keepdims=True)
            # mix with old to control jump size
            alpha = rng.uniform(0.6, 1.0)
            Xt[idxs] = alpha * Rn + (1 - alpha) * Xt[idxs]
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 550
            T0p, T1p = 0.012, 0.0008
            lr0, lr1 = 0.025, 0.0015
            mom = 0.89
        elif r < 0.85:
            # worst-point best-hole reinsertion
            Xt = bestX.copy()
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            worst_max = np.max(G, axis=1)
            order = np.argsort(-worst_max)
            nk = 1 if rng.rand() < 0.6 else (2 if rng.rand() < 0.7 else 3)
            worst = order[:nk]
            mask = np.ones(102, dtype=bool)
            mask[worst] = False
            Xo = Xt[mask]
            n_try = 1500
            Rc = rng.randn(n_try, 12)
            Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)
            Xcur = Xo.copy()
            for wi in worst:
                scores = np.max(Rc @ Xcur.T, axis=1)
                bi = int(np.argmin(scores))
                Xt[wi] = Rc[bi]
                Xcur = np.vstack([Xcur, Rc[bi][None, :]])
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 600
            T0p, T1p = 0.010, 0.0007
            lr0, lr1 = 0.022, 0.0012
            mom = 0.88
        else:
            # cap / cluster shake along random direction
            Xt = bestX.copy()
            u = rng.randn(12)
            u /= np.linalg.norm(u)
            proj = bestX @ u
            k = int(rng.randint(6, 22))
            if rng.rand() < 0.5:
                idxs = np.argsort(-proj)[:k]
            else:
                idxs = np.argsort(proj)[:k]
            sigma = 10 ** rng.uniform(-1.6, -0.5)
            Xt[idxs] += sigma * rng.randn(k, 12)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 500
            T0p, T1p = 0.013, 0.0009
            lr0, lr1 = 0.03, 0.0018
            mom = 0.90
        try:
            Xt = smooth_refine(Xt, steps, T0p, T1p, lr0, lr1, mom)
            # short cold snap
            Xt = smooth_refine(Xt, 120, 0.0007, 0.0003, 0.0012, 0.0005, 0.90)
        except Exception:
            continue
        if not np.all(np.isfinite(Xt)):
            continue
        v = max_cos(Xt)
        if v < bestV - 1e-12:
            bestV = v
            bestX = Xt.copy()
        if time.time() - t0 > 149:
            break

    # brief extra anneal of incumbent
    if time.time() - t0 < 162:
        try:
            Xt = smooth_refine(bestX.copy(), 500, 0.006, 0.0004, 0.006, 0.0004, 0.90)
            v = max_cos(Xt)
            if v < bestV and np.all(np.isfinite(Xt)):
                bestV = v
                bestX = Xt
        except Exception:
            pass

    remaining = 178.0 - (time.time() - t0)
    if remaining > 6:
        try:
            bestX = lp_polish(bestX, min(remaining - 2.0, 20.0))
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

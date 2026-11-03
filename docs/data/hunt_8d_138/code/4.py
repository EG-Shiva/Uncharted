import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower largest cosine."""
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


def anneal_exp(X0, steps, a0, a1, s0, s1, mu=0.8, noise=0.0, rng=None):
    X = X0.copy()
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * f
        st = s0 + (s1 - s0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        # stable exp weights focused near max
        W = np.exp(alpha * (G - cm))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        d = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - d * X
        rms = np.sqrt((gt ** 2).mean()) + 1e-12
        gt /= rms
        V = mu * V + gt
        dV = (V * X).sum(axis=1, keepdims=True)
        Vt = V - dV * X
        rmsV = np.sqrt((Vt ** 2).mean()) + 1e-12
        Vt /= rmsV
        X -= st * Vt
        if noise > 0:
            N = rng.standard_normal(X.shape)
            N -= (N * X).sum(axis=1, keepdims=True) * X
            X += noise * N
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X


def push_target(X0, target, steps, s0, s1, mu=0.9, noise=0.0, rng=None):
    X = X0.copy()
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        st = s0 + (s1 - s0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        if G.max() <= target:
            break
        E = G - target
        E[E < 0] = 0.0
        W = E * E  # smooth squared violation
        # add tiny global repulsion to keep rattlers spread
        # (small uniform weight)
        # W += 1e-4 * np.exp(10.0*(G - G.max()))
        grad = W @ X
        d = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - d * X
        rms = np.sqrt((gt ** 2).mean()) + 1e-12
        if rms < 1e-15:
            break
        gt /= rms
        V = mu * V + gt
        dV = (V * X).sum(axis=1, keepdims=True)
        Vt = V - dV * X
        rmsV = np.sqrt((Vt ** 2).mean()) + 1e-12
        Vt /= rmsV
        X -= st * Vt
        if noise > 0:
            N = rng.standard_normal(X.shape)
            N -= (N * X).sum(axis=1, keepdims=True) * X
            X += noise * N
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=float))
    if len(cands) == 0:
        # random fallback
        X = rng.standard_normal((138, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    # pick best of loaded
    best = None
    best_val = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    n, d = best.shape

    def time_left():
        return BUDGET - (time.time() - t0)

    # contact info for targeted moves
    def contacts(X, tol=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        cnt = (G > m - tol).sum(axis=1)
        return cnt, m, G

    trial = 0
    # main basin-hopping loop, reserve ~18s for final polish
    while time_left() > 20 and trial < 1000:
        trial += 1
        if time_left() < 5:
            break
        cnt, m, _ = contacts(best)
        order = np.argsort(-cnt)  # overloaded first
        mode = trial % 6
        Xinit = best.copy()
        if mode == 0 or mode == 3:
            # teleport k points (prefer overloaded + random rattlers)
            k = int(rng.integers(4, 11))
            # half from overloaded top, half random
            k1 = min(k // 2 + 1, n)
            idx1 = order[:30]
            pick1 = idx1[rng.choice(len(idx1), size=min(k1, len(idx1)), replace=False)]
            pick2 = rng.choice(n, size=k - len(pick1), replace=False)
            idx = np.unique(np.concatenate([pick1, pick2]))
            R = rng.standard_normal((len(idx), d))
            R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-30)
            Xinit[idx] = R
            # small shake of rest
            Xinit += 0.015 * rng.standard_normal(Xinit.shape)
            Xinit /= (np.linalg.norm(Xinit, axis=1, keepdims=True) + 1e-30)
            steps1 = 1400
            a0, a1 = 12.0, 250.0
            s0, s1 = 0.022, 0.002
            nz = 0.0015
        elif mode == 1:
            sigma = 0.22
            Xinit = best + sigma * rng.standard_normal(best.shape)
            Xinit /= (np.linalg.norm(Xinit, axis=1, keepdims=True) + 1e-30)
            steps1 = 1500
            a0, a1 = 10.0, 300.0
            s0, s1 = 0.025, 0.0015
            nz = 0.002
        elif mode == 2:
            sigma = 0.08
            Xinit = best + sigma * rng.standard_normal(best.shape)
            Xinit /= (np.linalg.norm(Xinit, axis=1, keepdims=True) + 1e-30)
            steps1 = 1200
            a0, a1 = 20.0, 350.0
            s0, s1 = 0.015, 0.001
            nz = 0.0008
        elif mode == 4:
            sigma = 0.12
            Xinit = best + sigma * rng.standard_normal(best.shape)
            Xinit /= (np.linalg.norm(Xinit, axis=1, keepdims=True) + 1e-30)
            steps1 = 1300
            a0, a1 = 15.0, 200.0
            s0, s1 = 0.02, 0.002
            nz = 0.001
        else:
            # big shake
            sigma = 0.35
            Xinit = best + sigma * rng.standard_normal(best.shape)
            Xinit /= (np.linalg.norm(Xinit, axis=1, keepdims=True) + 1e-30)
            steps1 = 1600
            a0, a1 = 8.0, 220.0
            s0, s1 = 0.03, 0.002
            nz = 0.0025

        X1 = anneal_exp(Xinit, steps=steps1, a0=a0, a1=a1, s0=s0, s1=s1, mu=0.8, noise=nz, rng=rng)
        v1 = max_cos(X1)

        # phase 2: push below best
        Xcand, vcand = X1, v1
        if v1 < best_val + 0.015:
            tgt = best_val - 0.0008 - float(rng.random()) * 0.002
            # don't aim impossibly low
            tgt = max(tgt, best_val - 0.006)
            X2 = push_target(X1, target=tgt, steps=700, s0=0.012, s1=0.0012, mu=0.9, noise=0.0005, rng=rng)
            v2 = max_cos(X2)
            if v2 < vcand:
                Xcand, vcand = X2, v2

        # refine promising candidates with sharp anneal
        if vcand < best_val + 0.0025:
            X3 = anneal_exp(Xcand, steps=450, a0=250.0, a1=500.0, s0=0.005, s1=0.0006, mu=0.85, noise=0.0, rng=rng)
            v3 = max_cos(X3)
            if v3 < vcand:
                Xcand, vcand = X3, v3

        # decide LP polish
        need_lp = False
        lp_time = 0
        if vcand < best_val - 1e-12:
            need_lp = True
            lp_time = 3.0
        elif vcand < best_val + 0.0012 and (trial % 3 == 0):
            need_lp = True
            lp_time = 1.6
        elif vcand < best_val + 0.0025 and (trial % 7 == 0):
            need_lp = True
            lp_time = 1.2

        if need_lp and time_left() > lp_time + 8:
            Xp = lp_polish(Xcand, seconds=lp_time)
            vp = max_cos(Xp)
            if vp < best_val:
                best = Xp
                best_val = vp
            elif vp < vcand and vcand < best_val:
                best = Xp
                best_val = vp
        else:
            if vcand < best_val:
                best = Xcand
                best_val = vcand

    # final polish of best with remaining time
    tl = time_left()
    if tl > 3:
        ft = min(tl - 1.5, 14.0)
        if ft > 1:
            best = lp_polish(best, seconds=ft)
            best_val = max_cos(best)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: remove NaN
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True) + 1e-30
    best = best / nr
    return best

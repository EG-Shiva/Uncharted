import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), float(g.max())


def _anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline, max_pairs=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:max_pairs]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _refine_void(v0, Xrem, beta=140.0, steps=160, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        nrm = np.linalg.norm(v) + 1e-30
        v = v / nrm
    return v


def _best_void(Xrem, rng, ncand=20000):
    nrem, d = Xrem.shape
    cand = rng.normal(size=(ncand, d))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    smax = (cand @ Xrem.T).max(axis=1)
    bi = int(np.argmin(smax))
    v0 = cand[bi]
    try:
        v0 = _refine_void(v0, Xrem, beta=140.0, steps=160, lr=0.08)
    except Exception:
        pass
    return v0


def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # short initial tighten
    try:
        best = _lp_polish(best, min(deadline - 60.0, time.time() + 2.0))
        best_val = _max_cos(best)
    except Exception:
        pass

    it = 0
    while time.time() < deadline - 9.0 and it < 500:
        it += 1
        now = time.time()
        mode = it % 6
        try:
            if mode == 0:
                # fresh random basin, cheap screen
                y = rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, steps=900, beta0=10.0, beta1=400.0, lr0=0.03, lr1=0.001)
                v = _max_cos(y)
                if v < best_val + 0.02 and time.time() < deadline - 9.0:
                    y = _lp_polish(y, min(deadline - 9.0, time.time() + 1.5))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best_val = v
                    best = y.copy()
            elif mode == 1:
                # rattler-void rebirth: remove worst + rattlers, greedy refill
                per, m = _per_point_max(best)
                ratt = np.where(per < m - 1e-6)[0]
                order = np.argsort(-per)
                worst = order[:8]
                # choose up to 12 victims: all worst + sample of rattlers
                cand_v = list(worst)
                if len(ratt) > 0:
                    nr = min(6, len(ratt))
                    sel = rng.choice(ratt, size=nr, replace=False)
                    for s in sel:
                        if int(s) not in cand_v:
                            cand_v.append(int(s))
                # a couple random others for diversity
                for _ in range(2):
                    r = int(rng.integers(0, n))
                    if r not in cand_v:
                        cand_v.append(r)
                y = best.copy()
                # slight core jiggle to open new topology
                y = y + 0.008 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                for vv in cand_v:
                    Xrem = np.delete(y, vv, axis=0)
                    y[vv] = _best_void(Xrem, rng, ncand=15000)
                y = _anneal(y, steps=600, beta0=30.0, beta1=350.0, lr0=0.015, lr1=0.001)
                v = _max_cos(y)
                if v < best_val + 0.02:
                    y = _lp_polish(y, min(deadline - 9.0, time.time() + 1.8))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best_val = v
                    best = y.copy()
            elif mode in (2, 3):
                sigma = 0.10 if mode == 2 else 0.035
                y = best + sigma * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                k = 5 if mode == 2 else 3
                idx = rng.choice(n, size=k, replace=False)
                rp = rng.normal(size=(k, d))
                rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                y[idx] = rp
                y = _anneal(y, steps=600, beta0=20.0, beta1=300.0, lr0=0.02, lr1=0.0015)
                v = _max_cos(y)
                if v < best_val + 0.02:
                    y = _lp_polish(y, min(deadline - 9.0, time.time() + 1.2))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best_val = v
                    best = y.copy()
            elif mode == 4:
                # contact-guided shake of overloaded cluster
                per, m = _per_point_max(best)
                order = np.argsort(-per)
                top = order[:20]
                y = best.copy()
                u = rng.normal(size=(d,))
                u = u / (np.linalg.norm(u) + 1e-30)
                # push top points along / opposite u to break symmetry
                for ii, ti in enumerate(top):
                    sgn = 1.0 if (ii % 2 == 0) else -1.0
                    y[ti] = y[ti] + sgn * 0.06 * u + 0.02 * rng.normal(size=(d,))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                # rebirth 2 rattlers as well
                per2, m2 = _per_point_max(y)
                ratt2 = np.where(per2 < m2 - 1e-6)[0]
                if len(ratt2) > 0:
                    for _ in range(min(2, len(ratt2))):
                        vv = int(rng.choice(ratt2))
                        Xrem = np.delete(y, vv, axis=0)
                        y[vv] = _best_void(Xrem, rng, ncand=12000)
                y = _anneal(y, steps=700, beta0=25.0, beta1=380.0, lr0=0.018, lr1=0.001)
                v = _max_cos(y)
                if v < best_val + 0.02:
                    y = _lp_polish(y, min(deadline - 9.0, time.time() + 1.5))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best_val = v
                    best = y.copy()
            else:
                # cap-based directional move using PCA-like random plane
                y = best + 0.06 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, steps=500, beta0=40.0, beta1=320.0, lr0=0.012, lr1=0.001)
                v = _max_cos(y)
                if v < best_val + 0.015:
                    y = _lp_polish(y, min(deadline - 9.0, time.time() + 1.2))
                    v = _max_cos(y)
                if v < best_val - 1e-12:
                    best_val = v
                    best = y.copy()
        except Exception:
            continue

    try:
        best = _lp_polish(best, min(deadline - 0.5, time.time() + 7.0), max_pairs=4000)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

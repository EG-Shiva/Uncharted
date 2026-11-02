import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _quench(X0, steps, beta0, beta1, lr0, lr1, rng, deadline, noise=0.0):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    mu = 0.9
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > deadline:
            break
        f = t / steps
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 * (lr1 / lr0) ** f if lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        # row-wise softmax weights, fast and stable
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-18
        Gw = (E @ X) / rs
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        gt = Gw - dot * X
        if noise > 0:
            nz = rng.standard_normal((n, d))
            nz -= np.sum(nz * X, axis=1, keepdims=True) * X
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            gt = (1.0 - noise) * gt + noise * nn.mean() * 0 + noise * nz * 0.5
        vel = mu * vel + (1.0 - mu) * gt
        # normalized step per point for stability
        vn = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12
        step = lr * vel / vn * min(1.0, float(np.median(vn)) * 50 + 0.2)
        # simpler: lr * normalized vel
        step = lr * vel / vn
        X = X - step
        X = _norm(X)
    return X

def _hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = c[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 138.0
    T_LP_END = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()

    trial = 0
    while time.time() < T_SEARCH_END:
        trial += 1
        remain = T_SEARCH_END - time.time()
        if remain < 2.0:
            break
        prop = x_best.copy()
        mode = trial % 7
        if mode == 0:
            sigma = 0.008 + 0.008 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1 = 700, 60.0, 600.0, 0.015, 0.0015
            nz = 0.0
        elif mode == 1:
            sigma = 0.03 + 0.04 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1 = 800, 40.0, 500.0, 0.02, 0.002
            nz = 0.05
        elif mode == 2:
            sigma = 0.12 + 0.12 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1 = 900, 25.0, 400.0, 0.025, 0.002
            nz = 0.08
        elif mode == 3:
            # subset re-randomize
            k = int(rng.integers(3, 7))
            victims = rng.choice(n, size=k, replace=False)
            for vi in victims:
                prop[vi] = rng.standard_normal(d)
            prop = _norm(prop)
            steps, b0, b1, l0, l1 = 900, 30.0, 500.0, 0.02, 0.0015
            nz = 0.02
        elif mode == 4:
            # hole reinsertion of overloaded hubs
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = np.max(G)
            cnt = np.sum(G > mx - 2e-3, axis=1)
            k = 1 if (trial % 2 == 0) else 2
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Y = prop[keep_mask]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm(prop + 0.006 * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1 = 800, 50.0, 600.0, 0.015, 0.0015
            nz = 0.0
        elif mode == 5:
            # cluster shake: pick a point and perturb its neighbours
            anchor = int(rng.integers(0, n))
            dots = prop @ prop[anchor]
            order = np.argsort(-dots)[:8]
            prop[order] = _norm(prop[order] + 0.15 * rng.standard_normal((len(order), d)))
            prop = _norm(prop + 0.01 * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1 = 800, 40.0, 550.0, 0.018, 0.0015
            nz = 0.03
        else:
            sigma = 0.05 * rng.random() + 0.01
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1 = 600, 80.0, 700.0, 0.012, 0.001
            nz = 0.0
        # adapt steps to remaining time
        if remain < 15:
            steps = min(steps, 400)
        prop = _quench(prop, steps=steps, beta0=b0, beta1=b1, lr0=l0, lr1=l1, rng=rng, deadline=T_SEARCH_END, noise=nz)
        s = _max_cos(prop)
        if s < best_score:
            best_score = s
            x_best = prop.copy()
        # occasional reset to best already (we always start from best)
        if trial % 40 == 0 and time.time() < T_SEARCH_END - 5:
            # small extra quench of current best to squeeze
            x_best = _quench(x_best, steps=300, beta0=200.0, beta1=800.0, lr0=0.008, lr1=0.001, rng=rng, deadline=T_SEARCH_END)
            s2 = _max_cos(x_best)
            if s2 < best_score:
                best_score = s2
            # else keep (quench is monotonic-ish but not guaranteed) -> reload if worsened a lot
            if s2 > best_score + 1e-4:
                # revert is handled by best_score; re-evaluate: keep best copy
                pass
            best_score = min(best_score, s2)

    # final LP squeeze on the best found
    if time.time() < T_LP_END:
        try:
            x_best = _lp_polish(x_best, T_LP_END)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

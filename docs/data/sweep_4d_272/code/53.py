import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(X), axis=1)
    if np.any(bad):
        for i in np.where(bad)[0]:
            v = np.array([1.0, 0.0, 0.0, 0.0])
            X[i] = v
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lp_polish(x, seconds, deadline=1e18):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def balanced_quench(X, steps, tau0, tau1, lr0, noise0, noise1, rng, max_step=0.02, trow=0.005):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1) if steps > 1 else 1.0
        T = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 * (1.0 - 0.7 * frac)
        noise_amp = noise0 * (1.0 - frac) + noise1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp((G - gmax) / T)
        np.fill_diagonal(W, 0.0)
        # hub boost: points near gmax get larger weight
        rowmax = np.max(G, axis=1)
        b = np.exp((rowmax - gmax) / trow)
        boost = 0.3 + 0.7 * b
        # also degree boost for heavy-contact points
        try:
            deg = np.sum(G > gmax - 2e-4, axis=1)
            boost = boost * (1.0 + 0.05 * np.minimum(deg, 10))
        except Exception:
            pass
        W = W * (boost[:, None] * boost[None, :])
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5)
        if S < 1e-30:
            gt = np.zeros_like(X)
        else:
            W /= S
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
        step = lr * gt
        norms = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, max_step / (norms + 1e-12))
        step = step * scale
        X = X - step
        if noise_amp > 0:
            N = rng.standard_normal(size=X.shape)
            ndot = np.sum(N * X, axis=1, keepdims=True)
            Nt = N - ndot * X
            X = X + (noise_amp * 0.5) * Nt
        X = _norm(X)
    return X


def best_hole(X, exclude, rng, n_cand=6000):
    cand = rng.standard_normal(size=(n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X.T
    if exclude is not None:
        try:
            S[:, exclude] = -2.0
        except Exception:
            pass
    vals = np.max(S, axis=1)
    return cand[int(np.argmin(vals))]


def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPL_END = t0 + 122.0
    rng = np.random.default_rng(7)

    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (272, 4):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal(size=(272, 4))
        X0 = _norm(X0)
        best0 = _max_cos(X0)

    Xbest = X0.copy()
    best_val = float(best0)
    n = 272

    trial = 0
    sigmas = [0.05, 0.09, 0.14, 0.22]
    while time.time() < EXPL_END and time.time() < HARD - 45:
        trial += 1
        remain = min(EXPL_END - time.time(), HARD - 45 - time.time())
        if remain < 0.8:
            break
        X = Xbest.copy()
        mode = trial % 7
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(np.max(G))
            rowmax = np.max(G, axis=1)
            cnt = np.sum(G > gmax - 1e-7, axis=1)
        except Exception:
            gmax = best_val
            rowmax = np.zeros(n)
            cnt = np.zeros(n)
        if mode == 0:
            worst = np.argsort(rowmax)[-3:]
            for idx in worst:
                h = best_hole(X, int(idx), rng)
                h = h + 0.002 * rng.standard_normal(4)
                h /= np.linalg.norm(h)
                X[int(idx)] = h
            X = X + 0.02 * rng.standard_normal(size=X.shape)
            X = _norm(X)
        elif mode == 1:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) > 0:
                for idx in ratt[:4]:
                    h = best_hole(X, int(idx), rng)
                    h = h + 0.002 * rng.standard_normal(4)
                    h /= np.linalg.norm(h)
                    X[int(idx)] = h
                X = X + 0.015 * rng.standard_normal(size=X.shape)
                X = _norm(X)
            else:
                sg = sigmas[trial % len(sigmas)]
                X = X + sg * rng.standard_normal(size=X.shape)
                X = _norm(X)
        elif mode == 2:
            hub = int(np.argmax(cnt + rowmax * 1e-6))
            d2 = np.sum((X - X[hub]) ** 2, axis=1)
            nn = np.argsort(d2)[:25]
            X[nn] = X[nn] + 0.25 * rng.standard_normal(size=(len(nn), 4))
            X = _norm(X)
        elif mode == 3:
            sg = sigmas[trial % len(sigmas)]
            X = X + sg * rng.standard_normal(size=X.shape)
            X = _norm(X)
        elif mode == 4:
            ij = int(np.argmax(G))
            ii, jj = ij // n, ij % n
            h = best_hole(X, int(jj), rng)
            X[int(jj)] = h
            X = X + 0.03 * rng.standard_normal(size=X.shape)
            X = _norm(X)
        elif mode == 5:
            sel = rng.choice(n, size=40, replace=False)
            X[sel] = X[sel] + 0.28 * rng.standard_normal(size=(40, 4))
            X = _norm(X)
        else:
            X = X + 0.035 * rng.standard_normal(size=X.shape)
            X = _norm(X)

        if remain < 8:
            steps = 350
        elif remain < 20:
            steps = 550
        else:
            steps = 750
        # vary temperature slightly per trial
        tau0 = 0.012 if (trial % 2 == 0) else 0.018
        tau1 = 0.0015 if (trial % 3 != 0) else 0.003
        lr0 = 10.0 if (trial % 2 == 0) else 13.0
        Xq = balanced_quench(X, steps=steps, tau0=tau0, tau1=tau1, lr0=lr0,
                             noise0=0.012, noise1=0.0, rng=rng, max_step=0.02,
                             trow=0.005)
        v = _max_cos(Xq)
        if v < best_val - 1e-12:
            Xbest = Xq
            best_val = float(v)
            if best_val < 0.8975 and time.time() < EXPL_END - 10:
                Xlp = lp_polish(Xbest, seconds=3.0, deadline=HARD - 30)
                vlp = _max_cos(Xlp)
                if vlp < best_val:
                    Xbest, best_val = Xlp, vlp

    remain = HARD - time.time() - 2.0
    if remain > 5:
        secs = min(remain, 45.0)
        Xp = lp_polish(Xbest, seconds=secs, deadline=HARD - 1.0)
        try:
            vp = _max_cos(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                Xbest, best_val = Xp, vp
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest

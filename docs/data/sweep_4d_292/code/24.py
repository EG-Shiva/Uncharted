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
    return g.max(axis=1)

def _anneal(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x

def _sharp_refine(x, steps=400, beta=1500.0, lr=0.02):
    # high-beta descent to directly push down the max
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.7
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
        # anneal beta slightly up
        beta *= 1.001
        if beta > 5000:
            beta = 5000
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _hole_fill(Ykeep, k, rng, Nc=2500):
    # greedily insert k points into largest holes of Ykeep
    Y = Ykeep.copy()
    for _ in range(k):
        cand = rng.standard_normal((Nc, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ Y.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        c = cand[bi].copy()
        # local refine c away from closest points (soft max descent, others fixed)
        for s in range(40):
            dots = Y @ c
            md = dots.max()
            W = np.exp(30.0 * (dots - md))
            W /= (W.sum() + 1e-300)
            g = W @ Y
            g = g - (g @ c) * c
            c = c - 0.12 * g
            n = np.linalg.norm(c)
            if n < 1e-12:
                c = cand[bi].copy()
                break
            c /= n
        Y = np.vstack([Y, c.reshape(1, 4)])
    return Y

def solve():
    t0 = time.time()
    TL = 172.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            xb = None
    if xb is None or xb.shape != (292, 4):
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    rng = np.random.default_rng(1234567)
    n, d = 292, 4
    explore_end = t0 + TL - 34.0
    it = 0
    sigmas = [0.005, 0.012, 0.03, 0.07, 0.15, 0.30]
    while time.time() < explore_end:
        it += 1
        if time.time() + 0.45 > explore_end and it > 8:
            break
        X = x_best.copy()
        r = rng.random()
        # choose surgery
        if r < 0.55:
            # kill worst points and fill largest holes
            ppm = _per_point_max(X)
            noise = rng.standard_normal(n) * 0.002
            score = ppm + noise
            if rng.random() < 0.35:
                k = int(rng.integers(1, 4))
            elif rng.random() < 0.7:
                k = int(rng.integers(4, 8))
            else:
                k = int(rng.integers(8, 13))
            k = min(k, 12)
            kill = np.argsort(-score)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            Ykeep = X[keep_mask]
            # small shake of keepers
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            if sigma > 0:
                Ykeep = Ykeep + sigma * 0.5 * rng.standard_normal(Ykeep.shape)
                Ykeep /= np.linalg.norm(Ykeep, axis=1, keepdims=True)
            X = _hole_fill(Ykeep, k, rng, Nc=2000)
            # shake all a little
            sigma2 = float(rng.choice([0.003, 0.008, 0.02]))
            X = X + sigma2 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif r < 0.75:
            # contact-edge targeted shake + cluster shake
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # pick a random active edge
            iu0 = np.triu_indices(n, 1)
            c0 = G[iu0]
            sel = np.where(c0 > m - 2e-3)[0]
            if len(sel) > 0:
                e = sel[int(rng.integers(len(sel)))]
                i, j = int(iu0[0][e]), int(iu0[1][e])
                # shake i,j and their neighbours strongly
                di = G[i].copy(); dj = G[j].copy()
                ni = np.argsort(-di)[:8]; nj = np.argsort(-dj)[:8]
                tgt = np.unique(np.concatenate([[i, j], ni, nj]))
                X[tgt] += 0.08 * rng.standard_normal((len(tgt), d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                X += 0.02 * rng.standard_normal(X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            # occasional extra random kills
            if rng.random() < 0.4:
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, size=k, replace=False)
                R = rng.standard_normal((k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X[idx] = R
        else:
            # global shake / restart mix
            if (it % 11) == 0 and rng.random() < 0.5:
                R = rng.standard_normal((n, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                if rng.random() < 0.5:
                    X = R
                else:
                    X = X + 0.6 * rng.standard_normal((n, d))
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                sigma = sigmas[int(rng.integers(len(sigmas)))]
                X = X + sigma * rng.standard_normal((n, d))
                if rng.random() < 0.5:
                    k = int(rng.integers(1, 6))
                    idx = rng.choice(n, size=k, replace=False)
                    R = rng.standard_normal((k, d))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    X[idx] = R
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        # random anneal hyperparams
        u = rng.random()
        if u < 0.35:
            steps, b0, b1, lr0, mu = 1100, 25.0, 350.0, 0.06, 0.85
        elif u < 0.65:
            steps, b0, b1, lr0, mu = 1600, 15.0, 250.0, 0.10, 0.88
        elif u < 0.85:
            steps, b0, b1, lr0, mu = 2200, 40.0, 500.0, 0.04, 0.80
        else:
            steps, b0, b1, lr0, mu = 800, 10.0, 150.0, 0.15, 0.90
        remain = explore_end - time.time()
        # ~0.00028 s per step estimate
        maxsteps = int(remain / 0.00028) - 200
        if maxsteps < 300:
            break
        if steps > maxsteps:
            steps = max(300, maxsteps)
        Xo = _anneal(X, steps, b0, b1, lr0, mu)
        val = _max_cos(Xo)
        if not np.all(np.isfinite(Xo)):
            continue
        # screen: if promising, sharp refine + short LP
        if val < best + 0.0025:
            Xo2 = _sharp_refine(Xo.copy(), steps=250, beta=800.0, lr=0.02)
            val2 = _max_cos(Xo2)
            if val2 < val:
                Xo, val = Xo2, val2
            if val < best + 0.0006:
                # short LP to reach basin bottom
                tlp = min(2.0, max(0.6, (explore_end - time.time()) * 0.15))
                try:
                    Xo3 = _lp_polish(Xo.copy(), tlp)
                    val3 = _max_cos(Xo3)
                    if np.all(np.isfinite(Xo3)):
                        Xo, val = Xo3, val3
                except Exception:
                    pass
        if val < best:
            best = val
            x_best = Xo.copy()
    # final stage: polish + rattler reinsertion loop
    time_left = TL - (time.time() - t0) - 1.0
    time_left = max(6.0, min(36.0, time_left))
    fstart = time.time()
    # initial polish with part of budget
    try:
        x_best = _lp_polish(x_best, min(12.0, time_left * 0.45))
    except Exception:
        pass
    best = _max_cos(x_best)
    # rattler / overloaded surgery under frozen LP
    while time.time() - fstart < time_left - 2.0:
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        deg = (G > m - 1e-7).sum(axis=1)
        ratt = np.where(deg == 0)[0]
        ppm = G.max(axis=1)
        # pick victims: rattlers first, else worst points
        if len(ratt) > 0:
            k = min(len(ratt), int(rng.integers(1, 4)))
            kill = rng.choice(ratt, size=k, replace=False)
        else:
            k = int(rng.integers(1, 4))
            kill = np.argsort(-ppm + rng.standard_normal(n) * 1e-4)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[kill] = False
        Ykeep = x_best[keep_mask]
        Ynew = _hole_fill(Ykeep, k, rng, Nc=3000)
        # tiny shake then anneal briefly then polish
        Ynew += 0.004 * rng.standard_normal(Ynew.shape)
        Ynew /= np.linalg.norm(Ynew, axis=1, keepdims=True)
        Ynew = _anneal(Ynew, 500, 30.0, 400.0, 0.05, 0.85)
        Ynew = _sharp_refine(Ynew, steps=200, beta=1000.0, lr=0.015)
        try:
            Yp = _lp_polish(Ynew, 2.5)
        except Exception:
            continue
        v = _max_cos(Yp)
        if v < best and np.all(np.isfinite(Yp)):
            best = v
            x_best = Yp.copy()
        else:
            # if no progress twice, do full polish of best and exit
            if rng.random() < 0.25:
                break
    # final polish remainder
    rem = TL - (time.time() - t0) - 0.8
    if rem > 1.0:
        try:
            x_best = _lp_polish(x_best, min(rem, 10.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return x_best

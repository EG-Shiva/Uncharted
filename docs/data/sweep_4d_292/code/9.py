import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _soft_anneal(x, steps, beta0, beta1, lr0, mu):
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
        v = v - vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
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

def _greedy_reinsert(X, victims, rng, n_cand=6000):
    n, d = X.shape
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    victims = np.asarray(victims, dtype=int).ravel()
    for v in victims:
        others = np.delete(Xn, v, axis=0)
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        smax = (C @ others.T).max(axis=1)
        bi = int(np.argmin(smax))
        cur = C[bi].copy()
        cur_score = float(smax[bi])
        for s, rep in ((0.05, 50), (0.02, 50), (0.008, 50), (0.003, 40)):
            for _ in range(rep):
                prop = cur + s * rng.standard_normal(d)
                nn = np.linalg.norm(prop)
                if nn < 1e-12:
                    continue
                prop /= nn
                sc = float((prop @ others.T).max())
                if sc < cur_score:
                    cur, cur_score = prop, sc
        Xn[v] = cur
    return Xn

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal((292, 4))
    if xb.shape != (292, 4):
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((292, 4))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    n, d = 292, 4
    rng = np.random.default_rng(12345)
    explore_deadline = t0 + TIME_LIMIT - 34.0
    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.30]
    while time.time() < explore_deadline:
        it += 1
        if time.time() + 0.4 > explore_deadline and it > 5:
            break
        X = x_best.copy()
        r = rng.random()
        # pick victims
        if r < 0.42:
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            X = X + sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if rng.random() < 0.65:
                k = int(rng.integers(1, 6))
                idx = rng.choice(n, size=k, replace=False)
                nc = 5000 if rng.random() < 0.5 else 9000
                X = _greedy_reinsert(X, idx, rng, n_cand=nc)
        elif r < 0.78:
            # targeted: worst per-point max
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            k = int(rng.integers(1, 7))
            victims = np.argsort(-per_max)[:k]
            if rng.random() < 0.35:
                victims[-1] = int(rng.integers(n))
            # small shake of rest
            X = X + 0.008 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _greedy_reinsert(X, victims, rng, n_cand=9000)
        else:
            # rattler / contact based
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-9).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            per_max = G2.max(axis=1)
            order = np.argsort(-per_max)
            k = int(rng.integers(2, 7))
            victims = list(order[:max(1, k - len(ratt))])
            for rr in ratt:
                if len(victims) >= k:
                    break
                victims.append(int(rr))
            # also sometimes include a random overloaded (8+ contacts)
            if rng.random() < 0.3:
                hi = np.where(cnt >= 7)[0]
                if len(hi) > 0:
                    victims[-1] = int(hi[int(rng.integers(len(hi)))])
            victims = np.array(victims[:k], dtype=int)
            X = X + 0.015 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _greedy_reinsert(X, victims, rng, n_cand=10000)
        # anneal hyperparams
        u = rng.random()
        if u < 0.35:
            steps, b0, b1, lr0, mu = 1200, 25.0, 350.0, 0.06, 0.85
        elif u < 0.65:
            steps, b0, b1, lr0, mu = 1800, 15.0, 250.0, 0.10, 0.88
        elif u < 0.85:
            steps, b0, b1, lr0, mu = 2500, 40.0, 500.0, 0.04, 0.80
        else:
            steps, b0, b1, lr0, mu = 900, 10.0, 150.0, 0.15, 0.90
        if time.time() + steps * 0.00025 > explore_deadline:
            steps = max(400, int((explore_deadline - time.time()) / 0.00025))
            if steps <= 100:
                break
        Xo = _soft_anneal(X, steps, b0, b1, lr0, mu)
        val = _max_cos(Xo)
        if val < best and np.all(np.isfinite(Xo)):
            # snap promising basins with quick LP to get true value
            if val < best + 0.002 and (explore_deadline - time.time()) > 4.0:
                try:
                    Xo2 = _lp_polish(Xo, 1.2)
                    val2 = _max_cos(Xo2)
                    if np.all(np.isfinite(Xo2)) and val2 < val + 1e-12:
                        Xo, val = Xo2, val2
                except Exception:
                    pass
                if time.time() > explore_deadline:
                    if val < best:
                        best = val
                        x_best = Xo.copy()
                    break
            if val < best:
                best = val
                x_best = Xo.copy()
    time_left = TIME_LIMIT - (time.time() - t0) - 1.0
    time_left = max(5.0, min(38.0, time_left))
    try:
        x_best = _lp_polish(x_best, time_left)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return x_best

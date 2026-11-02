import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None, deadline=1e18):
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if t0 is not None and time.time() - t0 > deadline:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_quench(x, steps=500, beta0=80.0, beta1=600.0, lr0=0.03, lr1=0.005):
    x = x.copy()
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5 + 1e-300
        grad = (W @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        # clip extreme steps
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # limit per-point step to ~0.1
        scale = np.minimum(1.0, 3.0 / np.maximum(gn, 1e-12))
        grad = grad * scale
        x = x - lr * grad
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # load starting points
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (58, 22):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((58, 22))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best = float(vals[bi])

    # helper to get contacts / worst points of x_best
    def _hub_order(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # contacts within 2e-4 for hub detection (broader than exact)
        cnt = (G > m - 2e-4).sum(axis=1)
        pmax = G.max(axis=1)
        order = np.argsort(-cnt - 0.01 * (pmax - m + 1))
        return order, cnt, pmax, m

    trial = 0
    # explore until ~115s, reserve rest for final polish
    while time.time() - t0 < 110 and trial < 60:
        if time.time() - t0 > 165:
            break
        order, cnt, pmax, m = _hub_order(x_best)
        mode = trial % 5
        y = x_best.copy()
        if mode == 0:
            sig = 0.02 + 0.01 * (trial % 3)
            y = y + sig * rng.standard_normal(y.shape)
        elif mode == 1:
            # shake overloaded hubs
            k = 4 + (trial % 3)
            idx = order[:k]
            sig = 0.12 + 0.04 * ((trial // 5) % 3)
            y[idx] += sig * rng.standard_normal((len(idx), y.shape[1]))
        elif mode == 2:
            # single worst-point hole reinsertion
            worst = int(order[0])
            # sample holes, keep best
            S = 2000
            cand = rng.standard_normal((S, y.shape[1]))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            rest = np.delete(y, worst, axis=0)
            dots = cand @ rest.T
            md = dots.max(axis=1)
            pick = int(np.argmin(md))
            y[worst] = cand[pick]
            # small noise on neighbours
            G = y @ y.T
            nbr = np.argsort(-G[worst])[:6]
            y[nbr] += 0.02 * rng.standard_normal((len(nbr), y.shape[1]))
        elif mode == 3:
            sig = 0.06 + 0.02 * ((trial // 5) % 2)
            y = y + sig * rng.standard_normal(y.shape)
        else:
            # random subset reinit + hole pick
            k = 2 + (trial % 2)
            idx = rng.choice(y.shape[0], size=k, replace=False)
            for ii in idx:
                S = 800
                cand = rng.standard_normal((S, y.shape[1]))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                rest = np.delete(y, ii, axis=0)
                dots = cand @ rest.T
                md = dots.max(axis=1)
                pick = int(np.argmin(md))
                y[ii] = cand[pick]
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        # fast smooth quench
        steps = 450 if mode in (0, 3) else 550
        try:
            y = _smooth_quench(y, steps=steps)
        except Exception:
            trial += 1
            continue
        # short LP repair (2s)
        remain = 112 - (time.time() - t0)
        if remain < 2.5:
            # no time for LP, just evaluate quenched
            v = _max_cos(y)
            if v < best:
                x_best, best = y, v
            trial += 1
            continue
        try:
            y2 = _lp_polish(y, seconds=2.2, t0=t0, deadline=112)
        except Exception:
            trial += 1
            continue
        v = _max_cos(y2)
        if v < best:
            x_best, best = y2, v
        trial += 1

    # final long polish of best
    remaining = 168 - (time.time() - t0)
    if remaining > 4:
        try:
            x_best = _lp_polish(x_best, seconds=min(remaining - 1, 40),
                                t0=t0, deadline=169)
        except Exception:
            pass
    x_best /= np.maximum(np.linalg.norm(x_best, axis=1, keepdims=True), 1e-12)
    return x_best

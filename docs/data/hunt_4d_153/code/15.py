import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 2e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            idx = np.argpartition(-csel, 2500)[:2500]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu0, Ju0 = iu
    lb0 = np.log(lr0); lb1 = np.log(lr1)
    gb0 = np.log(beta0); gb1 = np.log(beta1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(gb0 + (gb1 - gb0) * f)
        lr = np.exp(lb0 + (lb1 - lb0) * f)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gm = g.max()
        # stabilize, cap exponent
        e = np.exp(np.clip(beta * (g - gm), -30, 0.0))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _greedy_insert(x, idxs, rng, n_try=1500):
    n = x.shape[0]
    for idx in idxs:
        cand = rng.normal(size=(n_try, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x, idx, axis=0)
        dots = others @ cand.T  # (n-1, n_try)
        worst = dots.max(axis=0)
        best = np.argmin(worst)
        x[idx] = cand[best]
    return x

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu0, Ju0 = iu
    # population of alternates for diversity
    pool = [(best_val, x_best.copy())]
    it = 0
    while time.time() - t0 < HARD - 12:
        it += 1
        remain = HARD - 12 - (time.time() - t0)
        if remain < 3:
            break
        # pick base: mostly best, sometimes 2nd
        if len(pool) > 1 and (it % 5 == 4):
            base = pool[1][1]
        else:
            base = x_best
        G = base @ base.T
        gall = G[iu]
        mx = gall.max()
        # per-point worst and contacts
        Gw = G.copy()
        np.fill_diagonal(Gw, -2)
        pworst = Gw.max(axis=1)
        order_worst = np.argsort(-pworst)
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        ratt = np.where(cnt <= 1)[0]
        hubs = np.where(cnt >= 7)[0]
        y = base.copy()
        mode = it % 8
        if mode == 0:
            y += rng.normal(0, 0.015, size=y.shape)
        elif mode == 1:
            y += rng.normal(0, 0.035, size=y.shape)
            k = 1
            _greedy_insert(y, order_worst[:k], rng, n_try=2000)
        elif mode == 2:
            y += rng.normal(0, 0.06, size=y.shape)
            k = 2
            sel = np.concatenate([order_worst[:k], ratt[:1]]) if len(ratt) else order_worst[:k]
            sel = np.unique(sel)[:3]
            _greedy_insert(y, sel, rng, n_try=2000)
        elif mode == 3:
            y += rng.normal(0, 0.02, size=y.shape)
            # reinsert single worst with huge search
            _greedy_insert(y, order_worst[:1], rng, n_try=4000)
            y += rng.normal(0, 0.008, size=y.shape)
        elif mode == 4:
            y += rng.normal(0, 0.10, size=y.shape)
            if len(ratt):
                _greedy_insert(y, ratt[:2], rng, n_try=1500)
        elif mode == 5:
            # random subset full randomize then greedy
            nr = 4 + int(rng.integers(0, 3))
            pick = np.concatenate([order_worst[:nr], rng.choice(n, size=2, replace=False)])
            pick = np.unique(pick)
            for idx in pick:
                if rng.random() < 0.5:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] += rng.normal(0, 0.3, size=4)
            # greedy fix one of them
            _greedy_insert(y, pick[:1], rng, n_try=1500)
        elif mode == 6:
            y += rng.normal(0, 0.05, size=y.shape)
            if len(hubs):
                hb = hubs[rng.choice(len(hubs), size=min(2, len(hubs)), replace=False)]
                _greedy_insert(y, hb, rng, n_try=2000)
        else:
            y += rng.normal(0, 0.025, size=y.shape)
            # kick neighbors of current max pair
            mi = np.argmax(gall)
            a, b = Iu0[mi], Ju0[mi]
            y[a] += rng.normal(0, 0.15, size=4)
            y[b] += rng.normal(0, 0.15, size=4)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed smooth: length adapts to remaining time
        if remain > 60:
            steps = 1200
        elif remain > 25:
            steps = 800
        else:
            steps = 500
        # vary beta schedule by mode
        if mode in (4, 5):
            beta0, beta1 = 80, 2500
            lr0, lr1 = 0.02, 0.001
        else:
            beta0, beta1 = 150, 4000
            lr0, lr1 = 0.012, 0.0008
        y = _smooth(y, steps, beta0, beta1, lr0, lr1, iu)
        v = _max_cos(y)
        # accept to pool
        if v < best_val + 0.004:
            # short LP polish
            bud = 2.5 if remain > 40 else 1.2
            if time.time() - t0 + bud > HARD - 10:
                bud = max(0.5, HARD - 10 - (time.time() - t0))
            yp, vp = _lp_polish(y, bud)
            y, v = yp, vp
        if v < best_val - 1e-12:
            best_val = v
            x_best = y.copy()
            pool = [(best_val, x_best.copy())] + pool[:2]
        else:
            # keep diversity pool (best 3 distinct-ish)
            if len(pool) < 4 and v < best_val + 0.003:
                pool.append((v, y.copy()))
                pool.sort(key=lambda t: t[0])
                pool = pool[:4]
        # timeout guard
        if time.time() - t0 >= HARD - 11:
            break
    # final snap on incumbent with leftover time
    left = HARD - (time.time() - t0)
    if left > 1.0:
        x_best, best_val = _lp_polish(x_best, min(left, 8.0))
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

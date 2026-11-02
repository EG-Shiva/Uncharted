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
    radius = 3e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n), dtype=x.dtype)
        W[Iu, Ju] = w
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

def _pick_voids(y_keep, k, rng, thresh=0.85):
    # y_keep: (m,4) existing points, return k far voids
    cand = rng.normal(size=(5000, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    if len(y_keep) == 0:
        return cand[:k]
    dots = cand @ y_keep.T
    score = dots.max(axis=1)
    order = np.argsort(score)
    chosen = []
    for idx in order:
        c = cand[idx]
        ok = True
        for q in chosen:
            if float(c @ q) > thresh:
                ok = False
                break
        if ok:
            chosen.append(c)
            if len(chosen) >= k:
                break
    while len(chosen) < k:
        v = rng.normal(size=4); v /= np.linalg.norm(v)
        chosen.append(v)
    return np.array(chosen)

def solve():
    t0 = time.time()
    HARD_END = 177.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu

    # short snap of starting point
    if time.time() - t0 < HARD_END - 20:
        x_best, best_val = _lp_polish(x_best, 2.0)

    it = 0
    while time.time() - t0 < 150.0:
        it += 1
        if time.time() - t0 > HARD_END - 25:
            break
        mode = it % 8
        if mode == 1:
            sigma, krep, steps, b0, b1, l0, l1 = 0.015, 0, 500, 30.0, 350.0, 0.02, 0.002
        elif mode == 2:
            sigma, krep, steps, b0, b1, l0, l1 = 0.03, 1, 550, 25.0, 400.0, 0.02, 0.002
        elif mode == 3:
            sigma, krep, steps, b0, b1, l0, l1 = 0.06, 2, 600, 20.0, 450.0, 0.025, 0.002
        elif mode == 4:
            sigma, krep, steps, b0, b1, l0, l1 = 0.02, 2, 550, 40.0, 500.0, 0.015, 0.0015
        elif mode == 5:
            sigma, krep, steps, b0, b1, l0, l1 = 0.12, 2, 600, 15.0, 300.0, 0.03, 0.003
        elif mode == 6:
            sigma, krep, steps, b0, b1, l0, l1 = 0.05, 3, 650, 20.0, 450.0, 0.02, 0.002
        elif mode == 7:
            sigma, krep, steps, b0, b1, l0, l1 = 0.15, 1, 500, 15.0, 250.0, 0.03, 0.003
        else:
            sigma, krep, steps, b0, b1, l0, l1 = 0.025, 4, 650, 25.0, 500.0, 0.02, 0.0015

        y = x_best + rng.normal(0, sigma, size=(n, d))

        if krep > 0:
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            per = G0.max(axis=1)
            # contact count for hub detection
            mx0 = G0.max()
            tight = G0 > mx0 - 1e-4
            cnt = tight.sum(axis=1)
            # score: per-point max + hub bonus + random
            score = per + 0.002 * cnt + rng.normal(0, 0.002, size=n)
            # never victimize the rattler (min per) with high prob: keep it
            victims = np.argsort(-score)[:krep + 3]
            # random subset to diversify
            victims = rng.choice(victims, size=krep, replace=False)
            # normalize y before void computation
            yn = y / np.linalg.norm(y, axis=1, keepdims=True)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            voids = _pick_voids(yn[keep_mask], krep, rng)
            y[victims] = voids + rng.normal(0, 0.005, size=(krep, d))

        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _smooth(y, steps, b0, b1, l0, l1, Iu, Ju)
        v = _max_cos(y)
        # LP polish promising candidates
        do_lp = (v < best_val + 0.002) and (v < best_val or (it % 2 == 0))
        if do_lp:
            if time.time() - t0 > HARD_END - 25:
                # no time for LP, just check raw
                if v < best_val:
                    x_best, best_val = y, v
                continue
            yp, vp = _lp_polish(y, 1.2)
            if vp < best_val:
                x_best, best_val = yp, vp
            elif v < best_val:
                x_best, best_val = y, v
        else:
            if v < best_val:
                # still need LP to lock it but accept raw now
                x_best, best_val = y, v

    # final polish with remaining time
    rem = HARD_END - (time.time() - t0)
    if rem > 1.0:
        x_best, best_val = _lp_polish(x_best, min(rem, 15.0))
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

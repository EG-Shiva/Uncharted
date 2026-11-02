import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - thr
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return m, cnt, iu, g

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
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
        if time.time() - start >= budget:
            break
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu0, Ju0 = iu
    lb0, lb1 = np.log(beta0), np.log(beta1)
    lr_a, lr_b = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * f)
        lr = np.exp(lr_a + (lr_b - lr_a) * f)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
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
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _void_candidates(x_keep, m, rng):
    # sample random points on S^3, return those with smallest max-dot
    cand = rng.normal(size=(m * 8, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    md = dots.max(axis=1)
    idx = np.argpartition(md, m)[:m]
    return cand[idx]

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    # current walker for basin hopping
    cur_x = x_best.copy()
    cur_val = best_val
    it = 0
    # schedules
    while time.time() - t0 < 138.0:
        it += 1
        remain = HARD - (time.time() - t0)
        if remain < 32:
            break
        # analyse current
        Gm = cur_x @ cur_x.T
        gall = Gm[iu]
        mx = gall.max()
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        order = np.argsort(-cnt)
        ratt = np.where(cnt <= 1)[0]
        hubs = order[:6]
        mode = it % 7
        y = cur_x.copy()
        if mode == 0:
            y += rng.normal(0, 0.012, size=y.shape)
        elif mode == 1:
            y += rng.normal(0, 0.04, size=y.shape)
            # reinsert rattlers at voids
            if len(ratt) > 0:
                keep = np.ones(n, dtype=bool); keep[ratt[:3]] = False
                vc = _void_candidates(y[keep], min(len(ratt[:3]), 3), rng)
                for k, idx in enumerate(ratt[:3]):
                    if k < len(vc):
                        y[idx] = vc[k] + rng.normal(0, 0.02, size=4)
        elif mode == 2:
            y += rng.normal(0, 0.10, size=y.shape)
        elif mode == 3:
            # destroy top hubs
            nh = 2 + (it % 2)
            victims = hubs[:nh].copy()
            # also add random neighbours of hubs
            if rng.random() < 0.5 and len(victims) > 0:
                # pick closest to first hub
                dists = (y @ y[victims[0]])
                dists[victims[0]] = -2
                extra = np.argsort(-dists)[:1]
                victims = np.unique(np.concatenate([victims, extra]))
            keep = np.ones(n, dtype=bool); keep[victims] = False
            vc = _void_candidates(y[keep] / np.linalg.norm(y[keep], axis=1, keepdims=True), len(victims), rng)
            for k, idx in enumerate(victims):
                y[idx] = vc[k % len(vc)]
            y += rng.normal(0, 0.02, size=y.shape)
        elif mode == 4:
            y += rng.normal(0, 0.05, size=y.shape)
            # random cap rotation: pick random direction, perturb cap
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            proj = y @ u
            cap = np.where(proj > 0.55)[0]
            if len(cap) > 0:
                y[cap] += rng.normal(0, 0.12, size=(len(cap), d))
        elif mode == 5:
            y += rng.normal(0, 0.22, size=y.shape)
        else:
            # mass replacement of random subset
            k = 8
            victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            nk = y[keep] / np.linalg.norm(y[keep], axis=1, keepdims=True)
            vc = _void_candidates(nk, k, rng)
            for kk, idx in enumerate(victims):
                if rng.random() < 0.8:
                    y[idx] = vc[kk]
                else:
                    y[idx] = rng.normal(size=4)
            y += rng.normal(0, 0.015, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # fast smooth anneal
        if mode in (2, 5):
            steps, b0, b1, l0, l1 = 500, 80.0, 1500.0, 0.004, 0.0004
        elif mode == 3:
            steps, b0, b1, l0, l1 = 600, 60.0, 2000.0, 0.005, 0.0003
        else:
            steps, b0, b1, l0, l1 = 450, 60.0, 1200.0, 0.003, 0.0004
        y_s = _smooth_opt(y, steps, b0, b1, l0, l1, iu)
        v_s = _max_cos(y_s)
        # decide if worth LP polishing
        thresh = min(cur_val + 0.004, best_val + 0.0025)
        # periodically polish even worse to move walker
        do_lp = (v_s < thresh) or (it % 12 == 0 and v_s < cur_val + 0.008)
        if do_lp:
            bud = 1.6 if v_s > best_val else 2.5
            # cap budget near end
            if remain - bud < 28:
                bud = max(0.6, remain - 28)
            y_lp, v_lp = _lp_polish(y_s, bud)
        else:
            y_lp, v_lp = y_s, v_s
        # acceptance: greedy for best, tempered for walker
        if v_lp < best_val:
            x_best, best_val = y_lp.copy(), float(v_lp)
            cur_x, cur_val = y_lp.copy(), float(v_lp)
        elif v_lp < cur_val - 1e-12:
            cur_x, cur_val = y_lp.copy(), float(v_lp)
        elif v_lp < cur_val + 0.002 and rng.random() < 0.25:
            cur_x, cur_val = y_lp.copy(), float(v_lp)
        # restart walker to best if stuck
        if it % 25 == 0:
            cur_x = x_best.copy(); cur_val = best_val
            # kick best slightly to new region
            cur_x += rng.normal(0, 0.03, size=cur_x.shape)
            cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
            cur_val = _max_cos(cur_x)
        # early stop if big win? continue to squeeze
    # final squeeze of best
    fb = max(4.0, HARD - (time.time() - t0) - 2.0)
    fb = min(fb, 25.0)
    if fb > 0.5:
        x_best, best_val = _lp_polish(x_best, fb)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

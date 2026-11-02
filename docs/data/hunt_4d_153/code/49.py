import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return cnt, float(m), iu, g

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

def _anneal(x0, moved_idx, t_target, steps, beta0, beta1, lr0, lr1):
    n, d = x0.shape
    x = x0.copy()
    moved = np.zeros(n, dtype=bool)
    moved[np.asarray(moved_idx, dtype=int)] = True
    all_free = bool(moved.all())
    # Adam state only for moved
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    beb = np.log(beta1 / beta0) / max(steps - 1, 1)
    # involved mask upper
    iu0 = np.triu_indices(n, 1)
    if all_free:
        invol = np.ones(len(iu0[0]), dtype=bool)
    else:
        invol = moved[iu0[0]] | moved[iu0[1]]
    Iu = iu0[0][invol]; Ju = iu0[1][invol]
    for s in range(steps):
        beta = beta0 * np.exp(beb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju] - t_target
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        w = e / max(Z, 1e-300)
        W = np.zeros((n, n))
        W[Iu, Ju] = w
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
        if all_free:
            x = x - lr * step
        else:
            x[moved] = x[moved] - lr * step[moved]
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _void_refill(cur, rng, n_cand=20000, n_refine=250):
    # cur: (m,4) frozen+already inserted, find point minimizing max cosine
    m = cur.shape[0]
    cand = rng.normal(size=(n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ cur.T
    mx = dots.max(axis=1)
    bi = int(mx.argmin())
    best = cand[bi].copy()
    bval = float(mx[bi])
    sig = 0.05
    for k in range(n_refine):
        sig_k = sig * (1.0 - k / n_refine) + 0.002 * (k / n_refine)
        prop = best + rng.normal(0, sig_k, size=4)
        prop /= np.linalg.norm(prop)
        v = float((prop @ cur.T).max())
        if v < bval:
            bval = v
            best = prop
    return best, bval

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
    RECORD = 0.84910235828
    patch_cycle = [8, 12, 16]
    delta_cycle = [1e-4, 3e-4, 1e-3]
    trial = 0
    # main patch loop
    while time.time() - t0 < 132.0:
        trial += 1
        if time.time() - t0 > HARD - 32:
            break
        cnt, mx, iu_all, gall = _contacts(x_best, tol=1e-7)
        ratt = np.where(cnt == 0)[0]
        # choose centre: rattler first, then random, then hub
        if trial == 1 and len(ratt) > 0:
            centre = int(ratt[0])
        elif trial % 3 == 1 and len(ratt) > 0:
            centre = int(ratt[(trial // 3) % len(ratt)])
        elif trial % 3 == 2:
            centre = int(rng.integers(n))
        else:
            # highest-contact hub (random among top)
            order = np.argsort(-cnt, kind="stable")
            top = order[:8]
            centre = int(top[rng.integers(len(top))])
        k = patch_cycle[(trial - 1) % len(patch_cycle)]
        delta = delta_cycle[(trial - 1) % len(delta_cycle)]
        # neighbours: largest cosine to centre
        sims = x_best @ x_best[centre]
        sims[centre] = -2.0
        neigh = np.argsort(-sims, kind="stable")[:k-1]
        moved = np.concatenate(([centre], neigh)).astype(int)
        moved_set = set(moved.tolist())
        remain = np.array([i for i in range(n) if i not in moved_set], dtype=int)
        # frozen current set
        cur = x_best[remain].copy()
        new_pts = []
        for s in range(k):
            bp, _ = _void_refill(cur, rng, n_cand=20000, n_refine=250)
            new_pts.append(bp)
            cur = np.vstack([cur, bp.reshape(1, 4)])
        y = np.empty_like(x_best)
        y[remain] = x_best[remain]
        for idx, mi in enumerate(moved):
            y[mi] = new_pts[idx]
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        t_target = RECORD - delta
        # optimize only reinserted points
        y2 = _anneal(y, moved, t_target, steps=550, beta0=50.0, beta1=800.0, lr0=0.008, lr1=0.0008)
        # short all-free anneal
        y2 = _anneal(y2, np.arange(n), t_target, steps=300, beta0=80.0, beta1=1200.0, lr0=0.003, lr1=0.0005)
        # LP finish
        remain_budget = HARD - 10.0 - (time.time() - t0)
        lp_b = min(3.5, max(1.0, remain_budget / max(1, (14 - trial + 1))))
        if lp_b < 0.8:
            lp_b = 0.8
        y3, v3 = _lp_polish(y2, lp_b)
        if v3 < best_val:
            x_best, best_val = y3, v3
            continue
        # keep near-miss for extra all-free anneal (changed graph event)
        if v3 < best_val + 0.002:
            if time.time() - t0 > HARD - 32:
                continue
            y4 = _anneal(y3, np.arange(n), t_target - 2e-4, steps=600, beta0=100.0, beta1=1500.0, lr0=0.002, lr1=0.0003)
            y5, v5 = _lp_polish(y4, min(3.0, max(0.8, HARD - 10.0 - (time.time() - t0))))
            if v5 < best_val:
                x_best, best_val = y5, v5
        # occasional extra random-hub trial without extra cost already covered
    # final polish with remaining time
    left = HARD - (time.time() - t0)
    if left > 1.0:
        x_best, best_val = _lp_polish(x_best, min(left - 0.5, 12.0))
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contacts(x, thr=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    sel = g > m - thr
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][sel], 1)
    np.add.at(cnt, iu[1][sel], 1)
    return cnt, float(m), iu, sel

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

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, freeze=None):
    n, d = x0.shape
    x = x0.copy()
    iu0 = np.triu_indices(n, 1)
    Iu, Ju = iu0
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    fr = np.zeros(n, dtype=bool) if freeze is None else freeze
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        if fr.any():
            gt[fr] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        if fr.any():
            step[fr] = 0.0
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _best_insertion(X, rng, ntry=2500):
    # find point minimizing max dot to X
    best_v = None
    best_f = 1e9
    cand = rng.normal(size=(ntry, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X.T
    f = dots.max(axis=1)
    k = int(np.argmin(f))
    v = cand[k].copy()
    fv = float(f[k])
    # local tangential descent on soft-max to X
    beta = 60.0
    lr = 0.02
    m1 = np.zeros(4); vv = np.zeros(4)
    for s in range(300):
        d = X @ v
        dm = d.max()
        e = np.exp(beta * (d - dm))
        w = e / e.sum()
        grad = w @ X
        gt = grad - (grad @ v) * v
        m1 = 0.9 * m1 + 0.1 * gt
        vv = 0.999 * vv + 0.001 * (gt * gt)
        mhat = m1 / (1 - 0.9 ** (s + 1))
        vhat = vv / (1 - 0.999 ** (s + 1))
        st = mhat / (np.sqrt(vhat) + 1e-8)
        v = v - lr * st
        v /= np.linalg.norm(v)
        beta = min(beta * 1.005, 400.0)
        lr *= 0.998
    f2 = float(np.max(X @ v))
    if f2 < fv:
        return v, f2
    return cand[k], fv

def solve():
    t0 = time.time()
    HARD = 171.0
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

    # quick diverse random basins to seed alternative topology (cheap, no LP unless great)
    for r in range(4):
        if time.time() - t0 > 25:
            break
        y = rng.normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y2 = _smooth_opt(y, 700, 30.0, 300.0, 0.008, 0.0005)
        v2 = _max_cos(y2)
        if v2 < best_val - 1e-12:
            yp, vp = _lp_polish(y2, 2.0)
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp

    it = 0
    while time.time() - t0 < HARD - 32.0:
        it += 1
        if time.time() - t0 > HARD - 32:
            break
        cnt, mx, iu_full, _ = _contacts(x_best, thr=1e-7)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 7
        y = x_best.copy()
        if mode == 0:
            # cap excavation + resample
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            k = int(rng.integers(6, 15))
            vic = np.argsort(-dots)[:k]
            if len(ratt) > 0 and rng.random() < 0.7:
                vic[:min(len(vic), len(ratt))] = ratt[:min(len(vic), len(ratt))]
                # ensure unique
                vic = np.unique(vic)
                while len(vic) < k:
                    vic = np.append(vic, rng.integers(0, n))
                    vic = np.unique(vic)
                vic = vic[:k]
            y[vic] = rng.normal(size=(len(vic), 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # two-stage: freeze core first
            freeze = np.ones(n, dtype=bool); freeze[vic] = False
            y = _smooth_opt(y, 350, 20.0, 150.0, 0.01, 0.001, freeze=freeze)
            y = _smooth_opt(y, 900, 40.0, 320.0, 0.006, 0.0004)
        elif mode == 1:
            # drop-core-recreate: drop rattlers + random, optimise core, reinsert at holes
            nd = int(rng.integers(1, 4))
            drop = list(ratt[:min(len(ratt), nd)])
            while len(drop) < nd:
                drop.append(int(rng.integers(0, n)))
            drop = np.array(sorted(set(drop)), dtype=int)
            keep = np.array([i for i in range(n) if i not in set(drop.tolist())])
            core = y[keep].copy()
            # optimise core alone (fewer points -> can open hole)
            core = _smooth_opt(core, 800, 35.0, 300.0, 0.007, 0.0004)
            # reinsert one by one at hole centres
            cur = core
            for _d in range(len(drop)):
                v, _f = _best_insertion(cur, rng, ntry=2000)
                # jitter slightly
                v = v + rng.normal(0, 0.01, size=4)
                v /= np.linalg.norm(v)
                cur = np.vstack([cur, v[None, :]])
            # map back: keep order but replace all (topology changed anyway)
            # need 153 rows: cur has 153 rows
            y = cur.copy()
            # joint anneal
            y = _smooth_opt(y, 800, 40.0, 300.0, 0.005, 0.0004)
        elif mode == 2:
            # rigid rotation of a cap (collective move preserving local graph)
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            k = int(rng.integers(8, 20))
            vic = np.argsort(-dots)[:k]
            M = np.eye(4) + 0.6 * rng.normal(size=(4, 4))
            Q, Rr = np.linalg.qr(M)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            y[vic] = (Q @ y[vic].T).T
            y += rng.normal(0, 0.02, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 1000, 35.0, 350.0, 0.006, 0.0004)
        elif mode == 3:
            # hub-targeted dispersal: push hubs apart with extra noise
            y += rng.normal(0, 0.035, size=y.shape)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.08, size=(len(hubs), d))
            if len(ratt) > 0:
                for idx in ratt:
                    y[idx] = rng.normal(size=4)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 1100, 40.0, 280.0, 0.006, 0.0004)
        elif mode == 4:
            # large global kick + long anneal (basin escape)
            y += rng.normal(0, 0.10, size=y.shape)
            rep = rng.choice(n, size=2, replace=False)
            y[rep] = rng.normal(size=(2, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 1100, 25.0, 350.0, 0.007, 0.0004)
        elif mode == 5:
            # antipodal shuffle: reflect a cap through origin-ish to change parity
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            k = int(rng.integers(4, 10))
            vic = np.argsort(-dots)[:k]
            # random O(4) reshuffle of victims among themselves + noise
            perm = rng.permutation(len(vic))
            y[vic] = y[vic[perm]] + rng.normal(0, 0.05, size=(len(vic), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 1000, 30.0, 300.0, 0.006, 0.0004)
        else:
            # small kick local search
            y += rng.normal(0, 0.015, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 900, 50.0, 300.0, 0.004, 0.0003)
        v2 = _max_cos(y)
        thresh = best_val + (0.006 if mode in (0, 1, 2, 4) else 0.002)
        if v2 < thresh:
            remain = HARD - (time.time() - t0) - 24.0
            if remain < 1.0:
                # no time for LP, just greedy accept
                if v2 < best_val - 1e-12:
                    x_best, best_val = y, v2
                continue
            b = min(3.0, max(1.0, remain / 10.0))
            # less LP for exploratory modes unless very promising
            if mode in (4,) and v2 > best_val + 0.002:
                b = min(b, 1.2)
            yp, vp = _lp_polish(y, b)
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
            elif vp < best_val + 5e-7 and rng.random() < 0.03:
                # sideways drift to new basin occasionally
                x_best = yp
        if time.time() - t0 > HARD - 32:
            break
    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        remain = min(remain, 28.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    if steps <= 1:
        lrb = 0.0; betb = 0.0
    else:
        lrb = np.log(lr1 / lr0) / (steps - 1)
        betb = np.log(beta1 / beta0) / (steps - 1)
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n), dtype=float)
        W[Iu, Ju] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _lp_polish(x, budget):
    t0 = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        if time.time() - t0 >= budget:
            break
    return x, best

def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        xb = xb.reshape(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    cur = xb.copy()
    cur_val = best_val
    cand_list = [(best_val, xb.copy())]
    # exploration: fixed number of proposals, time-guarded
    NPROP = 70
    temp0 = 0.004
    temp1 = 0.00015
    for it in range(NPROP):
        if time.time() - t0 > 118.0:
            break
        frac = it / max(NPROP - 1, 1)
        Temp = temp0 * (temp1 / temp0) ** frac
        mode = it % 7
        if mode == 0:
            x0 = rng.normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1400, 12.0, 450.0, 0.04, 0.002
        elif mode == 1:
            sigma = 0.5
            x0 = cur + rng.normal(0, sigma, size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1100, 25.0, 550.0, 0.03, 0.002
        elif mode == 2:
            sigma = 0.2
            x0 = x_best + rng.normal(0, sigma, size=(n, d))
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            x0[idx] = rng.normal(size=(k, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1100, 25.0, 500.0, 0.03, 0.002
        elif mode == 3:
            # cap surgery: replace top cap along random direction
            v = rng.normal(size=d); v /= np.linalg.norm(v)
            dots = cur @ v
            order = np.argsort(dots)
            k = 12 + int(rng.integers(0, 10))
            mask_idx = order[-k:]
            x0 = cur.copy()
            x0[mask_idx] = rng.normal(size=(k, d))
            x0 += rng.normal(0, 0.08, size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1100, 20.0, 500.0, 0.03, 0.002
        elif mode == 4:
            sigma = 0.28
            x0 = cur + rng.normal(0, sigma, size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1000, 30.0, 600.0, 0.025, 0.0015
        elif mode == 5:
            # hub disruption: find high-contact points of cur
            Gc = cur @ cur.T
            gall = Gc[Iu, Ju]
            mx = gall.max()
            tight = gall > mx - 5e-4
            cnt = np.zeros(n, dtype=int)
            np.add.at(cnt, Iu[tight], 1)
            np.add.at(cnt, Ju[tight], 1)
            hubs = np.where(cnt >= 6)[0]
            x0 = cur.copy() + rng.normal(0, 0.12, size=(n, d))
            if len(hubs) > 0:
                x0[hubs] += rng.normal(0, 0.35, size=(hubs.shape[0], d))
            # also relocate low-contact points
            lows = np.where(cnt <= 3)[0]
            if len(lows) > 0:
                for j in lows:
                    if rng.random() < 0.5:
                        x0[j] = rng.normal(size=d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1100, 20.0, 550.0, 0.03, 0.002
        else:
            # random orthogonal remix of subset: shuffle + noise from best
            x0 = x_best.copy() + rng.normal(0, 0.35, size=(n, d))
            # random global rotation to diversify (does not change value alone but combined with noise does)
            A = rng.normal(size=(d, d))
            Q, _ = np.linalg.qr(A)
            if rng.random() < 0.5:
                x0 = x0 @ Q
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, beta0, beta1, lr0, lr1 = 1100, 25.0, 500.0, 0.03, 0.002
        y = _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju)
        y_val = _max_cos(y)
        if y_val < cur_val or rng.random() < np.exp((cur_val - y_val) / max(Temp, 1e-12)):
            cur = y.copy()
            cur_val = y_val
        if y_val < best_val - 1e-12:
            best_val = y_val
            x_best = y.copy()
            cand_list.append((y_val, y.copy()))
        elif y_val < best_val + 0.004 and len(cand_list) < 12:
            # keep a few diverse near-best for final polish
            if all(np.linalg.norm(y - c) > 1e-3 for _, c in cand_list):
                cand_list.append((y_val, y.copy()))
    # final polish: sort candidates and LP-polish top few within remaining time
    cand_list.sort(key=lambda t: t[0])
    # deduplicate to top 4
    top = cand_list[:4]
    # always include current best and cur
    remaining = 178.0 - (time.time() - t0)
    remaining = max(min(remaining, 55.0), 5.0)
    per = remaining / max(len(top), 1)
    per = min(per, 18.0)
    for _, xc in top:
        if time.time() - t0 > 172.0:
            break
        xp, vp = _lp_polish(xc, per)
        if vp < best_val:
            best_val = vp
            x_best = xp
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

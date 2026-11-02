import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, budget, r0=1.5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20.0 * radius, 1e-9)
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
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, Iu0, Ju0):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
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

def _void_sample(x, exclude, rng, ns=4000):
    # find point on S3 farthest from x (excluding index exclude)
    cand = rng.normal(size=(ns, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    mask = np.ones(x.shape[0], dtype=bool)
    if exclude is not None:
        mask[exclude] = False
    Xm = x[mask]
    # max cosine to set: lower is better (larger void)
    dots = cand @ Xm.T
    m = dots.max(axis=1)
    return cand[np.argmin(m)]

def solve():
    t0 = time.time()
    HARD = 171.0
    rng = np.random.default_rng(777)
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
    # pick better of best/record if both present
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float).reshape(153, 4)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu0, Ju0 = iu
    best_val = _max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best_val
    it = 0
    # short initial squeeze if time allows (helps baseline)
    # skip long squeeze, go straight to hopping for diversity
    while time.time() - t0 < HARD - 30.0:
        it += 1
        if time.time() - t0 > HARD - 32:
            break
        # contacts of current (for rattler/hub targeting, use best half the time)
        src = x_best if (it % 2 == 0) else x_cur
        G = src @ src.T
        gall = G[iu]
        mx = gall.max()
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 7
        if mode == 0:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.008, 0, 0, 40.0, 280.0, 850
        elif mode == 1:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.020, 1, 0, 35.0, 300.0, 900
        elif mode == 2:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.030, 2, 0, 25.0, 350.0, 950
        elif mode == 3:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.015, 1, 1, 40.0, 260.0, 850
        elif mode == 4:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.060, 2, 0, 22.0, 380.0, 1000
        elif mode == 5:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.100, 0, 2, 20.0, 320.0, 1000
        else:
            sigma, nrep_void, nrep_rand, b0, b1, steps = 0.040, 1, 1, 30.0, 300.0, 900
        # base: hop from cur (Metropolis) mostly, sometimes from best
        base = x_cur if rng.random() < 0.75 else x_best
        base_val = cur_val if np.may_share_memory(base, x_cur) or np.allclose(base, x_cur) else best_val
        # cheap identity check: use flag
        if rng.random() < 0.75:
            y = x_cur.copy()
            cur_from = True
        else:
            y = x_best.copy()
            cur_from = False
        y += rng.normal(0, sigma, size=y.shape)
        # rattler-targeted void reinsertion
        if nrep_void > 0:
            # prefer rattlers
            cand_idx = []
            if len(ratt) > 0:
                # take up to nrep_void rattlers at random
                rr = rng.choice(ratt, size=min(nrep_void, len(ratt)), replace=False)
                cand_idx.extend(rr.tolist())
            # fill remainder with random indices
            while len(cand_idx) < nrep_void:
                cand_idx.append(int(rng.integers(0, n)))
            for idx in cand_idx:
                # void-optimal replacement 80%, pure random 20%
                if rng.random() < 0.85:
                    y[idx] = _void_sample(y / np.linalg.norm(y, axis=1, keepdims=True), idx, rng, ns=3000)
                else:
                    y[idx] = rng.normal(size=4)
        if nrep_rand > 0:
            rep = rng.choice(n, size=nrep_rand, replace=False)
            for idx in rep:
                if rng.random() < 0.5:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] = _void_sample(y / np.linalg.norm(y, axis=1, keepdims=True), idx, rng, ns=3000)
        # extra hub jitter
        if len(hubs) > 0 and mode in (3, 4, 6):
            y[hubs] += rng.normal(0, sigma * 0.9, size=(len(hubs), d))
        # extra rattler big kick if many rattlers
        if len(ratt) > 0 and mode == 6:
            for idx in ratt[:2]:
                if rng.random() < 0.5:
                    y[idx] += rng.normal(0, 0.35, size=4)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y2 = _smooth_opt(y, steps, b0, b1, 0.006, 0.0004, Iu0, Ju0)
        v2 = _max_cos(y2)
        # decide LP budget
        thresh = best_val + (0.005 if mode in (4, 5) else 0.0025)
        do_lp = (v2 < thresh) or (v2 < cur_val + 0.0005 and rng.random() < 0.15)
        if do_lp:
            remain = HARD - (time.time() - t0) - 24.0
            if remain < 0.8:
                break
            b = min(2.5, max(1.0, remain / 10.0))
            # smaller budget for far worse
            if v2 > best_val + 0.002:
                b = min(b, 1.0)
            yp, vp = _lp_polish(y2, b)
            # update global best
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
                x_cur, cur_val = yp, vp
            else:
                # Metropolis acceptance for current
                T = 0.0006
                if vp < cur_val or rng.random() < np.exp(-(vp - cur_val) / T):
                    x_cur, cur_val = yp, vp
                # occasional sideways best replacement to escape plateau
                elif vp < best_val + 1e-12 and rng.random() < 0.02:
                    x_cur, cur_val = yp, vp
        else:
            # no LP: still allow cur drift if improved
            if v2 < cur_val:
                x_cur, cur_val = y2, v2
            elif v2 < best_val + 0.008 and rng.random() < 0.05:
                x_cur, cur_val = y2, v2
        if time.time() - t0 > HARD - 28:
            break
    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        remain = min(remain, 27.0)
        x_best, best_val = _lp_polish(x_best, remain, r0=1.2e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

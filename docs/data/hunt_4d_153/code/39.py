import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

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
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu, Ju = iu
    lb = np.log(beta1 / beta0) / max(steps - 1, 1)
    lr = np.log(lr1 / lr0) / max(steps - 1, 1)
    for s in range(steps):
        beta = beta0 * np.exp(lb * s)
        lr_h = lr0 * np.exp(lr * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = np.max(g)
        e = np.exp(beta * (g - gmax))
        Z = np.sum(e)
        w = e / Z
        W = np.zeros((n, n), dtype=x.dtype)
        W[Iu, Ju] = w
        W[Ju, Iu] = w
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr_h * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _holes(x, rng, ns=6000, k=4):
    # largest empty caps: sample, minimize max dot
    s = rng.normal(size=(ns, 4))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    D = s @ x.T  # ns x n
    md = D.max(axis=1)
    idx = np.argpartition(md, k)[:k]
    # order by md
    idx = idx[np.argsort(md[idx])]
    return s[idx]

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    x_cur = x_best.copy()
    cur_val = best_val
    T = 0.0012
    it = 0
    # main basin-hopping with collective moves
    while time.time() - t0 < 132.0:
        it += 1
        if time.time() - t0 > HARD - 40:
            break
        # analyse current walker contacts
        Gc = x_cur @ x_cur.T
        gall = Gc[iu]
        mx = float(gall.max())
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, Iu[tight], 1)
        np.add.at(cnt, Ju[tight], 1)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        # order hubs by count desc
        if len(hubs) > 0:
            hubs = hubs[np.argsort(-cnt[hubs])]
        mode = it % 7
        y = x_cur.copy()
        if mode == 0:
            # rattler-neighbor expansion
            sigma = 0.02
            y += rng.normal(0, sigma, size=y.shape)
            if len(ratt) > 0:
                r = ratt[0]
                dots = x_cur @ x_cur[r]
                nn = np.argsort(-dots)[1:11]
                alpha = 0.08 + 0.04 * rng.random()
                for j in nn:
                    y[j] = x_cur[j] - alpha * x_cur[r] + rng.normal(0, 0.02, size=4)
                # also jitter rattler
                if rng.random() < 0.7:
                    y[r] = rng.normal(size=4)
            steps, b0, b1, l0, l1 = 700, 80.0, 700.0, 0.008, 0.0008
        elif mode == 1:
            # hub relocation to largest holes
            y += rng.normal(0, 0.015, size=y.shape)
            try:
                hs = _holes(x_cur, rng, ns=6000, k=4)
            except Exception:
                hs = None
            if hs is not None and len(hubs) > 0:
                nh = min(len(hubs), 3, len(hs))
                for k2 in range(nh):
                    hidx = hubs[k2]
                    if rng.random() < 0.9:
                        y[hidx] = hs[k2 % len(hs)] + rng.normal(0, 0.02, size=4)
            steps, b0, b1, l0, l1 = 800, 100.0, 900.0, 0.008, 0.0008
        elif mode == 2:
            # random cap deletion / reinsertion
            c = rng.normal(size=4); c /= np.linalg.norm(c)
            dots = x_cur @ c
            mdel = int(rng.integers(5, 10))
            victims = np.argsort(-dots)[:mdel]
            y += rng.normal(0, 0.01, size=y.shape)
            for v in victims:
                if rng.random() < 0.5:
                    y[v] = rng.normal(size=4)
                else:
                    y[v] = c + rng.normal(0, 0.35, size=4)
            steps, b0, b1, l0, l1 = 900, 60.0, 800.0, 0.01, 0.001
        elif mode == 3:
            # medium global kick + rattler replacement
            sigma = 0.05
            y += rng.normal(0, sigma, size=y.shape)
            if len(ratt) > 0:
                for r in ratt:
                    if rng.random() < 0.7:
                        y[r] = rng.normal(size=4)
            steps, b0, b1, l0, l1 = 700, 120.0, 1000.0, 0.007, 0.0007
        elif mode == 4:
            # large kick to change topology
            sigma = 0.11
            y += rng.normal(0, sigma, size=y.shape)
            steps, b0, b1, l0, l1 = 900, 60.0, 600.0, 0.01, 0.001
        elif mode == 5:
            # directed hole drilling: push points out of random cap
            c = rng.normal(size=4); c /= np.linalg.norm(c)
            dots = x_cur @ c
            # points inside cap (dot > 0.85) pushed outward
            inside = np.where(dots > 0.80)[0]
            y += rng.normal(0, 0.012, size=y.shape)
            for j in inside:
                y[j] = x_cur[j] - 0.12 * c + rng.normal(0, 0.015, size=4)
            # place a hub into cap centre
            if len(hubs) > 0 and rng.random() < 0.8:
                y[hubs[0]] = c + rng.normal(0, 0.03, size=4)
            steps, b0, b1, l0, l1 = 800, 80.0, 800.0, 0.008, 0.0008
        else:
            # mild polish-wander around best
            base = x_best if rng.random() < 0.5 else x_cur
            y = base + rng.normal(0, 0.025, size=y.shape)
            steps, b0, b1, l0, l1 = 600, 150.0, 1200.0, 0.005, 0.0005
        # renormalize
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth optimize
        try:
            y = _smooth_opt(y, steps, b0, b1, l0, l1, iu)
        except Exception:
            continue
        v = _max_cos(y)
        # update best
        if v < best_val:
            x_best = y.copy()
            best_val = v
            x_cur = y.copy()
            cur_val = v
            T *= 0.97
        else:
            # Metropolis on walker to allow uphill to new basin
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / T):
                x_cur = y.copy()
                cur_val = v
            # occasional restart walker to best
            if rng.random() < 0.08:
                x_cur = x_best.copy()
                cur_val = best_val
        # cool temperature slowly
        T = max(T * 0.9995, 0.00025)
        # early exit if clearly record-beating with margin
        if best_val < 0.8485 and time.time() - t0 > 60:
            break
    # final squeeze: high-beta polish + LP
    remain = HARD - (time.time() - t0)
    if remain > 5:
        # high beta squeeze on best
        try:
            sq_steps = 1200 if remain > 25 else 500
            ysq = _smooth_opt(x_best, sq_steps, 800.0, 4000.0, 0.002, 0.0002, iu)
            vsq = _max_cos(ysq)
            if vsq < best_val:
                x_best, best_val = ysq, vsq
        except Exception:
            pass
    remain = HARD - (time.time() - t0)
    if remain > 8:
        try:
            x_best, best_val = _lp_polish(x_best, min(remain - 2, 28.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

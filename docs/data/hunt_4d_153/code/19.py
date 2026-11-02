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
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    Iu0, Ju0 = iu
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


def _plane_block_rotate(y, idx, theta, a, b):
    # rotate points idx coherently in plane span(a,b) by theta
    if len(idx) == 0:
        return y
    X = y[idx]
    xa = X @ a
    xb = X @ b
    c = np.cos(theta) - 1.0
    s = np.sin(theta)
    # x' = x + c*(xa*a + xb*b) + s*(xb*a - xa*b)
    y[idx] = X + c * (xa[:, None] * a[None, :] + xb[:, None] * b[None, :]) + s * (xb[:, None] * a[None, :] - xa[:, None] * b[None, :])
    return y


def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(987654321)
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
    x_cur = xb.copy()
    cur_val = best_val
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu0, Ju0 = iu
    it = 0
    Temp0 = 0.0012
    while time.time() - t0 < 138.0:
        it += 1
        if time.time() - t0 > HARD - 30:
            break
        # anneal temperature
        Temp = Temp0 * (1.0 - (time.time() - t0) / 160.0)
        Temp = max(Temp, 1e-4)
        # contacts of walker for targeted moves
        G = x_cur @ x_cur.T
        gall = G[iu]
        mx = gall.max()
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        ratt = np.where(cnt <= 1)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 8
        y = x_cur.copy()
        if mode == 0:
            # small global jitter
            y += rng.normal(0, 0.015, size=y.shape)
        elif mode == 1:
            # medium jitter + rattler reinsert
            y += rng.normal(0, 0.03, size=y.shape)
            if len(ratt) > 0:
                for idx in ratt:
                    if rng.random() < 0.7:
                        y[idx] = rng.normal(size=4)
                    else:
                        y[idx] += rng.normal(0, 0.3, size=4)
            else:
                rep = rng.choice(n, size=1, replace=False)
                y[rep] = rng.normal(size=(1, 4))
        elif mode == 2:
            # coherent block rotation of a cap (topology change, preserves interior)
            center = rng.normal(size=4)
            center /= np.linalg.norm(center)
            dots = y @ center
            k = int(rng.integers(8, 28))
            thr = np.sort(dots)[n - k]
            idx = np.where(dots >= thr)[0]
            # random plane
            A = rng.normal(size=4)
            B = rng.normal(size=4)
            A /= np.linalg.norm(A)
            B = B - (B @ A) * A
            nb = np.linalg.norm(B)
            if nb < 1e-9:
                B = rng.normal(size=4)
                B /= np.linalg.norm(B)
            else:
                B /= nb
            theta = rng.normal(0, 0.25)
            y = _plane_block_rotate(y, idx, theta, A, B)
            y += rng.normal(0, 0.01, size=y.shape)
        elif mode == 3:
            # dense-cluster surgery: pick worst hub + neighbors, scramble
            order = np.argsort(-cnt)
            h = order[0]
            # neighbors in tight graph
            nbset = set([h])
            # find tight pairs involving h
            ti = np.where(tight)[0]
            for t in ti:
                a0 = iu[0][t]; b0 = iu[1][t]
                if a0 == h:
                    nbset.add(b0)
                elif b0 == h:
                    nbset.add(a0)
                if len(nbset) >= 10:
                    break
            nbset = list(nbset)
            for idx in nbset:
                if rng.random() < 0.5:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] += rng.normal(0, 0.12, size=4)
            y += rng.normal(0, 0.008, size=y.shape)
        elif mode == 4:
            # large kick + 2 replacements
            y += rng.normal(0, 0.08, size=y.shape)
            rep = rng.choice(n, size=2, replace=False)
            y[rep] = rng.normal(size=(2, 4))
            if len(ratt) > 0:
                for idx in ratt:
                    if rng.random() < 0.6:
                        y[idx] = rng.normal(size=4)
        elif mode == 5:
            # pair split: push current max pairs apart explicitly then jitter
            # find top ~5 pairs
            k5 = min(5, len(gall))
            top = np.argpartition(-gall, k5 - 1)[:k5]
            for t in top:
                a0 = iu[0][t]; b0 = iu[1][t]
                eps = 0.02
                y[a0] = y[a0] - eps * y[b0]
                y[b0] = y[b0] - eps * y[a0]
            y += rng.normal(0, 0.02, size=y.shape)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.02, size=(len(hubs), d))
        elif mode == 6:
            # cap shift: translate a cap along tangent
            center = rng.normal(size=4)
            center /= np.linalg.norm(center)
            dots = y @ center
            k = int(rng.integers(12, 35))
            thr = np.sort(dots)[n - k]
            idx = np.where(dots >= thr)[0]
            shift = rng.normal(size=4)
            shift = shift - (shift @ center) * center
            shift /= (np.linalg.norm(shift) + 1e-12)
            mag = rng.uniform(0.03, 0.12)
            y[idx] += mag * shift[None, :]
            y += rng.normal(0, 0.012, size=y.shape)
        else:
            # random 3 replacements + moderate noise
            y += rng.normal(0, 0.05, size=y.shape)
            rep = rng.choice(n, size=3, replace=False)
            y[rep] = rng.normal(size=(3, 4))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth descent, vary schedule
        if mode in (2, 4, 6):
            beta0, beta1 = 25.0, 350.0
            steps = 900
        elif mode == 7:
            beta0, beta1 = 30.0, 400.0
            steps = 1000
        else:
            beta0, beta1 = 40.0, 280.0
            steps = 800
        # shorten steps when time low
        if time.time() - t0 > HARD - 55:
            steps = 500
        y2 = _smooth_opt(y, steps, beta0, beta1, 0.006, 0.0004, iu)
        v2 = _max_cos(y2)
        # decide LP polish
        thresh_cur = cur_val + (0.005 if mode in (2, 4, 7) else 0.002)
        thresh_best = best_val + (0.005 if mode in (2, 4, 7) else 0.002)
        do_lp = (v2 < thresh_cur) or (v2 < thresh_best)
        vp = v2
        yp = y2
        if do_lp:
            remain = HARD - (time.time() - t0) - 22.0
            if remain >= 1.0:
                b = min(2.5, max(0.8, remain / 10.0))
                # give less time to far candidates
                if v2 > best_val + 0.002:
                    b = min(b, 1.0)
                yp, vp = _lp_polish(y2, b)
        # acceptance: best always, walker with temperature
        if vp < best_val - 1e-12:
            x_best, best_val = yp, vp
            x_cur, cur_val = yp, vp
        elif vp < cur_val:
            x_cur, cur_val = yp, vp
        else:
            if rng.random() < np.exp(-(vp - cur_val) / Temp):
                x_cur, cur_val = yp, vp
            # occasional restart walker to best to avoid drift
            if it % 25 == 0 and rng.random() < 0.7:
                x_cur, cur_val = x_best.copy(), best_val
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 2.0:
        remain = min(remain, 28.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

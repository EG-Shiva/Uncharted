import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, budget, r0=1.2e-3):
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

def _contact_counts(x, tol=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    mx = g.max()
    tight = g > mx - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return cnt, mx

def _void_center(others, rng, ns=4000, refine=120):
    cand = rng.normal(size=(ns, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = dots.max(axis=1)
    best = cand[np.argmin(m)]
    # refine: move away from closest points (minimize max dot)
    v = best.copy()
    for it in range(refine):
        d = others @ v  # (m,)
        mx = d.max()
        # softmax weights on closest
        beta = 80.0 + it * 1.5
        e = np.exp(beta * (d - mx))
        w = e / e.sum()
        grad = w @ others  # direction to closest cluster
        # move opposite, tangent
        grad_t = grad - (grad @ v) * v
        lr = 0.03 / (1.0 + it * 0.03)
        v = v - lr * grad_t
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = cand[0]
        else:
            v = v / nv
    return v

def _cap_rotate(y, idx, rng, theta):
    # random plane rotation in R4
    A = rng.normal(size=(4, 2))
    Q, _ = np.linalg.qr(A)
    u = Q[:, 0]; v = Q[:, 1]
    c = np.cos(theta); s = np.sin(theta)
    Xc = y[idx]
    a = Xc @ u
    b = Xc @ v
    an = a * c - b * s
    bn = a * s + b * c
    Xn = Xc + np.outer(an - a, u) + np.outer(bn - b, v)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    y[idx] = Xn
    return y

def solve():
    t0 = time.time()
    HARD = 171.0
    rng = np.random.default_rng(2026)
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
    while time.time() - t0 < HARD - 26.0:
        it += 1
        elapsed = time.time() - t0
        if elapsed > HARD - 28:
            break
        # temperature anneals high->low to allow topology escape early
        T = 0.0016 * np.exp(-elapsed / 55.0) + 0.00018
        # choose source
        if rng.random() < 0.68:
            y = x_cur.copy()
            from_cur = True
        else:
            y = x_best.copy()
            from_cur = False
        src = x_cur if from_cur else x_best
        cnt, mx = _contact_counts(src, tol=1e-7)
        hubs = np.where(cnt >= 6)[0]
        ratt = np.where(cnt <= 2)[0]
        mode = it % 6
        if mode == 0:
            # hub-to-void transplant (core novel move)
            sigma, steps, b0, b1 = 0.012, 750, 35.0, 300.0
            y += rng.normal(0, sigma, size=y.shape)
            # pick donor: hub preferred
            if len(hubs) > 0:
                donor = int(rng.choice(hubs))
            else:
                donor = int(np.argmax(cnt)) if rng.random() < 0.8 else int(rng.integers(0, n))
            yn = y / np.linalg.norm(y, axis=1, keepdims=True)
            others = np.delete(yn, donor, axis=0)
            y[donor] = _void_center(others, rng, ns=4000, refine=100)
            # small jitter to neighbours of donor void to break symmetry
            y += rng.normal(0, 0.004, size=y.shape)
        elif mode == 1:
            # coherent cap rotation
            sigma, steps, b0, b1 = 0.008, 750, 40.0, 280.0
            y += rng.normal(0, sigma, size=y.shape)
            cdir = rng.normal(size=4); cdir /= np.linalg.norm(cdir)
            yn = y / np.linalg.norm(y, axis=1, keepdims=True)
            dots = yn @ cdir
            K = int(rng.integers(8, 19))
            cap = np.argsort(-dots)[:K]
            theta = float(rng.uniform(0.06, 0.28)) * (1 if rng.random() < 0.5 else -1)
            y = _cap_rotate(y, cap, rng, theta)
            y[cap] += rng.normal(0, 0.006, size=(len(cap), d))
            if len(ratt) > 0 and rng.random() < 0.6:
                # nudge rattler toward its nearest neighbour to engage it
                r = int(rng.choice(ratt))
                yn2 = y / np.linalg.norm(y, axis=1, keepdims=True)
                dd = yn2[r] @ yn2.T
                dd[r] = -2
                nn = int(np.argmax(dd))
                y[r] = yn2[r] + 0.15 * (yn2[nn] - (yn2[nn] @ yn2[r]) * yn2[r]) + rng.normal(0, 0.01, size=4)
        elif mode == 2:
            # double transplant: two hubs to two voids
            sigma, steps, b0, b1 = 0.010, 850, 28.0, 340.0
            y += rng.normal(0, sigma, size=y.shape)
            yn = y / np.linalg.norm(y, axis=1, keepdims=True)
            donors = []
            if len(hubs) >= 2:
                donors = rng.choice(hubs, size=2, replace=False).tolist()
            elif len(hubs) == 1:
                donors = [int(hubs[0]), int(rng.integers(0, n))]
            else:
                donors = rng.choice(n, size=2, replace=False).tolist()
            for donor in donors:
                yn = y / np.linalg.norm(y, axis=1, keepdims=True)
                others = np.delete(yn, donor, axis=0)
                y[donor] = _void_center(others, rng, ns=3000, refine=80)
        elif mode == 3:
            # medium global shake + hub extra spread
            sigma, steps, b0, b1 = 0.035, 850, 30.0, 300.0
            y += rng.normal(0, sigma, size=y.shape)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, sigma * 0.8, size=(len(hubs), d))
        elif mode == 4:
            # large shake for basin escape
            sigma, steps, b0, b1 = 0.085, 950, 22.0, 330.0
            y += rng.normal(0, sigma, size=y.shape)
            rep = rng.choice(n, size=int(rng.integers(1, 3)), replace=False)
            yn = y / np.linalg.norm(y, axis=1, keepdims=True)
            for idx in rep:
                if rng.random() < 0.7:
                    others = np.delete(yn, idx, axis=0)
                    y[idx] = _void_center(others, rng, ns=3000, refine=60)
                    yn[idx] = y[idx] / np.linalg.norm(y[idx])
                else:
                    y[idx] = rng.normal(size=4)
        else:
            # small fine jitter
            sigma, steps, b0, b1 = 0.012, 700, 45.0, 260.0
            y += rng.normal(0, sigma, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y2 = _smooth_opt(y, steps, b0, b1, 0.006, 0.0004, Iu0, Ju0)
        v2 = _max_cos(y2)
        # decide LP
        thresh = best_val + (0.006 if mode in (4,) else 0.003)
        do_lp = (v2 < thresh) or (v2 < cur_val + 0.0004 and rng.random() < 0.12)
        if do_lp:
            remain = HARD - (time.time() - t0) - 24.0
            if remain < 0.7:
                # no time for LP, just metropolis on v2
                if v2 < best_val - 1e-12:
                    x_best, best_val = y2, v2
                    x_cur, cur_val = y2, v2
                elif v2 < cur_val or rng.random() < np.exp(-(v2 - cur_val) / T):
                    x_cur, cur_val = y2, v2
                continue
            b = min(2.2, max(0.9, remain / 12.0))
            if v2 > best_val + 0.002:
                b = min(b, 0.9)
            yp, vp = _lp_polish(y2, b)
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
                x_cur, cur_val = yp, vp
            else:
                if vp < cur_val or rng.random() < np.exp(-(vp - cur_val) / T):
                    x_cur, cur_val = yp, vp
        else:
            if v2 < cur_val or rng.random() < np.exp(-(v2 - cur_val) / T):
                x_cur, cur_val = y2, v2
            if v2 < best_val - 1e-12:
                x_best, best_val = y2, v2
    remain = HARD - (time.time() - t0) - 1.2
    if remain > 1.5:
        remain = min(remain, 24.0)
        x_best, best_val = _lp_polish(x_best, remain, r0=1.0e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

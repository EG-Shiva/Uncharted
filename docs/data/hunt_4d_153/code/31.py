import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, iu, thresh):
    G = x @ x.T
    gall = G[iu]
    mx = gall.max()
    tight = gall > mx - thresh
    cnt = np.zeros(x.shape[0], dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return mx, cnt, G

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

def _anneal(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    Iu, Ju = iu
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    if steps <= 1:
        lb = 0.0; bb = 0.0
    else:
        lb = np.log(lr1 / lr0) / (steps - 1)
        bb = np.log(beta1 / beta0) / (steps - 1)
    for s in range(steps):
        beta = beta0 * np.exp(bb * s)
        lr = lr0 * np.exp(lb * s)
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

def _small_rotation(rng, d, theta):
    A = rng.normal(size=(d, d))
    K = (A - A.T) * (theta * 0.5)
    I = np.eye(d)
    try:
        R = np.linalg.solve(I - 0.5 * K, I + 0.5 * K)
    except Exception:
        R = I
    return R

def solve():
    t0 = time.time()
    HARD = 170.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    it = 0
    # main hopping until ~140s
    while time.time() - t0 < 132.0:
        it += 1
        if time.time() - t0 > HARD - 35:
            break
        mx, cnt, G = _contacts(x_best, iu, 1e-7)
        order = np.argsort(cnt)
        rattlers = np.where(cnt <= 2)[0]
        hubs = np.argsort(-cnt)[:8]
        mode = it % 8
        y = x_best.copy()
        if mode == 1 or mode == 0:
            # hole expansion around weakest point
            r = int(order[0])
            dots = x_best[r] @ x_best.T
            nn = np.argsort(-dots)
            nn = nn[nn != r][:12]
            if mode == 0:
                alpha = 0.10 + rng.random() * 0.10
                sig = 0.020
                steps, b0, b1, l0, l1 = 900, 200.0, 4000.0, 0.008, 0.001
            else:
                alpha = 0.22 + rng.random() * 0.15
                sig = 0.040
                steps, b0, b1, l0, l1 = 1200, 150.0, 6000.0, 0.010, 0.001
            for j in nn:
                tang = x_best[r] - (float(x_best[r] @ y[j])) * y[j]
                y[j] = y[j] - alpha * tang + rng.normal(0, sig, size=4)
            y[r] += rng.normal(0, 0.015, size=4)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps, b0, b1, l0, l1, iu)
        elif mode == 2:
            # delete hubs + rattler and farthest-point reinsert
            kdel = 5
            victim = np.concatenate([hubs[:4], order[:1]])
            victim = np.unique(victim)[:kdel]
            # candidate pool
            pool = rng.normal(size=(800, 4))
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
            keep_mask = np.ones(n, dtype=bool); keep_mask[victim] = False
            kept = y[keep_mask]
            chosen = []
            for _ in range(len(victim)):
                if len(chosen) == 0:
                    base = kept
                else:
                    base = np.vstack([kept, np.array(chosen)])
                dots = base @ pool.T  # (nb, pool)
                worst = dots.max(axis=0)
                idx = int(np.argmin(worst))
                chosen.append(pool[idx])
                # remove nearby candidates to diversify
                dd = pool[idx] @ pool.T
                pool = pool[dd < 0.97]
                if len(pool) == 0:
                    pool = rng.normal(size=(200, 4))
                    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
            y[victim] = np.array(chosen[:len(victim)])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 1100, 120.0, 5000.0, 0.009, 0.001, iu)
        elif mode == 3:
            # patch rotation: rotate k nearest of random center
            c = int(rng.integers(0, n))
            dots = x_best[c] @ x_best.T
            k = 10 + int(rng.integers(0, 8))
            patch = np.argsort(-dots)[:k]
            theta = 0.15 + rng.random() * 0.25
            R = _small_rotation(rng, d, theta)
            y[patch] = y[patch] @ R.T
            y[patch] += rng.normal(0, 0.02, size=(len(patch), 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 900, 200.0, 5000.0, 0.008, 0.001, iu)
        elif mode == 4:
            # global small jitter + low-beta spreading
            y = y + rng.normal(0, 0.025, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 800, 60.0, 2500.0, 0.010, 0.0015, iu)
        elif mode == 5:
            # medium jitter + rattler replacement
            y = y + rng.normal(0, 0.06, size=y.shape)
            if len(rattlers) > 0:
                for rr in rattlers[:2]:
                    if rng.random() < 0.7:
                        v = rng.normal(size=4); v /= np.linalg.norm(v)
                        y[int(rr)] = v
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 1000, 150.0, 6000.0, 0.009, 0.001, iu)
        elif mode == 6:
            # antipodal / cap shuffle: pick random direction, flip cap
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            proj = y @ u
            cap = np.where(proj > np.quantile(proj, 0.90))[0]
            if len(cap) > 0:
                R = _small_rotation(rng, d, 0.5)
                y[cap] = y[cap] @ R.T + rng.normal(0, 0.03, size=(len(cap), 4))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 1000, 100.0, 5000.0, 0.009, 0.001, iu)
        else:
            # large global kick
            y = y + rng.normal(0, 0.12, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 1200, 80.0, 4000.0, 0.010, 0.001, iu)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
            # quick LP snap to lock in
            if time.time() - t0 < HARD - 30:
                x_best, best_val = _lp_polish(x_best, 2.0)
        elif v < best_val + 0.003 and (it % 3 == 0):
            # promising: try short LP
            if time.time() - t0 < HARD - 30:
                yp, vp = _lp_polish(y, 1.2)
                if vp < best_val - 1e-12:
                    x_best, best_val = yp, vp
        # time guard
        if time.time() - t0 > 132.0:
            break
    # final polish of incumbent
    remain = HARD - (time.time() - t0) - 1.0
    if remain > 3.0:
        # short re-anneal at high beta then LP
        try:
            ya = _anneal(x_best, 600, 2000.0, 12000.0, 0.003, 0.0005, iu)
            va = _max_cos(ya)
            if va < best_val + 1e-4:
                # LP on better of the two
                cand = ya if va < best_val else x_best
                xp, vp = _lp_polish(cand, min(remain, 22.0))
                if vp < best_val:
                    x_best, best_val = xp, vp
            else:
                xp, vp = _lp_polish(x_best, min(remain, 22.0))
                if vp < best_val:
                    x_best, best_val = xp, vp
        except Exception:
            try:
                xp, vp = _lp_polish(x_best, min(remain, 15.0))
                if vp < best_val:
                    x_best, best_val = xp, vp
            except Exception:
                pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

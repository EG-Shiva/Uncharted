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


def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    iu = np.triu_indices(n, 1)
    Iu0, Ju0 = iu
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    btb = np.log(beta1 / beta0) / max(steps - 1, 1)
    W = np.zeros((n, n))
    for s in range(steps):
        beta = beta0 * np.exp(btb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        w = e / Z
        W[Iu0, Ju0] = w
        W[Ju0, Iu0] = w
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x


def _find_voids(y_keep, n_need, rng, n_cand=20000):
    if n_need <= 0:
        return np.zeros((0, 4))
    cand = rng.normal(size=(n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ y_keep.T
    cur = S.max(axis=1)
    chosen = []
    for _ in range(n_need):
        idx = int(np.argmin(cur))
        chosen.append(cand[idx])
        upd = cand @ cand[idx]
        np.maximum(cur, upd, out=cur)
        cur[idx] = 2.0
    return np.array(chosen)


def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None:
            try:
                z = np.asarray(z, dtype=float).reshape(n, d)
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                cands.append(( _max_cos(z), z))
            except Exception:
                pass
    if len(cands) == 0:
        z = rng.normal(size=(n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append((_max_cos(z), z))
    cands.sort(key=lambda t: t[0])
    best_val, x_best = cands[0][0], cands[0][1].copy()
    iu = np.triu_indices(n, 1)
    it = 0
    # hopping phase until ~135s, reserve rest for final LP
    while time.time() - t0 < 132.0:
        it += 1
        if time.time() - t0 > HARD - 35:
            break
        G = x_best @ x_best.T
        gall = G[iu]
        mx = gall.max()
        tight = gall > mx - 1e-4
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        np.fill_diagonal(G, -2.0)
        mi = G.max(axis=1)
        order_worst = np.argsort(-mi)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 7
        if mode == 0:
            sigma, nrep = 0.008, 0
            hub_k = 0.0
            steps = 300
        elif mode == 1:
            sigma, nrep = 0.020, 1
            hub_k = 0.0
            steps = 350
        elif mode == 2:
            sigma, nrep = 0.040, 2
            hub_k = 0.0
            steps = 400
        elif mode == 3:
            sigma, nrep = 0.012, 1
            hub_k = 0.05
            steps = 350
        elif mode == 4:
            sigma, nrep = 0.090, 2
            hub_k = 0.09
            steps = 450
        elif mode == 5:
            sigma, nrep = 0.150, 3
            hub_k = 0.0
            steps = 450
        else:
            sigma, nrep = 0.030, 1
            hub_k = 0.04
            steps = 350
        y = x_best.copy()
        y += rng.normal(0, sigma, size=y.shape)
        if len(hubs) > 0 and hub_k > 0:
            y[hubs] += rng.normal(0, hub_k, size=(len(hubs), d))
        # pick victims as current worst (recompute quickly)
        yn = y / np.linalg.norm(y, axis=1, keepdims=True)
        Gt = yn @ yn.T
        np.fill_diagonal(Gt, -2.0)
        mit = Gt.max(axis=1)
        victims = np.argsort(-mit)[:nrep]
        # force rattler replacement sometimes
        if mode == 3 and len(ratt) > 0 and nrep > 0:
            victims[0] = ratt[rng.integers(len(ratt))] if len(ratt) > 0 else victims[0]
        if nrep > 0:
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            y_keep = (y[keep_mask] / np.linalg.norm(y[keep_mask], axis=1, keepdims=True))
            voids = _find_voids(y_keep, nrep, rng, n_cand=20000)
            y[victims] = voids + rng.normal(0, sigma * 0.25 + 1e-4, size=(nrep, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed smooth repulsion
        beta0, beta1 = 200.0, 8000.0
        lr0, lr1 = 0.02, 0.002
        if sigma > 0.08:
            lr0 = 0.03
        y2 = _smooth_opt(y, steps, beta0, beta1, lr0, lr1)
        v2 = _max_cos(y2)
        # decide to LP polish
        thresh = best_val + 0.003
        do_polish = (v2 < thresh and (it % 2 == 0 or v2 < best_val))
        if v2 < best_val - 1e-12:
            do_polish = True
        if do_polish:
            remain = HARD - (time.time() - t0) - 22.0
            if remain < 0.5:
                # no time for intermediate polish, keep smooth if better
                if v2 < best_val:
                    # quick accept without LP? still need LP later; store smooth
                    pass
                continue
            bud = min(3.0, remain)
            yp, vp = _lp_polish(y2, bud)
            if vp < best_val:
                best_val = vp
                x_best = yp
        if time.time() - t0 >= HARD - 22.0:
            break
    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        x_best, best_val = _lp_polish(x_best, min(remain, 30.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

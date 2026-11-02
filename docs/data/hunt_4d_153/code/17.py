import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _contacts(x, iu, thresh=1e-7):
    G = x @ x.T
    gall = G[iu]
    mx = float(np.max(gall))
    tight = gall > mx - thresh
    n = x.shape[0]
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return mx, cnt, G


def _lp_polish(x, budget, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = radius0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        sel = c_all > m - max(20.0 * radius, 1e-9)
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
            x = y
            best = v
            xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return xb, best


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
        gmax = np.max(g)
        e = np.exp(beta * (g - gmax))
        Z = np.sum(e)
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        v1 = b2 * v1 + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        # keep tangential component only
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x


def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(0)
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

    trial = 0
    # main basin-hopping loop
    while time.time() - t0 < 148.0:
        trial += 1
        if time.time() - t0 > HARD - 30:
            break
        mx, cnt, G = _contacts(x_best, iu, thresh=1e-7)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = trial % 10
        y = x_best.copy()
        if mode == 0:
            sigma, nrep, steps = 0.012, 0, 250
            b0, b1, l0, l1 = 80.0, 1200.0, 0.004, 0.0004
            y += rng.normal(0, sigma, size=y.shape)
        elif mode == 1:
            sigma, nrep, steps = 0.03, 1, 350
            b0, b1, l0, l1 = 60.0, 1500.0, 0.005, 0.0004
            y += rng.normal(0, sigma, size=y.shape)
            idx = rng.choice(n, size=nrep, replace=False)
            y[idx] = rng.normal(size=(nrep, d))
        elif mode == 2:
            sigma, nrep, steps = 0.07, 2, 450
            b0, b1, l0, l1 = 40.0, 2000.0, 0.006, 0.0005
            y += rng.normal(0, sigma, size=y.shape)
            idx = rng.choice(n, size=nrep, replace=False)
            y[idx] = rng.normal(size=(nrep, d))
        elif mode == 3:
            # contact-cluster shake
            steps = 450
            b0, b1, l0, l1 = 40.0, 2000.0, 0.006, 0.0005
            y += rng.normal(0, 0.02, size=y.shape)
            seed = int(rng.integers(0, n))
            sims = G[seed]
            order = np.argsort(-sims)
            k = int(rng.integers(4, 10))
            cl = order[:k]
            y[cl] += rng.normal(0, 0.18, size=(len(cl), d))
            if len(ratt) > 0 and rng.random() < 0.7:
                y[ratt] = rng.normal(size=(len(ratt), d))
        elif mode == 4:
            # hub replacement
            steps = 400
            b0, b1, l0, l1 = 60.0, 1500.0, 0.005, 0.0004
            y += rng.normal(0, 0.025, size=y.shape)
            if len(hubs) > 0:
                k = min(len(hubs), int(rng.integers(1, 3)))
                pick = rng.choice(hubs, size=k, replace=False)
                y[pick] = rng.normal(size=(k, d))
            else:
                idx = rng.choice(n, size=1, replace=False)
                y[idx] = rng.normal(size=(1, d))
        elif mode == 5:
            sigma, steps = 0.18, 550
            b0, b1, l0, l1 = 30.0, 2500.0, 0.008, 0.0006
            y += rng.normal(0, sigma, size=y.shape)
        elif mode == 6:
            # fresh random start
            steps = 800
            b0, b1, l0, l1 = 20.0, 2500.0, 0.008, 0.0004
            y = rng.normal(size=(n, d))
        elif mode == 7:
            # rattler-targeted + medium noise
            steps = 400
            b0, b1, l0, l1 = 50.0, 1800.0, 0.005, 0.0004
            y += rng.normal(0, 0.05, size=y.shape)
            if len(ratt) > 0:
                y[ratt] = rng.normal(size=(len(ratt), d))
            else:
                # replace lowest-gap points
                gall = G[iu]
                # points in worst pairs
                worst = gall > mx - 1e-6
                cand = np.unique(np.concatenate([iu[0][worst], iu[1][worst]]))
                if len(cand) > 0:
                    k = min(len(cand), 2)
                    pick = rng.choice(cand, size=k, replace=False)
                    y[pick] = rng.normal(size=(k, d))
        elif mode == 8:
            sigma, nrep, steps = 0.05, 3, 500
            b0, b1, l0, l1 = 40.0, 2000.0, 0.006, 0.0005
            y += rng.normal(0, sigma, size=y.shape)
            idx = rng.choice(n, size=nrep, replace=False)
            y[idx] = rng.normal(size=(nrep, d))
        else:
            sigma, steps = 0.10, 500
            b0, b1, l0, l1 = 35.0, 2200.0, 0.007, 0.0005
            y += rng.normal(0, sigma, size=y.shape)
            if len(ratt) > 0 and rng.random() < 0.5:
                y[rng.choice(ratt)] = rng.normal(size=(4,))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        try:
            ys = _smooth_opt(y, steps, b0, b1, l0, l1, iu)
        except Exception:
            continue
        v = _max_cos(ys)
        # decide to LP-polish
        do_lp = False
        lp_budget = 1.6
        if v < best_val:
            do_lp = True
            lp_budget = 3.0
        elif v < best_val + 0.004 and (trial % 3 == 0):
            do_lp = True
            lp_budget = 1.2
        elif mode == 6 and v < best_val + 0.008:
            do_lp = True
            lp_budget = 1.5
        if do_lp:
            if time.time() - t0 > HARD - 25:
                break
            try:
                yl, vl = _lp_polish(ys, lp_budget)
            except Exception:
                continue
            if vl < best_val:
                best_val = float(vl)
                x_best = yl.copy()
        else:
            # even without LP, accept strict win (rare)
            if v < best_val:
                best_val = float(v)
                x_best = ys.copy()
        # timeout guard
        if time.time() - t0 > HARD - 20:
            break

    # final snap with remaining time
    remain = HARD - (time.time() - t0)
    if remain > 3:
        try:
            xf, vf = _lp_polish(x_best, min(remain - 1.0, 14.0), radius0=2e-3)
            if vf < best_val:
                x_best = xf
                best_val = vf
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

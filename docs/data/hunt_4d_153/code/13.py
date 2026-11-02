import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - thr
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return m, cnt, iu, g

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

def _smooth(x0, iu, steps, beta0, beta1, lr0, lr1, mu=0.9, noise=0.0, rng=None):
    n, d = x0.shape
    x = x0.copy()
    vel = np.zeros_like(x)
    Iu, Ju = iu
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        g = G[Iu, Ju]
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        if Z == 0 or not np.isfinite(Z):
            break
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W += W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        # bias correction-ish scale
        step = vel / (1.0 - mu ** (s + 1) + 1e-12)
        x = x - lr * step
        if noise > 0 and rng is not None:
            x = x + rng.normal(0, noise * lr, size=x.shape)
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
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu_all = np.triu_indices(n, 1)

    it = 0
    # cached contact counts of incumbent
    _, cnt_best, _, _ = _contacts(x_best)

    while time.time() - t0 < 145.0:
        it += 1
        remain = HARD - (time.time() - t0)
        if remain < 30:
            break
        # choose perturbation mode
        u = rng.random()
        y = x_best.copy()
        if u < 0.25:
            sigma = float(rng.choice([0.01, 0.02, 0.035, 0.06]))
            y += rng.normal(0, sigma, size=y.shape)
        elif u < 0.50:
            sigma = float(rng.uniform(0.02, 0.10))
            y += rng.normal(0, sigma, size=y.shape)
            # extra kick rattlers (few contacts)
            ratt = np.where(cnt_best <= 2)[0]
            for idx in ratt:
                if rng.random() < 0.7:
                    y[idx] = rng.normal(size=4)
        elif u < 0.78:
            # replace hubs + rattlers with random points
            k = int(rng.integers(2, 8))
            order = np.argsort(-cnt_best)
            cands = order[:max(k * 2, 10)]
            chosen = rng.choice(cands, size=min(k, len(cands)), replace=False)
            ratt = np.where(cnt_best <= 2)[0]
            extra = []
            for idx in ratt:
                if rng.random() < 0.8:
                    extra.append(idx)
            chosen = np.unique(np.concatenate([chosen, np.array(extra, dtype=int)]) if len(extra) else chosen)
            for idx in chosen:
                if rng.random() < 0.75:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] += rng.normal(0, 0.3, size=4)
            # small global jitter too
            y += rng.normal(0, 0.01, size=y.shape)
        else:
            # larger surgery: replace random block + worst pair partner
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2)
            fi, fj = np.unravel_index(int(np.argmax(G)), G.shape)
            y[fi] = rng.normal(size=4)
            if rng.random() < 0.5:
                y[fj] = rng.normal(size=4)
            k = int(rng.integers(3, 12))
            idxs = rng.choice(n, size=k, replace=False)
            for idx in idxs:
                y[idx] = rng.normal(size=4)
            y += rng.normal(0, 0.015, size=y.shape)

        y /= np.linalg.norm(y, axis=1, keepdims=True)

        # anneal params varied per iteration
        steps = int(rng.choice([600, 900, 1200, 1600]))
        # keep steps small if little time
        if remain < 60:
            steps = min(steps, 800)
        beta0 = float(rng.choice([15.0, 25.0, 40.0]))
        beta1 = float(rng.choice([250.0, 400.0, 600.0]))
        lr0 = float(rng.uniform(0.015, 0.035))
        lr1 = float(rng.uniform(0.002, 0.006))
        mu = 0.9
        y2 = _smooth(y, iu_all, steps, beta0, beta1, lr0, lr1, mu=mu, noise=0.0, rng=None)
        v = _max_cos(y2)
        # accept to LP stage if promising
        thresh = best_val + (0.015 if it % 3 == 0 else 0.006)
        if v < thresh:
            # short LP polish
            budget = 2.0 if v > best_val - 0.002 else 3.5
            if remain < 40:
                budget = min(budget, 1.2)
            y3, v3 = _lp_polish(y2, budget)
            if v3 < best_val:
                x_best, best_val = y3, v3
                _, cnt_best, _, _ = _contacts(x_best)
                continue
            # also allow equal-but-different basin to move (diversify)
            if v3 < best_val + 1e-9 and rng.random() < 0.1:
                # check different: keep if contact structure differs
                x_best, best_val = y3, v3
                _, cnt_best, _, _ = _contacts(x_best)
        else:
            # still track if outright better (rare without polish)
            if v < best_val:
                y3, v3 = _lp_polish(y2, 3.0)
                if v3 < best_val:
                    x_best, best_val = y3, v3
                    _, cnt_best, _, _ = _contacts(x_best)
        # occasional random restart equilibration
        if it % 25 == 0 and time.time() - t0 < 110:
            yr = rng.normal(size=(n, d))
            yr /= np.linalg.norm(yr, axis=1, keepdims=True)
            yr = _smooth(yr, iu_all, 1000, 20.0, 350.0, 0.025, 0.004, mu=0.9)
            vr = _max_cos(yr)
            if vr < best_val + 0.02:
                yr2, vr2 = _lp_polish(yr, 2.0)
                if vr2 < best_val:
                    x_best, best_val = yr2, vr2
                    _, cnt_best, _, _ = _contacts(x_best)

    # final intensive polish of incumbent with remaining time
    rem = HARD - (time.time() - t0)
    if rem > 4:
        x_best, best_val = _lp_polish(x_best, min(rem - 1.0, 25.0))
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

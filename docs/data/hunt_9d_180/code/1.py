import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard per iteration
        if time.time() - start > seconds - 0.2:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        # cap constraints for speed
        if np.count_nonzero(sel) > 2500:
            # keep tightest 2500
            idx = np.where(sel)[0]
            vals = c_all[idx]
            keep = np.argpartition(vals, len(vals)-2500)[-2500:]
            nsel = np.zeros_like(sel)
            nsel[idx[keep]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        c = c_all[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            # accept
            x, best = y, v
            xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            # check if predicted improvement possible: if res.fun + m < best then shrink else break?
            radius *= 0.5
    # return best found
    return xb

def best_void_for(x_try, idx, rng, M=25000):
    n, d = x_try.shape
    cand = rng.normal(size=(M, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_try.T
    dots[:, idx] = -2.0
    scores = dots.max(axis=1)
    j = int(np.argmin(scores))
    return cand[j], float(scores[j])

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    BUDGET = 172.0
    # load start
    try:
        xb0 = np.load("best.npy")
    except Exception:
        xb0 = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in [xb0, xr]:
        if a is not None:
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((max_cos(a), a))
    if len(cands) == 0:
        a = rng.normal(size=(180, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best, x_best = cands[0][0], cands[0][1].copy()
    n, d = x_best.shape
    # no long initial polish: already at optimum, do tiny 2s to ensure
    if time.time() - t0 < BUDGET - 30:
        x_best = lp_polish(x_best, min(2.0, BUDGET - 30 - (time.time()-t0)), r0=1e-3)
        best = max_cos(x_best)
    it = 0
    # main basin hopping until ~25s left for final polish
    while time.time() - t0 < BUDGET - 25:
        it += 1
        remain = BUDGET - 25 - (time.time() - t0)
        if remain < 3:
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        overall = float(row_max.max())
        cnt = (G > overall - 5e-4).sum(axis=1)
        order_worst = np.argsort(-row_max)
        order_cnt = np.argsort(-cnt)
        ratt = np.where(row_max < overall - 1e-7)[0]
        x_try = x_best.copy()
        mode = it % 4
        if mode == 0 or mode == 1:
            # void reinsertion: rattlers + worst 3
            repl = []
            for v in ratt.tolist()[:5]:
                repl.append(int(v))
            for v in order_worst[:3]:
                v = int(v)
                if v not in repl:
                    repl.append(v)
            repl = repl[:5]
            for idx in repl:
                bv, _ = best_void_for(x_try, idx, rng, M=25000)
                x_try[idx] = bv
            # small symmetry-breaking noise
            x_try = x_try + rng.normal(scale=0.008, size=x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            budget = min(9.0, remain)
            r0 = 0.015
        elif mode == 2:
            k = 12
            victims = order_cnt[:k].tolist()
            extra = rng.choice(n, size=5, replace=False).tolist()
            for v in extra:
                if v not in victims:
                    victims.append(int(v))
            sigma = 0.30 if (it % 12 == 2) else 0.12
            noise = rng.normal(scale=sigma, size=(len(victims), d))
            x_try[np.array(victims, dtype=int)] += noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            budget = min(9.0, remain)
            r0 = 0.015
        else:
            sigma = 0.025
            # occasionally larger global shake
            if it % 16 == 3:
                sigma = 0.06
            x_try = x_try + rng.normal(scale=sigma, size=x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            budget = min(8.0, remain)
            r0 = 0.01
        if budget < 2.0:
            break
        x_pol = lp_polish(x_try, budget, r0=r0)
        v = max_cos(x_pol)
        if v < best - 1e-12:
            x_best = x_pol
            best = v
            # opportunistic: continue exploiting same basin with smaller kicks
        # else discard
    # final polish of best
    frem = BUDGET - (time.time() - t0) - 1.0
    if frem > 2:
        x_best = lp_polish(x_best, min(frem, 22.0), r0=1e-3)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

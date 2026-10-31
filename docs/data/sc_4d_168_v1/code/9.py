import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        iu_c = G[iu]
        m = iu_c.max()
        sel = iu_c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _adam(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) * beta
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        c1 = 1.0 - b1a ** t
        c2 = 1.0 - b2a ** t
        mh = m / c1
        vh = vv / c2
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    HOP_END = 150.0
    rng = np.random.default_rng(7)
    n, d = 168, 4
    try:
        xb = _norm(np.load("best.npy").astype(float))
    except Exception:
        xb = None
    try:
        xr = _norm(np.load("record.npy").astype(float))
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (n, d):
        cands.append(xb)
    if xr is not None and xr.shape == (n, d):
        cands.append(xr)
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(n, d)))
        best = _max_cos(bestX)
    else:
        vals = [_max_cos(c) for c in cands]
        bestX = cands[int(np.argmin(vals))].copy()
        best = float(min(vals))
    curX = bestX.copy()
    cur = best
    T0, T1 = 0.003, 0.00008
    it = 0
    while True:
        el = time.time() - t0
        if el > HOP_END:
            break
        it += 1
        frac = el / HOP_END
        Temp = T0 * (1.0 - frac) + T1 * frac
        # choose mode
        r = rng.random()
        if r < 0.04:
            # fresh
            Xtry = _norm(rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1 = 1800, 15.0, 300.0, 0.02, 0.002
        elif r < 0.24:
            # large kick + full reinsertions
            sigma = 0.30
            Xtry = _norm(curX + sigma * rng.normal(size=(n, d)))
            k = int(rng.integers(8, 13))
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
            steps, b0, b1e, lr0, lr1 = 1200, 25.0, 300.0, 0.015, 0.0015
        elif r < 0.44:
            # cluster kick around random center
            ci = int(rng.integers(n))
            sim = curX @ curX[ci]
            nn = np.argsort(-sim)[:16]
            Xtry = curX + 0.02 * rng.normal(size=(n, d))
            Xtry[nn] += 0.28 * rng.normal(size=(len(nn), d))
            Xtry = _norm(Xtry)
            # occasionally resample 1 point in cluster
            if rng.random() < 0.4:
                Xtry[int(rng.choice(nn))] = _norm(rng.normal(size=(d,)))
            steps, b0, b1e, lr0, lr1 = 1000, 20.0, 280.0, 0.015, 0.0015
        elif r < 0.64:
            # contact-guided removal + greedy void fill
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            thr = mx - 0.012
            deg = (G > thr).sum(axis=1)
            k = int(rng.integers(3, 7))
            worst = np.argsort(-deg)[:k]
            # also mix in random one sometimes
            if rng.random() < 0.3:
                worst[-1] = int(rng.integers(n))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            Xkeep = curX[keep_mask]
            newpts = []
            for s in range(k):
                cand = _norm(rng.normal(size=(2500, d)))
                if len(newpts) == 0:
                    ex = Xkeep
                else:
                    ex = np.vstack([Xkeep, np.array(newpts)])
                sc = (cand @ ex.T).max(axis=1)
                bi = int(np.argmin(sc))
                # add small jitter among top candidates to diversify
                if rng.random() < 0.3:
                    order = np.argsort(sc)[:20]
                    bi = int(order[int(rng.integers(len(order)))])
                newpts.append(cand[bi])
            Xtry = np.vstack([Xkeep, np.array(newpts)])
            # shuffle rows to avoid optimizer bias
            perm = rng.permutation(n)
            Xtry = Xtry[perm]
            steps, b0, b1e, lr0, lr1 = 1200, 20.0, 320.0, 0.014, 0.0012
        else:
            # small shake from current
            sigmas = [0.012, 0.035, 0.08]
            sigma = sigmas[int(rng.integers(3))]
            Xtry = _norm(curX + sigma * rng.normal(size=(n, d)))
            if rng.random() < 0.25:
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, size=k, replace=False)
                Xtry[idx] = _norm(Xtry[idx] + 0.15 * rng.normal(size=(k, d)))
            if it % 2 == 0:
                steps, b0, b1e, lr0, lr1 = 800, 30.0, 220.0, 0.018, 0.0018
            else:
                steps, b0, b1e, lr0, lr1 = 1000, 18.0, 320.0, 0.012, 0.0012
        # optimize
        try:
            Xopt = _adam(Xtry, steps, b0, b1e, lr0, lr1)
        except Exception:
            continue
        v = _max_cos(Xopt)
        # decide polish
        polished = False
        vp = v
        Xp = Xopt
        want_polish = False
        ptime = 0.0
        if v < best:
            want_polish = True
            ptime = 1.8
        elif v < best + 0.0015 and (it % 5 == 0):
            want_polish = True
            ptime = 0.9
        elif v < cur + 0.0005 and (it % 20 == 0):
            want_polish = True
            ptime = 0.7
        if want_polish and (time.time() - t0) < HOP_END - 1.0:
            try:
                Xp = _lp_polish(Xopt, ptime)
                vp = _max_cos(Xp)
                polished = True
            except Exception:
                Xp = Xopt
                vp = v
        else:
            Xp = Xopt
            vp = v
        # update best
        if vp < best:
            best = vp
            bestX = Xp.copy()
            curX = Xp.copy()
            cur = vp
            continue
        if v < best and not polished:
            # unpolished better than best but polish failed to improve; still record
            if v < best:
                best = v
                bestX = Xopt.copy()
                curX = Xopt.copy()
                cur = v
                continue
        # Metropolis move of current
        use_v, use_X = (vp, Xp) if polished else (v, Xopt)
        # if polished improved over raw, prefer it for walk when close
        if use_v < cur:
            curX = use_X.copy()
            cur = use_v
        else:
            if Temp > 0 and rng.random() < np.exp((cur - use_v) / Temp):
                curX = use_X.copy()
                cur = use_v
        # occasional restart of walk to best to avoid drift too far
        if it % 60 == 0 and rng.random() < 0.5:
            # pull walk back toward best slightly: interpolate
            if cur > best + 0.004:
                curX = bestX.copy()
                cur = best
        if time.time() - t0 > HOP_END:
            break
    rem = 178.0 - (time.time() - t0)
    if rem > 2.0:
        rem = min(rem, 26.0)
        try:
            bestX = _lp_polish(bestX, rem)
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX

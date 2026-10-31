import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (X @ X.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best, radius = Y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine(Xinit, steps, beta0, beta1, lr0, lr1, mu=0.88):
    X = Xinit.copy()
    vel = np.zeros_like(X)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        vel = mu * vel + (1.0 - mu) * gt
        X = X - lr * vel
        X = _norm(X)
    return X

def _contacts(X, tol=1e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    return m, cnt, G

def _voids(X, rng, k=8, ns=12000):
    S = rng.normal(size=(ns, 4))
    S = _norm(S)
    mx = (S @ X.T).max(axis=1)
    idx = np.argsort(mx)[:k]
    return S[idx], mx[idx]

def _refine_single(s0, Xref, rng, steps=90, beta0=40.0, beta1=220.0, lr0=0.09, lr1=0.006):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        d = Xref @ s
        cm = float(np.max(d))
        w = np.exp(np.clip(beta * (d - cm), -20.0, 0.0))
        w = w / (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - float(np.dot(g, s)) * s
        s = s - lr * gt
        nrm = np.linalg.norm(s)
        if nrm < 1e-12:
            s = rng.normal(size=(4,))
            nrm = np.linalg.norm(s)
        s = s / nrm
    return s

def _greedy_fill(Xkeep, k, rng, ns=7000):
    pts = []
    for _ in range(k):
        if len(pts) == 0:
            Xref = Xkeep
        else:
            Xref = np.vstack([Xkeep, np.array(pts)])
        C = _norm(rng.normal(size=(ns, 4)))
        mc = (C @ Xref.T).max(axis=1)
        bi = int(np.argmin(mc))
        s = _refine_single(C[bi], Xref, rng)
        pts.append(s)
    return np.array(pts)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 142.0
    rng = np.random.default_rng(20261031)
    n, d = 168, 4
    try:
        xb = np.load("best.npy").astype(float)
        xb = _norm(xb)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (n, d):
        cands.append(xb)
    if xr is not None and xr.shape == (n, d):
        cands.append(xr)
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(n, d)))
    else:
        vals = [_max_cos(c) for c in cands]
        bestX = cands[int(np.argmin(vals))].copy()
    best = _max_cos(bestX)
    curX = bestX.copy()
    cur = best
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.0:
            break
        # choose base: mostly walk from cur, sometimes restart from best
        if trial % 12 == 0:
            base = bestX
        else:
            base = curX
        typ = trial % 10
        try:
            m, cnt, G = _contacts(base, tol=1e-3)
        except Exception:
            m, cnt = best, np.zeros(n)
        order = np.argsort(-cnt)
        if typ in (0, 3):
            # cluster deletion around overloaded point + greedy reinsert
            k = 6 + (trial % 5)  # 6..10
            if typ == 0:
                center = int(order[0])
            else:
                # random among top-8 overloaded
                top = order[:8]
                center = int(top[rng.integers(len(top))])
            dots = base @ base[center]
            victims = np.argsort(-dots)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = base[keep_mask]
            newpts = _greedy_fill(Xkeep, k, rng, ns=7000)
            Xtry = base.copy()
            Xtry[victims] = newpts
            Xtry = _norm(Xtry + 0.006 * rng.normal(size=(n, d)))
            steps, b0, b1, lr0, lr1 = 750, 28.0, 420.0, 0.013, 0.0008
        elif typ in (1, 6):
            # void transplant of overloaded points
            k = 2 + (trial % 3)  # 2..4
            victims = order[:k].copy()
            V, _ = _voids(base, rng, k=10, ns=12000)
            Xtry = base + 0.015 * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            for a, vi in enumerate(victims):
                vv = V[a % len(V)] + 0.02 * rng.normal(size=(d,))
                vv = vv / (np.linalg.norm(vv) + 1e-30)
                # local refine vs rest
                keep = np.ones(n, dtype=bool); keep[vi] = False
                Xref = Xtry[keep]
                vv = _refine_single(vv, Xref, rng, steps=70)
                Xtry[vi] = vv
            steps, b0, b1, lr0, lr1 = 700, 30.0, 380.0, 0.012, 0.0009
            if typ == 6:
                steps = 900
        elif typ == 2:
            # medium kick + random reinsert
            Xtry = _norm(base + 0.16 * rng.normal(size=(n, d)))
            k = 5
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
            steps, b0, b1, lr0, lr1 = 800, 24.0, 450.0, 0.014, 0.0008
        elif typ == 4:
            # small jitter exploitation
            Xtry = _norm(base + 0.022 * rng.normal(size=(n, d)))
            steps, b0, b1, lr0, lr1 = 650, 35.0, 350.0, 0.011, 0.0010
        elif typ == 5:
            # rattler-adjacent shake: shake jammed hard, keep rattlers
            sig = np.where(cnt >= 6, 0.11, np.where(cnt <= 1, 0.008, 0.035))
            Xtry = _norm(base + sig[:, None] * rng.normal(size=(n, d)))
            steps, b0, b1, lr0, lr1 = 700, 26.0, 400.0, 0.013, 0.0009
        elif typ == 7:
            # large random restart portion
            if trial % 30 == 7:
                Xtry = _norm(rng.normal(size=(n, d)))
                steps, b0, b1, lr0, lr1 = 1100, 20.0, 500.0, 0.015, 0.0007
            else:
                Xtry = _norm(base + 0.30 * rng.normal(size=(n, d)))
                k = 8
                idx = rng.choice(n, size=k, replace=False)
                Xtry[idx] = _norm(rng.normal(size=(k, d)))
                steps, b0, b1, lr0, lr1 = 850, 22.0, 460.0, 0.014, 0.0008
        else:
            # typ 8,9: moderate shake, few voids
            sigma = 0.05 if typ == 8 else 0.09
            Xtry = _norm(base + sigma * rng.normal(size=(n, d)))
            if trial % 3 == 0:
                k = 3
                # replace random low-utility? just random
                idx = rng.choice(n, size=k, replace=False)
                km = np.ones(n, dtype=bool); km[idx] = False
                Xkeep = Xtry[km]
                newpts = _greedy_fill(Xkeep, k, rng, ns=5000)
                Xtry[idx] = newpts
            steps, b0, b1, lr0, lr1 = 700, 28.0, 420.0, 0.012, 0.0009
        # global refine
        try:
            Xopt = _refine(Xtry, steps, b0, b1, lr0, lr1)
        except Exception:
            continue
        v = _max_cos(Xopt)
        # LP squeeze only for new best (save time)
        if v < best and time.time() < HARD_END - 3.0:
            try:
                Xp = _lp_polish(Xopt, min(time.time() + 1.6, HARD_END - 1.0))
                vp = _max_cos(Xp)
                if vp < v:
                    Xopt, v = Xp, vp
            except Exception:
                pass
        if v < best:
            best = v
            bestX = Xopt.copy()
            curX, cur = Xopt.copy(), v
        elif v < cur + 0.0035:
            # Metropolis walk to different basin
            # accept with prob depending on gap
            gap = v - cur
            if gap <= 0 or rng.random() < float(np.exp(-gap / 0.0012)):
                curX, cur = Xopt.copy(), v
            # also occasionally accept worse to escape even if prob fails
            elif trial % 25 == 0 and v < cur + 0.006:
                curX, cur = Xopt.copy(), v
        # periodic recenter if stuck
        if trial % 40 == 0 and cur > best + 0.004:
            curX, cur = bestX.copy(), best
        if time.time() > SEARCH_END:
            break
    # final squeeze
    rem = HARD_END - time.time()
    if rem > 3.0:
        try:
            bestX = _lp_polish(bestX, time.time() + min(rem - 0.5, 24.0))
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX

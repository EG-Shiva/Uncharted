import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def gd_refine(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
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
        mhat = m / c1
        vhat = vv / c2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def find_voids(X, rng, n_voids=8, n_samples=20000):
    # directions far from all points (minimise max dot)
    S = _norm(rng.normal(size=(n_samples, 4)))
    md = S @ X.T
    mx = md.max(axis=1)
    idx = np.argsort(mx)[:n_voids]
    return S[idx], mx[idx]

def contact_counts(X, tol=1e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - tol)
    cnt = C.sum(axis=1)
    return m, cnt

def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(_norm(xb))
    if xr is not None:
        cands.append(_norm(xr))
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(168, 4)))
        best = max_cos(bestX)
    else:
        vals = [max_cos(c) for c in cands]
        bi = int(np.argmin(vals))
        bestX = cands[bi].copy()
        best = float(vals[bi])
    n, d = 168, 4
    HOP_END = 148.0
    it = 0
    # precompute void-teleport schedule
    while True:
        now = time.time() - t0
        if now > HOP_END:
            break
        it += 1
        mode = it % 10
        # default jitter settings
        if mode == 0:
            # fresh random (diversity)
            Xtry = _norm(rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1 = 2000, 20.0, 260.0, 0.02, 0.002
        elif mode in (1, 4, 7):
            # VOID-GUIDED: move overloaded points into empty caps
            m, cnt = contact_counts(bestX, tol=1e-3)
            # overloaded indices
            order = np.argsort(-cnt)
            voids, vscore = find_voids(bestX, rng, n_voids=12, n_samples=15000)
            Xtry = bestX + 0.02 * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            # number to transplant
            k = 2 + (it % 4)  # 2..5
            victims = order[:k]
            # pick k best voids (most empty), add small jitter so not identical
            for a, vi in enumerate(victims):
                vv = voids[a % len(voids)] + 0.03 * rng.normal(size=(d,))
                vv = vv / np.linalg.norm(vv)
                Xtry[vi] = vv
            # also jitter neighbours of victims a bit more to break contacts
            if it % 2 == 0:
                Xtry = Xtry + 0.03 * rng.normal(size=(n, d))
                Xtry = _norm(Xtry)
                # restore transplanted (keep them)
                for a, vi in enumerate(victims):
                    pass
            steps, b0, b1e, lr0, lr1 = 1700, 25.0, 300.0, 0.015, 0.0012
            if mode == 7:
                steps = 2000
        elif mode in (2, 8):
            # large kick + full re-insertions (topology change)
            sigma = 0.30 if mode == 2 else 0.18
            Xtry = bestX + sigma * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            k = 8 if mode == 2 else 5
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
            steps, b0, b1e, lr0, lr1 = 1800, 22.0, 280.0, 0.016, 0.0015
        elif mode == 5:
            # rattler-centred shake: keep rattlers, shake jammed hard
            m, cnt = contact_counts(bestX, tol=1e-3)
            Xtry = bestX.copy()
            # small move for low-contact, big for high-contact
            sig = np.where(cnt >= 6, 0.12, np.where(cnt <= 1, 0.01, 0.04))
            Xtry = Xtry + sig[:, None] * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            steps, b0, b1e, lr0, lr1 = 1600, 30.0, 220.0, 0.018, 0.0018
        else:
            # medium shake + few re-inserts
            sigma = [0.02, 0.06, 0.12][it % 3]
            Xtry = bestX + sigma * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            if it % 3 == 0:
                k = 3 + (it % 4)
                idx = rng.choice(n, size=k, replace=False)
                Xtry[idx] = _norm(rng.normal(size=(k, d)))
            if it % 2 == 0:
                steps, b0, b1e, lr0, lr1 = 1500, 30.0, 200.0, 0.02, 0.002
            else:
                steps, b0, b1e, lr0, lr1 = 1800, 20.0, 320.0, 0.012, 0.001
        Xopt = gd_refine(Xtry, steps, b0, b1e, lr0, lr1)
        v = max_cos(Xopt)
        need_polish = False
        polish_time = 1.0
        if v < best:
            need_polish = True
            polish_time = 2.2
        elif v < best + 0.002 and (it % 4 == 0):
            need_polish = True
            polish_time = 1.0
        elif v < best + 0.006 and (it % 12 == 0):
            need_polish = True
            polish_time = 0.8
        if need_polish:
            if time.time() - t0 > HOP_END - 2.0:
                if v < best:
                    best = v
                    bestX = Xopt.copy()
                break
            try:
                Xp = lp_polish(Xopt, polish_time)
            except Exception:
                Xp = Xopt
            vp = max_cos(Xp)
            if vp < best:
                best = vp
                bestX = Xp.copy()
            elif v < best:
                best = v
                bestX = Xopt.copy()
        else:
            if v < best:
                best = v
                bestX = Xopt.copy()
        if time.time() - t0 > HOP_END:
            break
    rem = 178.0 - (time.time() - t0)
    if rem > 3.0:
        rem = min(rem, 26.0)
        try:
            bestX = lp_polish(bestX, rem)
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX

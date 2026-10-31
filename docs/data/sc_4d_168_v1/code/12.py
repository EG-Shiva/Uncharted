import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def lp_polish(x, seconds):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        # limit to tightest 4000 to keep LP small
        if np.count_nonzero(sel) > 4000:
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:4000]] = True
            sel = sel & keep
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # equality: x_i . dx_i = 0
            er = np.repeat(np.arange(n), d)
            ec = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (er, ec)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = _norm(x + dx)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def gd_anneal(X, steps, b0, b1, lr0, lr1, noise0=0.0, rng=None):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(max(b0, 1e-9)); lb1 = np.log(max(b1, 1e-9))
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / max(steps, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) * (beta / max(n, 1))
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # decaying Langevin noise to escape
        if noise0 > 0 and rng is not None:
            amp = noise0 * (1.0 - f)
            if amp > 0:
                Nz = rng.normal(size=X.shape)
                Nz -= np.sum(Nz * X, axis=1, keepdims=True) * X
                Gt = Gt + amp * Nz
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        c1 = 1.0 - b1a ** t
        c2 = 1.0 - b2a ** t
        mhat = m / c1
        vhat = vv / c2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def farthest_reinsert(X, idx_remove, rng, n_cand=2000):
    n, d = X.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[np.asarray(idx_remove)] = False
    Xkeep = X[keep_mask]
    Y = np.empty((n, d))
    Y[keep_mask] = Xkeep
    # greedy sequential fill
    cur = Xkeep.copy()
    for k in np.asarray(idx_remove):
        cand = _norm(rng.normal(size=(n_cand, d)))
        # max dot to current set
        dots = cand @ cur.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        Y[k] = cand[bi]
        cur = np.vstack([cur, cand[bi:bi+1]])
    return _norm(Y)

def contact_counts(X, tol=2e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - tol).sum(axis=1)
    return C, m

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
    TIME_LIMIT = 178.0
    HOP_END = 148.0
    it = 0
    while True:
        now = time.time() - t0
        if now > HOP_END:
            break
        if now > TIME_LIMIT - 32.0:
            break
        it += 1
        mode = it % 10
        # default gd params
        if mode == 0:
            # fresh random exploration
            Xtry = _norm(rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1, nz = 2000, 8.0, 220.0, 0.03, 0.002, 0.0
        elif mode == 1:
            sigma = 0.02
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1, nz = 1200, 20.0, 320.0, 0.015, 0.0015, 0.0
        elif mode == 2:
            sigma = 0.07
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            # reinsert 3 worst
            try:
                G = Xtry @ Xtry.T
                np.fill_diagonal(G, -2.0)
                # find worst pairs, collect indices
                iu = np.triu_indices(n, 1)
                c = G[iu]
                order = np.argsort(-c)[:6]
                bad = set()
                for o in order:
                    bad.add(int(iu[0][o])); bad.add(int(iu[1][o]))
                    if len(bad) >= 3:
                        break
                bad = list(bad)[:3]
                Xtry = farthest_reinsert(Xtry, bad, rng, n_cand=2500)
            except Exception:
                pass
            steps, b0, b1e, lr0, lr1, nz = 1500, 15.0, 350.0, 0.02, 0.0012, 0.002
        elif mode == 3:
            sigma = 0.22
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            C, _ = contact_counts(Xtry, tol=2e-4)
            # rattlers + overloaded + random
            order_low = np.argsort(C)  # rattlers first
            order_high = np.argsort(-C)
            k = 10
            sel = set(order_low[:3].tolist()) | set(order_high[:4].tolist())
            while len(sel) < k:
                sel.add(int(rng.integers(0, n)))
            Xtry = farthest_reinsert(Xtry, list(sel), rng, n_cand=2500)
            steps, b0, b1e, lr0, lr1, nz = 1800, 10.0, 300.0, 0.025, 0.0015, 0.003
        elif mode == 4:
            # targeted surgery around single worst edge
            Xtry = bestX.copy()
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            # neighbourhood of i,j
            ni = np.argsort(-G[ij[0]])[:4].tolist()
            nj = np.argsort(-G[ij[1]])[:4].tolist()
            sel = list(set(ni + nj))[:7]
            Xtry = farthest_reinsert(Xtry, sel, rng, n_cand=3000)
            # small shake rest
            Xtry = _norm(Xtry + 0.01 * rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1, nz = 1600, 18.0, 400.0, 0.018, 0.001, 0.001
        elif mode == 5:
            sigma = 0.04
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            k = 2 + (it % 4)
            idx = rng.choice(n, size=k, replace=False).tolist()
            Xtry = farthest_reinsert(Xtry, idx, rng, n_cand=2000)
            steps, b0, b1e, lr0, lr1, nz = 1400, 22.0, 380.0, 0.015, 0.001, 0.0
        elif mode == 6:
            sigma = 0.12
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1, nz = 1300, 12.0, 260.0, 0.02, 0.002, 0.004
        elif mode == 7:
            # shake only overloaded cluster
            Xtry = bestX.copy()
            C, _ = contact_counts(Xtry, tol=1e-6)
            hi = np.argsort(-C)[:20]
            Xtry[hi] = _norm(Xtry[hi] + 0.15 * rng.normal(size=(len(hi), d)))
            Xtry = _norm(Xtry)
            steps, b0, b1e, lr0, lr1, nz = 1500, 16.0, 450.0, 0.015, 0.0008, 0.0
        elif mode == 8:
            sigma = 0.30
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            k = 6
            idx = rng.choice(n, size=k, replace=False).tolist()
            # pure random reinsert for topology jump
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
            steps, b0, b1e, lr0, lr1, nz = 1900, 8.0, 280.0, 0.025, 0.0015, 0.002
        else:
            sigma = 0.05
            Xtry = _norm(bestX + sigma * rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1, nz = 1200, 25.0, 500.0, 0.012, 0.0008, 0.0
        # annealed GD
        try:
            Xopt = gd_anneal(Xtry, steps, b0, b1e, lr0, lr1, noise0=nz, rng=rng)
        except Exception:
            continue
        try:
            coarse = max_cos(Xopt)
        except Exception:
            continue
        if coarse < best:
            best = coarse
            bestX = Xopt.copy()
            # quick snap to true optimum
            rem = TIME_LIMIT - (time.time() - t0) - 8.0
            if rem > 3.0:
                try:
                    Xs = lp_polish(bestX, min(4.0, rem))
                    vs = max_cos(Xs)
                    if vs < best:
                        best = vs
                        bestX = Xs
                except Exception:
                    pass
        elif coarse < best + 0.008:
            # promising basin, short polish
            rem = HOP_END - (time.time() - t0)
            if rem > 5.0:
                try:
                    Xs = lp_polish(Xopt, 1.8)
                    vs = max_cos(Xs)
                    if vs < best:
                        best = vs
                        bestX = Xs.copy()
                except Exception:
                    pass
        # timeout guard
        if time.time() - t0 > HOP_END:
            break
    # final long polish of best
    try:
        rem = TIME_LIMIT - (time.time() - t0) - 1.5
        if rem > 2.0:
            Xs = lp_polish(bestX, min(28.0, rem))
            vs = max_cos(Xs)
            if vs < max_cos(bestX):
                bestX = Xs
    except Exception:
        pass
    bestX = _norm(bestX)
    return bestX

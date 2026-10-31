import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _cnt_adj(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    A = G > m - tol
    cnt = A.sum(axis=1)
    return m, cnt, A, G

def _smooth(X, steps, b0, b1, lr0, lr1):
    X = _norm(X.copy())
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1_ = 0.9; b2_ = 0.999; eps = 1e-8
    p1 = 1.0; p2 = 1.0
    for s in range(1, steps + 1):
        f = s / max(steps, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = G.max()
        W = np.exp(beta * (G - cm))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        Gr = W @ X
        pr = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - pr * X
        p1 *= b1_; p2 *= b2_
        m1 *= b1_; m1 += (1 - b1_) * Gt
        vv *= b2_; vv += (1 - b2_) * Gt * Gt
        mh = m1 / (1 - p1); vh = vv / (1 - p2)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _lp(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    rad = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and rad > 1e-14:
        if time.time() - t0 > seconds - 0.15:
            break
        G = X @ X.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * rad, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1500:
            o = np.argsort(c[idx])[-1500:]
            idx = idx[o]
        I, J = iu[0][idx], iu[1][idx]
        cs = c[idx]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc1 = I[:, None] * d + np.arange(d)
        cc2 = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cc1, cc2, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            rad *= 0.5
            continue
        if res.status != 0:
            rad *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return bx, best

def _rand_rot(rng, d=6):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q

def _cluster(rng, A, cnt, size):
    seed = int(np.argmax(cnt + rng.random(len(cnt)) * 2.0))
    vis = {seed}
    frontier = [seed]
    while len(vis) < size and frontier:
        u = frontier.pop(rng.integers(0, len(frontier)))
        nbrs = np.where(A[u])[0]
        rng.shuffle(nbrs)
        for w in nbrs:
            w = int(w)
            if w not in vis:
                vis.add(w)
                frontier.append(w)
                if len(vis) >= size:
                    break
    if len(vis) < size:
        extra = rng.choice(len(cnt), size - len(vis), replace=False)
        for w in extra:
            vis.add(int(w))
    return np.array(sorted(vis), dtype=int)

def solve():
    t0 = time.time()
    DEAD = 178.0
    rng = np.random.default_rng(12345)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(float)
            if a.shape == (116, 6) and np.all(np.isfinite(a)):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy(); best = float(vals[bi])
    cur = Xbest.copy(); curv = best
    if time.time() - t0 < 150:
        Xp, vp = _lp(Xbest, min(4.0, max(0.5, DEAD - (time.time() - t0) - 145)))
        if vp < best:
            Xbest, best = Xp, vp
            cur, curv = Xp.copy(), vp
    it = 0
    explore_end = t0 + 140.0
    while time.time() < explore_end and time.time() - t0 < DEAD - 38:
        it += 1
        mode = it % 6
        m, cnt, A, G = _cnt_adj(cur, tol=2e-7)
        Y = cur.copy()
        if mode == 0:
            sig = float(rng.uniform(0.02, 0.06))
            Y = _norm(Y + sig * rng.standard_normal(Y.shape))
            Y = _smooth(Y, 600, 30, 700, 0.02, 0.004)
        elif mode == 1:
            k = int(rng.integers(8, 18))
            cl = _cluster(rng, A, cnt, k)
            R = _rand_rot(rng)
            Y[cl] = (R @ Y[cl].T).T
            Y[cl] += 0.02 * rng.standard_normal((len(cl), 6))
            Y += 0.004 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 800, 50, 1200, 0.03, 0.003)
        elif mode == 2:
            iu2 = np.triu_indices(116, 1)
            c = G[iu2]
            o = np.argsort(-c)[:4]
            for kk in o:
                a = int(iu2[0][kk]); b = int(iu2[1][kk])
                dab = float(Y[a] @ Y[b])
                ga = Y[b] - dab * Y[a]
                gb = Y[a] - dab * Y[b]
                st = float(rng.uniform(0.08, 0.2))
                Y[a] = Y[a] - st * ga
                Y[b] = Y[b] - st * gb
            Y = _norm(Y + 0.006 * rng.standard_normal(Y.shape))
            Y = _smooth(Y, 800, 40, 1000, 0.02, 0.003)
        elif mode == 3:
            k = int(rng.integers(15, 30))
            idx = rng.choice(116, k, replace=False)
            R = _rand_rot(rng)
            Y[idx] = (R @ Y[idx].T).T
            Y = _norm(Y + 0.008 * rng.standard_normal(Y.shape))
            Y = _smooth(Y, 700, 30, 900, 0.025, 0.003)
        elif mode == 4:
            order = np.argsort(-cnt)
            k = int(rng.integers(5, 10))
            idx = order[:k]
            Y[idx] += 0.25 * rng.standard_normal((k, 6))
            Y += 0.01 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 800, 60, 1500, 0.03, 0.0025)
        else:
            sig = float(rng.uniform(0.008, 0.025))
            Y = _norm(Y + sig * rng.standard_normal(Y.shape))
            Y = _smooth(Y, 700, 80, 2000, 0.015, 0.002)
        v = _maxcos(Y)
        if v < best:
            rem = min(3.0, max(0.3, explore_end - time.time()))
            Y2, v2 = _lp(Y, rem)
            if v2 < v:
                Y, v = Y2, v2
            if v < best:
                Xbest, best = Y.copy(), v
                cur, curv = Y.copy(), v
                continue
        temp = 0.0004 * max(0.0, 1.0 - (time.time() - t0) / max(explore_end - t0, 1e-9))
        if v < curv or (temp > 0 and rng.random() < np.exp(-(v - curv) / max(temp, 1e-9))):
            cur, curv = Y.copy(), v
        if it % 12 == 0 and time.time() < explore_end - 5:
            Y2, v2 = _lp(cur, 1.2)
            if v2 < best:
                Xbest, best = Y2.copy(), v2
            if v2 < curv:
                cur, curv = Y2.copy(), v2
    left = DEAD - (time.time() - t0)
    if left > 1.0:
        Xp, vp = _lp(Xbest, min(left - 0.5, 34.0))
        if vp < best:
            Xbest, best = Xp, vp
    left = DEAD - (time.time() - t0)
    if left > 4.0:
        Y = _smooth(Xbest, 1000, 100, 2500, 0.008, 0.001)
        v = _maxcos(Y)
        if v < best + 2e-6:
            Y2, v2 = _lp(Y, left - 0.5)
            if v2 < best:
                Xbest, best = Y2, v2
    return _norm(Xbest)

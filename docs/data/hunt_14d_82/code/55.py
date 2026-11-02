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

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _quench(X0, Tlist, lrs, t_end, rng=None):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for si, (T, p) in enumerate(Tlist):
        lr = lrs[si]
        steps = 120
        # fewer steps for last stage
        if si == len(Tlist) - 1:
            steps = 120
        for _ in range(steps):
            t += 1
            if (t % 60) == 0 and time.time() > t_end:
                return X
            G = X @ X.T
            np.fill_diagonal(G, 0.0)
            D = G - T
            mask = D > 0
            if not np.any(mask):
                continue
            W = np.zeros_like(G)
            Dm = D[mask]
            if p == 2:
                W[mask] = 2.0 * Dm
            else:
                # clip to avoid overflow
                W[mask] = float(p) * np.power(Dm, float(p - 1))
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            m1 = b1 * m1 + (1 - b1) * Gt
            v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            upd = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
            X = X - lr * upd
            X = _norm(X)
    return X

def _lp_polish(X, t_end, radius_init=4e-4, cap=1000, max_it=6):
    X = _norm(X.copy())
    best = _max_cos(X)
    Xbest = X.copy()
    radius = float(radius_init)
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and radius > 1e-12 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-7)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        C = np.empty((p, 2 * d + 1), dtype=int)
        cd = np.arange(d)
        C[:, :d] = I[:, None] * d + cd
        C[:, d:2 * d] = J[:, None] * d + cd
        C[:, 2 * d] = n * d
        cols = C.ravel()
        M = np.empty((p, 2 * d + 1))
        M[:, :d] = X[J]
        M[:, d:2 * d] = X[I]
        M[:, 2 * d] = -1.0
        vals = M.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 4.0})
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
            Xbest = Y.copy()
            radius = min(radius * 1.5, 2e-3)
        else:
            radius *= 0.5
    return Xbest

def solve():
    t0 = time.time()
    HARD = t0 + 132.0
    n, d = 82, 14
    rng = np.random.default_rng(7)
    Xfb = None; vfb = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < vfb:
                        vfb = v; Xfb = a.copy()
        except Exception:
            pass
    if Xfb is None:
        Xfb = _norm(rng.standard_normal((n, d)))
        vfb = _max_cos(Xfb)
    Xbest = Xfb.copy(); vbest = vfb
    Xcur = Xfb.copy(); vcur = vfb

    # hopping loop until reserve for polish
    POLISH_RESERVE = 16.0
    it = 0
    while time.time() < HARD - POLISH_RESERVE:
        it += 1
        if time.time() > HARD - POLISH_RESERVE:
            break
        # temperature for accepting worse cur
        elapsed = time.time() - t0
        temp = 0.0015 * max(0.0, 1.0 - elapsed / 120.0) + 1e-4
        # choose move type
        u = rng.random()
        Xtry = Xcur.copy()
        if u < 0.22:
            # far-hole reinsertion of 1-2 worst points
            rm, G = _row_max(Xcur)
            order = np.argsort(-rm)
            k = 1 if rng.random() < 0.7 else 2
            victims = order[:k]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xkeep = Xcur[keep]
            for v in victims:
                cand = _norm(rng.standard_normal((1500, d)))
                dots = cand @ Xkeep.T
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                Xtry[v] = cand[bi]
            # small jitter around inserted
            Xtry = _norm(Xtry + 0.005 * rng.standard_normal((n, d)))
        else:
            # kick worst subset + global jitter
            rm, _ = _row_max(Xcur)
            order = np.argsort(-rm)
            if rng.random() < 0.5:
                k = int(rng.integers(4, 16))
                idx = order[:k]
                # larger kick for topology change
                scale = float(rng.choice(np.array([0.02, 0.05, 0.10, 0.20])))
                Xtry[idx] = _norm(Xtry[idx] + scale * rng.standard_normal((k, d)))
            else:
                # global moderate kick
                scale = float(rng.choice(np.array([0.01, 0.03, 0.06])))
                Xtry = _norm(Xtry + scale * rng.standard_normal((n, d)))
        # build annealed thresholds anchored at global best
        b = vbest
        Tlist = [(b - 0.0045, 2), (b - 0.0020, 3), (b - 0.0006, 4), (b, 6)]
        lrs = [0.008, 0.006, 0.004, 0.003]
        # quench with time cap
        q_end = min(HARD - POLISH_RESERVE, time.time() + 2.5)
        Xq = _quench(Xtry, Tlist, lrs, q_end)
        if not np.all(np.isfinite(Xq)):
            continue
        vq = _max_cos(Xq)
        if vq < vbest - 1e-12:
            vbest = vq; Xbest = Xq.copy()
        # move cur: greedy + SA
        if vq < vcur:
            Xcur, vcur = Xq, vq
        else:
            if rng.random() < np.exp(-(vq - vcur) / temp):
                Xcur, vcur = Xq, vq
            else:
                # occasionally restart from best with fresh kick
                if rng.random() < 0.15:
                    Xcur, vcur = Xbest.copy(), vbest
        # early exit if time low
        if time.time() > HARD - POLISH_RESERVE:
            break

    # final LP squeeze on best
    remain = HARD - time.time()
    if remain > 5.0:
        try:
            Xp = _lp_polish(Xbest, HARD - 1.0, radius_init=4e-4, cap=1000, max_it=6)
            vp = _max_cos(Xp)
            if vp < vbest and np.all(np.isfinite(Xp)):
                Xbest = Xp; vbest = vp
        except Exception:
            pass
    Xbest = _norm(np.asarray(Xbest, dtype=float))
    return Xbest

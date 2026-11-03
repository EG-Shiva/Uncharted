import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n) | (n == 0)] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _power_adam(X, steps, p0, p1, lr, cap=0.03, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lp0, lp1 = np.log(p0), np.log(p1)
    for t in range(1, steps + 1):
        if t % 250 == 0 and time.time() > deadline:
            break
        frac = t / max(steps, 1)
        p = float(np.exp(lp0 + (lp1 - lp0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        if not np.isfinite(M):
            break
        Mm = M if M > 1e-12 else 1.0
        R = G / Mm
        np.clip(R, 0.0, 1.0, out=R)
        np.fill_diagonal(R, 0.0)
        # power weights, max weight = 1
        with np.errstate(over='ignore', under='ignore'):
            W = np.power(R, p - 1.0, out=np.zeros_like(R), where=R > 1e-12)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * (1.0 - 0.5 * frac) * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, cap / rn)
        step = step * scale
        X = X - step
        X = _norm(X)
    return X

def _hard_adam(X, steps, margin, lr, cap=0.02, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        thr = M - margin
        W = (G > thr).astype(float)
        # linear ramp to focus hardest
        try:
            W = W * (0.3 + 0.7 * np.clip((G - thr) / max(margin, 1e-12), 0, 1))
        except Exception:
            pass
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, cap / rn)
        step = step * scale
        X = X - step
        X = _norm(X)
    return X

def _greedy_fill(keep, k, rng, C=5000):
    d = keep.shape[1]
    cur = keep.copy()
    for _ in range(k):
        cand = rng.standard_normal(size=(C, d))
        cand = _norm(cand)
        if cur.shape[0] > 0:
            sc = np.max(cand @ cur.T, axis=1)
            bi = int(np.argmin(sc))
        else:
            bi = 0
        cur = np.vstack([cur, cand[bi:bi + 1]])
    return cur

def _lp_polish(X, seconds, deadline=1e18):
    X = _norm(X)
    n, d = X.shape
    best = _mc(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (X @ X.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if int(np.count_nonzero(sel)) > 2500:
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:2500]] = True
            sel = sel & keep
        I, J = iu[0][sel], iu[1][sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _mc(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 154, 10
    Xbest = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                vv = _mc(a)
                if vv < best_val:
                    best_val = vv
                    Xbest = a.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = _norm(rng.standard_normal(size=(n, d)))
        best_val = _mc(Xbest)
    Xbest = _norm(Xbest.copy())

    trial = 0
    # quick initial fresh pool to get different basins early
    while time.time() < DEAD - 14.0:
        trial += 1
        if time.time() > DEAD - 14.0:
            break
        try:
            mode = trial % 5
            if mode == 1:
                # completely fresh random basin
                X0 = _norm(rng.standard_normal(size=(n, d)))
                s_pow, s_hard = 1100, 300
                p0, p1, lr0 = 6.0, 90.0, 0.006
            elif mode == 2:
                # huge global kick of incumbent
                X0 = Xbest + 0.55 * rng.standard_normal(size=(n, d))
                X0 = _norm(X0)
                s_pow, s_hard = 900, 250
                p0, p1, lr0 = 8.0, 100.0, 0.005
            elif mode == 3:
                # mass extinction: kill 30-60 random points
                k = int(30 + (trial % 4) * 10)  # 30..60
                idx = np.sort(rng.choice(n, size=k, replace=False))
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                keep = Xbest[mask]
                X0 = _greedy_fill(keep, k, rng, C=4000)
                X0 = _norm(X0 + 0.005 * rng.standard_normal(size=(n, d)))
                s_pow, s_hard = 900, 250
                p0, p1, lr0 = 7.0, 90.0, 0.005
            elif mode == 4:
                # kill worst-overloaded neighbourhood + random
                Gc = Xbest @ Xbest.T
                np.fill_diagonal(Gc, -2.0)
                mv = float(np.max(Gc))
                cnt = np.sum(Gc > mv - 0.01, axis=1)
                order = np.argsort(-cnt - 0.5 * rng.random(n))
                k = 24 + int(rng.integers(0, 16))
                idx = np.sort(order[:k])
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                keep = Xbest[mask]
                X0 = _greedy_fill(keep, k, rng, C=5000)
                X0 = _norm(X0 + 0.008 * rng.standard_normal(size=(n, d)))
                s_pow, s_hard = 1000, 280
                p0, p1, lr0 = 6.0, 110.0, 0.006
            else:
                # targeted worst-point hole regrow (medium kick)
                Gc = Xbest @ Xbest.T
                np.fill_diagonal(Gc, -2.0)
                per = np.max(Gc, axis=1)
                w = int(np.argmax(per))
                mask = np.ones(n, dtype=bool)
                mask[w] = False
                # also kill its nearest neighbours
                sims = Gc[w].copy()
                nn = np.argsort(-sims)[:int(rng.integers(4, 10))]
                mask[nn] = False
                k = int(np.sum(~mask))
                keep = Xbest[mask]
                X0 = _greedy_fill(keep, k, rng, C=5000)
                X0 = _norm(X0 + 0.006 * rng.standard_normal(size=(n, d)))
                s_pow, s_hard = 900, 250
                p0, p1, lr0 = 7.0, 100.0, 0.005

            dl = DEAD - 14.0
            X0 = _power_adam(X0, steps=s_pow, p0=p0, p1=p1, lr=lr0, cap=0.03, deadline=dl)
            if time.time() > dl:
                vv = _mc(X0)
                if vv < best_val - 1e-12:
                    Xbest, best_val = _norm(X0), vv
                continue
            # anneal margin hard phase
            X0 = _hard_adam(X0, steps=s_hard, margin=0.012, lr=0.004, cap=0.02, deadline=dl)
            if time.time() > dl:
                vv = _mc(X0)
                if vv < best_val - 1e-12:
                    Xbest, best_val = _norm(X0), vv
                continue
            X0 = _hard_adam(X0, steps=200, margin=0.003, lr=0.003, cap=0.012, deadline=dl)
            vv = _mc(X0)
            if vv < best_val - 1e-12:
                Xbest, best_val = _norm(X0.copy()), vv
        except Exception:
            continue

    # final polish of incumbent: hard descent + short LP snap, keep best
    try:
        rem = DEAD - time.time() - 1.0
        if rem > 3.0:
            Xc = _norm(Xbest.copy())
            Xc = _hard_adam(Xc, steps=400, margin=0.005, lr=0.003, cap=0.012, deadline=DEAD - 7.0)
            v = _mc(Xc)
            if v < best_val:
                Xbest, best_val = Xc, v
            rem2 = DEAD - time.time() - 0.8
            if rem2 > 2.0:
                try:
                    Y = _lp_polish(Xbest, seconds=min(6.0, rem2), deadline=DEAD - 0.6)
                    vy = _mc(Y)
                    if vy < best_val:
                        Xbest, best_val = Y, vy
                except Exception:
                    pass
    except Exception:
        pass
    Xbest = _norm(Xbest)
    if not np.all(np.isfinite(Xbest)):
        try:
            Xbest = _norm(np.asarray(np.load("best.npy"), dtype=float))
        except Exception:
            Xbest = _norm(rng.standard_normal(size=(n, d)))
    return Xbest

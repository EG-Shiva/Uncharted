import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    cnts = C.sum(axis=1).astype(int)
    return m, cnts, G

def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _adam_quench(X0, schedule, rng=None):
    # schedule: list of (beta, steps, lr)
    n, d = X0.shape
    Y = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for beta, steps, lr in schedule:
        m1 = np.zeros_like(Y)
        m2 = np.zeros_like(Y)
        for t in range(1, steps + 1):
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            # weights over ordered pairs (symmetric)
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                break
            W = E / S
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            Gt = GY - dot * Y  # tangent grad (up to scale)
            # Adam (descent: Y -= lr * step)
            m1 = b1 * m1 + (1 - b1) * Gt
            m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
            m1h = m1 / (1 - b1 ** t)
            m2h = m2 / (1 - b2 ** t)
            step = m1h / (np.sqrt(m2h) + eps)
            # project step to tangent
            sdot = (step * Y).sum(axis=1, keepdims=True)
            stept = step - sdot * Y
            Y = Y - lr * stept
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Y / nr
    return Y

def solve():
    t0 = time.time()
    HARD = 174.0
    rng = np.random.default_rng(12345)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    it = 0
    stall = 0
    # schedules: fast (~600 steps) and intensive (~1000 steps)
    sched_fast = [(80.0, 180, 0.020), (220.0, 180, 0.010), (550.0, 200, 0.005)]
    sched_mid = [(120.0, 220, 0.015), (350.0, 220, 0.008), (800.0, 250, 0.004)]

    while time.time() - t0 < HARD - 16:
        it += 1
        tl = HARD - 16 - (time.time() - t0)
        if tl < 3.0:
            break
        try:
            mcur, cnts, G = _contacts(bestX, tol=1e-7)
        except Exception:
            mcur, cnts = bestV, np.zeros(n, dtype=int)
        order_lo = np.argsort(cnts, kind='stable')
        order_hi = np.argsort(-cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 6
        if mode == 5:
            sig = 0.08 + 0.08 * rng.random()
            Xtry = bestX + sig * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            if mode == 0: k = 1
            elif mode == 1: k = 1
            elif mode == 2: k = 2
            elif mode == 3: k = 2
            else: k = 3
            if stall > 10:
                k = min(4, k + 1)
            pool = order_lo[:20]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for idx in chosen:
                idx = int(idx)
                nc = 3000
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                Gc = cand @ Xtry.T
                Gc[:, idx] = -2.0
                scores = Gc.max(axis=1)
                bi = int(np.argmin(scores))
                if rng.random() < 0.30:
                    part = np.argpartition(scores, 5)[:5]
                    bi = int(rng.choice(part))
                Xtry[idx] = cand[bi]
            # nudge top hub tangentially if jammed
            try:
                hub = int(order_hi[0])
                if cnts[hub] > 22:
                    Xtry[hub] = Xtry[hub] + rng.standard_normal(d) * 0.025
                    Xtry[hub] /= np.linalg.norm(Xtry[hub])
                Xtry = Xtry + rng.standard_normal((n, d)) * 0.005
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            except Exception:
                pass
        # quench
        try:
            sched = sched_mid if (stall > 8 and mode % 2 == 0) else sched_fast
            Xq = _adam_quench(Xtry, sched)
        except Exception:
            stall += 1
            continue
        try:
            vq = _max_cos(Xq)
        except Exception:
            stall += 1
            continue
        # selective LP snap only if promising
        if vq < bestV + 2e-5:
            try:
                bdg = 0.6 if tl > 6 else 0.3
                Xp = _lp_polish(Xq, time.time() + bdg)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xq, vq
        else:
            Xp, vp = Xq, vq
        if vp < bestV - 1e-12:
            bestX = Xp; bestV = vp; stall = 0
        else:
            stall += 1
        if time.time() - t0 > HARD - 16:
            break

    # final intensive polish with remaining time
    try:
        rem = HARD - (time.time() - t0) - 1.0
        if rem > 1.5:
            bestX = _lp_polish(bestX, time.time() + min(rem * 0.6, 7.0))
            bestV = _max_cos(bestX)
            rem2 = HARD - (time.time() - t0) - 0.6
            if rem2 > 2.5:
                try:
                    Xq = _adam_quench(bestX, [(900.0, 250, 0.003)])
                    Xp = _lp_polish(Xq, time.time() + min(2.5, rem2 - 0.4))
                    if _max_cos(Xp) < bestV:
                        bestX = Xp
                except Exception:
                    pass
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_start, t_limit):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < t_limit and (time.time() - t_start) < seconds + 1e9:
        # use remaining budget
        if time.time() - t_start > t_limit:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
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
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
        # stop if out of time for this phase
        # outer caller controls total limit; check elapsed via t_start
    return x

def solve():
    t_start = time.time()
    T_LIMIT = 168.0
    rng = np.random.default_rng(0)
    n, d = 68, 12
    # load best
    try:
        bestX = np.load("best.npy")
    except Exception:
        try:
            bestX = np.load("record.npy")
        except Exception:
            bestX = rng.normal(size=(n, d))
    bestX = np.asarray(bestX, dtype=np.float64)
    if bestX.shape != (n, d):
        # fallback: random
        bestX = rng.normal(size=(n, d))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    best_score = _max_cos(bestX)

    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=np.float64)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        rs = _max_cos(rec)
        if rs < best_score:
            best_score = rs
            bestX = rec.copy()
    except Exception:
        pass

    # ---------- smooth annealed LSE refiner ----------
    def smooth_refine(Y0, steps, beta0, beta1, lr0, lr1, noise0=0.0, rng_=None):
        Y = Y0.copy()
        mmt = np.zeros_like(Y)
        vel = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        # precompute log ratio
        lb = np.log(beta1 / beta0) if beta1 != beta0 else 0.0
        for t in range(1, steps + 1):
            if time.time() - t_start > T_LIMIT - 2.0:
                break
            frac = t / steps
            beta = beta0 * np.exp(lb * frac)
            lr = lr0 + (lr1 - lr0) * frac
            ns = noise0 * (1.0 - frac)
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m0 = np.max(G)
            # weights exp(beta*(G-m0)), stable (<=0 arg)
            W = np.exp(beta * (G - m0))
            np.fill_diagonal(W, 0.0)
            S = np.sum(W) * 0.5
            if S <= 0 or not np.isfinite(S):
                break
            W /= S
            grad = W @ Y
            dot = np.sum(grad * Y, axis=1, keepdims=True)
            gt = grad - dot * Y
            # Adam
            mmt = b1 * mmt + (1.0 - b1) * gt
            vel = b2 * vel + (1.0 - b2) * (gt * gt)
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            mhat = mmt / bc1
            vhat = vel / bc2
            step = lr * mhat / (np.sqrt(vhat) + eps)
            # clip per-row step to avoid explosion
            sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            clip = 0.08
            scale = np.minimum(1.0, clip / (sn + 1e-12))
            step = step * scale
            Y = Y - step
            if ns > 0 and rng_ is not None:
                nz = rng_.normal(size=Y.shape)
                nz -= np.sum(nz * Y, axis=1, keepdims=True) * Y
                nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
                nz /= nn
                Y += ns * nz
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def find_hole(X, rng_, nsamp=12000):
        R = rng_.normal(size=(nsamp, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        dots = R @ X.T
        worst = np.max(dots, axis=1)
        idx = np.argmin(worst)
        return R[idx]

    # ---------- basin hopping ----------
    # reserve ~35s for final polish
    explore_deadline = t_start + 122.0
    trial = 0
    # quick diverse beta/lr settings
    while time.time() - t_start < explore_deadline - 1.0:
        if time.time() - t_start > T_LIMIT - 40.0:
            break
        trial += 1
        mode = trial % 7
        Xinit = None
        steps = 1400
        beta0, beta1 = 15.0, 280.0
        lr0, lr1 = 0.03, 0.002
        noise0 = 0.0
        if mode == 0:
            # pure random start
            Xinit = rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            steps = 2200
            beta0, beta1 = 12.0, 300.0
            lr0, lr1 = 0.04, 0.002
        elif mode == 1:
            sig = 0.04 + 0.02 * ((trial // 7) % 3)
            Xinit = bestX + sig * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            steps = 1500
            noise0 = 0.001
        elif mode == 2:
            sig = 0.12 + 0.05 * ((trial // 7) % 3)
            Xinit = bestX + sig * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            steps = 1700
            beta0, beta1 = 18.0, 350.0
        elif mode == 3:
            # hole re-insertion: replace 1-2 overloaded points
            Xinit = bestX.copy()
            G = Xinit @ Xinit.T
            np.fill_diagonal(G, -2.0)
            thr = best_score - 0.005
            cnt = np.sum(G > thr, axis=1)
            order = np.argsort(-cnt)
            k = 1 + (trial // 7) % 2  # 1 or 2
            for kk in range(k):
                victim = int(order[kk])
                h = find_hole(Xinit, rng, nsamp=10000)
                # small jitter
                h = h + 0.02 * rng.normal(size=d)
                h /= np.linalg.norm(h)
                Xinit[victim] = h
            steps = 1600
            beta0, beta1 = 15.0, 320.0
        elif mode == 4:
            # subset reinit: 5 random points random
            Xinit = bestX.copy()
            idx = rng.choice(n, size=5, replace=False)
            R = rng.normal(size=(5, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xinit[idx] = R
            steps = 1800
        elif mode == 5:
            sig = 0.30
            Xinit = bestX + sig * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            steps = 1900
            beta0, beta1 = 10.0, 250.0
            lr0, lr1 = 0.05, 0.003
        else:
            # large hole + noise: replace 3 points by holes
            Xinit = bestX.copy()
            G = Xinit @ Xinit.T
            np.fill_diagonal(G, -2.0)
            thr = best_score - 0.005
            cnt = np.sum(G > thr, axis=1)
            order = np.argsort(-cnt)
            for kk in range(3):
                victim = int(order[kk])
                h = find_hole(Xinit, rng, nsamp=8000)
                Xinit[victim] = h / np.linalg.norm(h)
            Xinit += 0.03 * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            steps = 1700
        Y = smooth_refine(Xinit, steps, beta0, beta1, lr0, lr1, noise0=noise0, rng_=rng)
        s = _max_cos(Y)
        if s < best_score and np.all(np.isfinite(Y)):
            best_score = s
            bestX = Y.copy()
        # early stop if big improvement? keep searching for more
        if trial > 90:
            break

    # ---------- intensify around best ----------
    if time.time() - t_start < T_LIMIT - 8.0:
        remain = T_LIMIT - (time.time() - t_start)
        # strong anneal with large beta
        steps = 3500 if remain > 50 else 1500
        Y = smooth_refine(bestX, steps, 25.0, 700.0, 0.015, 0.0008, noise0=0.0, rng_=rng)
        s = _max_cos(Y)
        if s < best_score:
            best_score = s
            bestX = Y.copy()

    # ---------- final LP polish ----------
    remain = T_LIMIT - (time.time() - t_start)
    if remain > 6.0:
        # give LP up to remain-2 seconds, capped 38
        lp_time = min(38.0, remain - 2.0)
        lp_deadline = time.time() + lp_time
        # _lp_polish uses t_start/T_LIMIT; emulate by passing deadline as t_limit
        # we call with seconds huge and t_limit = lp_deadline
        bestX = _lp_polish(bestX, 1e9, time.time() - 1e9 + 0, lp_deadline)
        # renormalize
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = _max_cos(bestX)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
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
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    T_END = t_start + 168.0
    rng = np.random.default_rng(12345)
    n, d = 68, 12
    try:
        bestX = np.load("best.npy")
    except Exception:
        try:
            bestX = np.load("record.npy")
        except Exception:
            bestX = rng.normal(size=(n, d))
    bestX = np.asarray(bestX, dtype=np.float64)
    if bestX.shape != (n, d):
        bestX = rng.normal(size=(n, d))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    best_score = _max_cos(bestX)
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=np.float64)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        rs = _max_cos(rec)
        if rs < best_score:
            best_score, bestX = rs, rec.copy()
    except Exception:
        pass

    def find_hole(X, nsamp):
        R = rng.normal(size=(nsamp, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        dots = R @ X.T
        worst = np.max(dots, axis=1)
        return R[int(np.argmin(worst))]

    def anneal(Y0, steps, mode, p0, p1, lr0, lr1, noise0=0.0, dropout=0.0, cycle=0.0):
        Y = Y0.copy()
        mmt = np.zeros_like(Y)
        vel = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lp = np.log(p1 / p0) if p1 != p0 else 0.0
        for t in range(1, steps + 1):
            if t % 128 == 0 and time.time() > T_END - 40.0:
                break
            frac = t / steps
            base = p0 * np.exp(lp * frac)
            if cycle > 0:
                base = base * (1.0 + cycle * np.sin(6.0 * np.pi * frac))
                base = max(base, 2.0)
            lr = lr0 + (lr1 - lr0) * frac
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            if mode == 'lse':
                beta = base
                m0 = float(np.max(G))
                W = np.exp(beta * (G - m0))
                np.fill_diagonal(W, 0.0)
                S = np.sum(W) * 0.5
                if not np.isfinite(S) or S <= 0:
                    break
                W /= S
            else:  # riesz
                s = base
                D2 = 2.0 - 2.0 * G
                np.fill_diagonal(D2, 1.0)
                np.clip(D2, 1e-4, None, out=D2)
                W = D2 ** (-s * 0.5 - 1.0)
                np.fill_diagonal(W, 0.0)
                S = np.sum(W) * 0.5
                if not np.isfinite(S) or S <= 0:
                    break
                W /= S
            if dropout > 0:
                M = rng.random((n, n)) > dropout
                M = np.triu(M, 1)
                M = M + M.T
                W = W * M
                S2 = np.sum(W) * 0.5
                if S2 > 1e-12:
                    W *= (S / S2) if 'S' in locals() else 1.0
            grad = W @ Y
            dot = np.sum(grad * Y, axis=1, keepdims=True)
            gt = grad - dot * Y
            mmt = b1 * mmt + (1.0 - b1) * gt
            vel = b2 * vel + (1.0 - b2) * (gt * gt)
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            mhat = mmt / bc1
            vhat = vel / bc2
            step = lr * mhat / (np.sqrt(vhat) + eps)
            sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            scale = np.minimum(1.0, 0.08 / sn)
            Y = Y - step * scale
            if noise0 > 0:
                ns = noise0 * (1.0 - frac)
                nz = rng.normal(size=Y.shape)
                nz -= np.sum(nz * Y, axis=1, keepdims=True) * Y
                nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
                Y += ns * (nz / nn)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def cap_kick(X, m, alpha):
        u = rng.normal(size=d); u /= np.linalg.norm(u)
        proj = X @ u
        idx = np.argsort(proj)[-m:]
        v = rng.normal(size=d); v -= (v @ u) * u
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = rng.normal(size=d); nv = np.linalg.norm(v)
        v /= nv
        X2 = X.copy()
        X2[idx] += alpha * v + 0.02 * rng.normal(size=(m, d))
        X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
        return X2

    explore_deadline = t_start + 118.0
    trial = 0
    while time.time() < explore_deadline and time.time() < T_END - 50.0:
        trial += 1
        kind = trial % 8
        Xinit = None
        mode, p0, p1, lr0, lr1, steps, noise0, dropout, cycle = 'lse', 15.0, 300.0, 0.03, 0.002, 1500, 0.0, 0.0, 0.0
        if kind == 0:  # jittered LSE
            sig = 0.05 + 0.03 * ((trial // 8) % 4)
            Xinit = bestX + sig * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            mode, p0, p1, lr0, lr1, steps, noise0 = 'lse', 15.0, 320.0, 0.03, 0.002, 1500, 0.001
        elif kind == 1:  # riesz from jitter
            sig = 0.08 + 0.05 * ((trial // 8) % 3)
            Xinit = bestX + sig * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            mode, p0, p1, lr0, lr1, steps = 'riesz', 4.0, 24.0, 0.03, 0.002, 1600
            dropout = 0.0
        elif kind == 2:  # random + riesz then lse chained inside? use riesz long
            Xinit = rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            mode, p0, p1, lr0, lr1, steps = 'riesz', 2.0, 18.0, 0.04, 0.003, 2200
        elif kind == 3:  # hole re-insertion of overloaded hubs
            Xinit = bestX.copy()
            G = Xinit @ Xinit.T; np.fill_diagonal(G, -2.0)
            thr = best_score - 0.005
            cnt = np.sum(G > thr, axis=1)
            order = np.argsort(-cnt)
            k = 2 + (trial // 8) % 3  # 2..4
            for kk in range(k):
                victim = int(order[kk])
                h = find_hole(np.delete(Xinit, victim, axis=0), 10000)
                h = h + 0.01 * rng.normal(size=d); h /= np.linalg.norm(h)
                Xinit[victim] = h
            mode, p0, p1, lr0, lr1, steps = 'lse', 12.0, 340.0, 0.035, 0.002, 1700
        elif kind == 4:  # coherent cap shift
            m = 4 + int(rng.integers(0, 7))  # 4..10
            alpha = 0.15 + 0.1 * ((trial // 8) % 4)  # 0.15..0.45
            Xinit = cap_kick(bestX, m, alpha)
            mode, p0, p1, lr0, lr1, steps = 'lse', 14.0, 300.0, 0.03, 0.002, 1600
            cycle = 0.3
        elif kind == 5:  # subset reinit
            Xinit = bestX.copy()
            kk = 5 + int(rng.integers(0, 4))  # 5..8
            idx = rng.choice(n, size=kk, replace=False)
            R = rng.normal(size=(kk, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xinit[idx] = R
            mode, p0, p1, lr0, lr1, steps = 'lse', 10.0, 280.0, 0.04, 0.002, 1800
            dropout = 0.05
        elif kind == 6:  # large jitter + dropout + cycling (basin jump)
            sig = 0.30
            Xinit = bestX + sig * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            mode, p0, p1, lr0, lr1, steps = 'lse', 10.0, 260.0, 0.05, 0.003, 1800
            dropout, cycle = 0.1, 0.4
        else:  # targeted hub kick along tangent away from neighbours
            Xinit = bestX.copy()
            G = Xinit @ Xinit.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 1e-7, axis=1)
            hub = int(np.argmax(cnt))
            neigh = np.where(G[hub] > m - 0.01)[0]
            if len(neigh) > 0:
                push = np.sum(Xinit[neigh], axis=0)
                push -= np.dot(push, Xinit[hub]) * Xinit[hub]
                npush = np.linalg.norm(push) + 1e-12
                push /= npush
                Xinit[hub] -= 0.25 * push
                Xinit[hub] /= np.linalg.norm(Xinit[hub])
            Xinit += 0.03 * rng.normal(size=(n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            mode, p0, p1, lr0, lr1, steps = 'lse', 15.0, 320.0, 0.03, 0.002, 1600
        Y = anneal(Xinit, steps, mode, p0, p1, lr0, lr1, noise0, dropout, cycle)
        # chain second stage with opposite energy for hybrid trials
        if kind in (1, 2) and time.time() < T_END - 55.0:
            Y = anneal(Y, 900, 'lse', 20.0, 500.0, 0.015, 0.001, 0.0, 0.0, 0.0)
        s = _max_cos(Y)
        if np.all(np.isfinite(Y)) and s < best_score:
            best_score = s
            bestX = Y.copy()
        if trial > 120:
            break

    if time.time() < T_END - 35.0:
        steps = 3000 if (T_END - time.time()) > 60 else 1500
        Y = anneal(bestX, steps, 'lse', 25.0, 800.0, 0.012, 0.0006, 0.0, 0.0, 0.0)
        s = _max_cos(Y)
        if s < best_score and np.all(np.isfinite(Y)):
            best_score, bestX = s, Y.copy()

    remain = T_END - time.time()
    if remain > 8.0:
        lp_time = min(34.0, remain - 2.0)
        bestX = _lp_polish(bestX, time.time() + lp_time)
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

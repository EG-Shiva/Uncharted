import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 >= budget:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            nrm = np.linalg.norm(y, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            y = y / nrm
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_polish(X, steps, T0, T1, lr0, lr1, mu=0.6):
    n, d = X.shape
    Y = X.copy()
    V = np.zeros_like(Y)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        row_sum = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(row_sum)
        mask = row_sum > 1e-12
        inv[mask] = 1.0 / row_sum[mask]
        avgWY = WY * inv[:, None]
        avgR = r * inv
        Dir = avgWY - avgR[:, None] * Y
        # freeze essentially isolated rows to avoid noise
        Dir[~mask] = 0.0
        V *= mu
        V += Dir
        Y -= lr * V
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y


def solve():
    t_start = time.time()
    rng = np.random.default_rng(0)
    np.random.seed(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (181, 9):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best = None
    best_val = np.inf
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    n, d = 181, 9
    cycle = 0
    # exploration phase, reserve ~35s for final LP
    while time.time() - t_start < 122.0 and cycle < 400:
        cycle += 1
        u = float(rng.random())
        cur = best.copy()
        # per-point worst for targeted modes
        if u >= 0.35:
            try:
                Gt = cur @ cur.T
                np.fill_diagonal(Gt, -2.0)
                row_max = np.max(Gt, axis=1)
                order = np.argsort(-row_max)
            except Exception:
                order = np.arange(n)
        if u < 0.40:
            # global small shake
            sigma = float(rng.uniform(0.008, 0.025))
            noise = rng.standard_normal((n, d)) * sigma
            # tangentialize
            proj = np.sum(noise * cur, axis=1, keepdims=True)
            noise -= proj * cur
            cur += noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.012, 0.025))
            T1 = float(rng.uniform(0.0008, 0.003))
            lr0 = float(rng.uniform(0.03, 0.07))
            lr1 = float(rng.uniform(0.003, 0.01))
            steps = 700
        elif u < 0.62:
            # shake worst cluster hard
            K = int(rng.integers(8, 18))
            idx = order[:K]
            sigma = float(rng.uniform(0.05, 0.12))
            noise = rng.standard_normal((K, d)) * sigma
            proj = np.sum(noise * cur[idx], axis=1, keepdims=True)
            noise -= proj * cur[idx]
            cur[idx] += noise
            cur[idx] /= np.linalg.norm(cur[idx], axis=1, keepdims=True)
            T0 = float(rng.uniform(0.015, 0.03))
            T1 = float(rng.uniform(0.0008, 0.0025))
            lr0 = float(rng.uniform(0.04, 0.08))
            lr1 = float(rng.uniform(0.003, 0.008))
            steps = 800
        elif u < 0.78:
            # random subset shake
            K = int(rng.integers(12, 30))
            idx = rng.choice(n, size=K, replace=False)
            sigma = float(rng.uniform(0.03, 0.08))
            noise = rng.standard_normal((K, d)) * sigma
            proj = np.sum(noise * cur[idx], axis=1, keepdims=True)
            noise -= proj * cur[idx]
            cur[idx] += noise
            cur[idx] /= np.linalg.norm(cur[idx], axis=1, keepdims=True)
            T0 = float(rng.uniform(0.01, 0.02))
            T1 = float(rng.uniform(0.001, 0.0035))
            lr0 = float(rng.uniform(0.03, 0.06))
            lr1 = float(rng.uniform(0.004, 0.01))
            steps = 700
        elif u < 0.90:
            # replace worst point by random far point then joint anneal
            idx = int(order[0])
            v = rng.standard_normal((d,))
            v /= np.linalg.norm(v)
            cur[idx] = v
            # also small global jitter
            sigma = 0.01
            noise = rng.standard_normal((n, d)) * sigma
            proj = np.sum(noise * cur, axis=1, keepdims=True)
            noise -= proj * cur
            cur += noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.015, 0.03))
            T1 = float(rng.uniform(0.001, 0.003))
            lr0 = float(rng.uniform(0.04, 0.07))
            lr1 = float(rng.uniform(0.003, 0.008))
            steps = 800
        else:
            # big global jump to new basin
            sigma = float(rng.uniform(0.10, 0.18))
            noise = rng.standard_normal((n, d)) * sigma
            proj = np.sum(noise * cur, axis=1, keepdims=True)
            noise -= proj * cur
            cur += noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.02, 0.035))
            T1 = float(rng.uniform(0.001, 0.004))
            lr0 = float(rng.uniform(0.05, 0.09))
            lr1 = float(rng.uniform(0.004, 0.012))
            steps = 900
        try:
            polished = _smooth_polish(cur, steps, T0, T1, lr0, lr1, mu=0.6)
            v = _max_cos(polished)
        except Exception:
            continue
        if v < best_val - 1e-12:
            # squeeze with short LP to get true local optimum
            remain = 122.0 - (time.time() - t_start)
            lp_budget = min(3.0, max(0.8, remain * 0.15)) if remain > 10 else 0.8
            try:
                if time.time() - t_start < 118.0:
                    refined = _lp_polish(polished, lp_budget)
                    vr = _max_cos(refined)
                else:
                    refined, vr = polished, v
            except Exception:
                refined, vr = polished, v
            if vr < best_val:
                best, best_val = refined, vr
        elif v < best_val + 0.0004 and (cycle % 7 == 0):
            # promising near-miss: try short LP occasionally
            if time.time() - t_start > 115.0:
                continue
            try:
                refined = _lp_polish(polished, 1.5)
                vr = _max_cos(refined)
                if vr < best_val - 1e-12:
                    best, best_val = refined, vr
            except Exception:
                pass
    # final squeeze with remaining time
    try:
        elapsed = time.time() - t_start
        remain = 175.0 - elapsed
        if remain > 2.0:
            budget = min(30.0, remain - 2.0)
            best = _lp_polish(best, budget)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

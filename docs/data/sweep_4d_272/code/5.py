import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def penalty_push(x, target, steps, eta0=0.12, mu=0.7):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        V = G - target
        np.maximum(V, 0.0, out=V)
        # early success check
        if V.max() <= 0:
            break
        W = V  # linear weighting
        grad = W @ x
        rs = W.sum(axis=1, keepdims=True)
        # average direction to avoid scale blowup
        grad = grad / (rs + 1e-9)
        # tangent projection
        pr = np.einsum('ij,ij->i', grad, x)[:, None]
        grad = grad - pr * x
        # clip extreme
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        np.minimum(gn, 2.0, out=gn)
        # avoid div by zero: scale only if >1?
        # keep as is
        vel = mu * vel + (1 - mu) * grad
        eta = eta0 * (1.0 - 0.7 * s / max(steps, 1))
        x = x - eta * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # tiny annealing noise early
        if s % 200 == 199:
            # check max
            m = float(G.max())
            if m <= target + 1e-12:
                break
    return x

def find_void(x, exclude, rng, M=15000):
    C = rng.standard_normal(size=(M, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ x.T  # M x n
    if exclude is not None:
        # exclude one or list
        if np.isscalar(exclude):
            D[:, exclude] = -2.0
        else:
            D[:, np.array(exclude)] = -2.0
    m = D.max(axis=1)
    return C[np.argmin(m)]

def worst_points(x, k=1):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    # per point max and contact count
    rowmax = G.max(axis=1)
    contacts = (G > m - 1e-4).sum(axis=1)
    order = np.lexsort((-contacts, -rowmax))
    # order[0] is worst
    return order[:k], float(m)

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (272, 4):
        # try to fix by trunc/pad with random
        n0 = xb.shape[0]
        if n0 > 272:
            xb = xb[:272]
        else:
            extra = rng.standard_normal(size=(272 - n0, 4))
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            xb = np.vstack([xb, extra])
    best = xb.copy()
    best_val = max_cos(best)
    it = 0
    sigmas = [0.015, 0.04, 0.09, 0.18, 0.30]
    deltas = [0.0015, 0.003, 0.005, 0.008]
    # exploration phase, reserve ~30s for final LP
    while time.time() - t0 < TIME_LIMIT - 32.0:
        it += 1
        mode = it % 5
        if mode in (0, 1, 2):
            sigma = sigmas[(it // 5 + mode) % len(sigmas)]
            # occasionally large kick
            if it % 23 == 0:
                sigma = 0.35
            delta = deltas[it % len(deltas)]
            # start from best
            x0 = best + sigma * rng.standard_normal(size=best.shape)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            target = best_val - delta
            # clamp target not too low
            if target < 0.80:
                target = best_val - 0.002
            steps = 700 if mode != 2 else 1000
            eta0 = 0.15 if sigma < 0.1 else 0.10
            x1 = penalty_push(x0.copy(), target, steps, eta0=eta0, mu=0.75)
            v1 = max_cos(x1)
            if v1 < best_val:
                best, best_val = x1, v1
                # quick squeeze with short LP if big win to lock in
                if delta > 0.002 and time.time() - t0 < TIME_LIMIT - 60:
                    try:
                        x2 = lp_polish(best.copy(), 3.0)
                        v2 = max_cos(x2)
                        if v2 < best_val:
                            best, best_val = x2, v2
                    except Exception:
                        pass
            # also try small random hill around x1 even if not better: second push to best_val
            elif v1 < best_val + 0.002 and (it % 7 == 0):
                # continue pushing x1 toward best_val
                x2 = penalty_push(x1, best_val - 0.001, 400, eta0=0.08)
                v2 = max_cos(x2)
                if v2 < best_val:
                    best, best_val = x2, v2
        elif mode == 3:
            # void reinsertion of worst point
            try:
                widx, _ = worst_points(best, k=1)
                wi = int(widx[0])
                void = find_void(best, wi, rng, M=15000)
                x0 = best.copy()
                # jitter neighbours slightly to open hole
                x0[wi] = void
                # small jitter of top-5 worst
                w5, _ = worst_points(best, k=6)
                for jj in w5[1:3]:
                    x0[int(jj)] += 0.02 * rng.standard_normal(size=4)
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                target = best_val - deltas[it % len(deltas)] * 0.7
                x1 = penalty_push(x0, target, 800, eta0=0.12)
                v1 = max_cos(x1)
                if v1 < best_val:
                    best, best_val = x1, v1
            except Exception:
                pass
        else:
            # double replacement: two worst -> two voids
            try:
                widx, _ = worst_points(best, k=2)
                x0 = best.copy()
                # find first void
                v1pos = find_void(best, int(widx[0]), rng, M=12000)
                # temp set for second search
                tmp = best.copy()
                tmp[int(widx[0])] = v1pos
                v2pos = find_void(tmp, int(widx[1]), rng, M=12000)
                x0[int(widx[0])] = v1pos
                x0[int(widx[1])] = v2pos
                # jitter a bit
                x0 += 0.01 * rng.standard_normal(size=x0.shape)
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                target = best_val - 0.002
                x1 = penalty_push(x0, target, 800, eta0=0.12)
                v1 = max_cos(x1)
                if v1 < best_val:
                    best, best_val = x1, v1
            except Exception:
                pass
        # time escape
        if time.time() - t0 > TIME_LIMIT - 32.0:
            break
    # final squeeze
    remain = TIME_LIMIT - (time.time() - t0) - 1.0
    if remain > 5.0:
        try:
            # give most to LP
            lp_time = min(remain, 28.0)
            best = lp_polish(best, lp_time)
            best_val = max_cos(best)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: remove nan
    best = np.nan_to_num(best, nan=1.0, posinf=1.0, neginf=-1.0)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

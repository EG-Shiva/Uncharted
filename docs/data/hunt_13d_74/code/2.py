import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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


def adam_smooth(x_init, steps, tau0, tau1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    iu = np.triu_indices(n, 1)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # preallocate W
    for t in range(1, steps + 1):
        frac = t / steps
        tau = tau0 * (tau1 / tau0) ** frac if tau0 != tau1 else tau0
        G = x @ x.T
        gv = G[iu]
        gmax = gv.max()
        # stable softmax
        e = np.exp((gv - gmax) / tau)
        S = e.sum()
        if S == 0 or not np.isfinite(S):
            break
        w = e / S
        W = np.zeros((n, n), dtype=x.dtype)
        W[iu] = w
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def solve():
    rng = np.random.default_rng(0)
    n, d = 74, 13
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            x = np.load(f).astype(np.float64)
            if x.shape == (n, d):
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
                vv = max_cos(x)
                if vv < best_val:
                    best_val = vv
                    best = x.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)

    t_start = time.time()
    BUDGET = 168.0
    it = 0
    iu_all = np.triu_indices(n, 1)
    while True:
        tl = BUDGET - (time.time() - t_start)
        if tl < 3.0:
            break
        it += 1
        if it == 1:
            x_try = best.copy()
            steps, tau0, tau1, lr = 1200, 0.008, 0.0015, 0.002
            lp_t = min(4.0, tl - 0.5)
        else:
            r = rng.random()
            if r < 0.6:
                sigmas = np.array([0.003, 0.007, 0.015, 0.03, 0.06])
                sigma = float(rng.choice(sigmas))
                noise = rng.standard_normal(best.shape)
                x_try = best + sigma * noise
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                steps, tau0, tau1, lr = 900, 0.012, 0.002, 0.0025
                lp_t = min(2.5, tl - 0.5)
            else:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                gv = G[iu_all]
                K = min(300, gv.size)
                idx = np.argpartition(-gv, K)[:K]
                Ii = iu_all[0][idx]
                Ji = iu_all[1][idx]
                cnt = np.zeros(n, dtype=int)
                np.add.at(cnt, Ii, 1)
                np.add.at(cnt, Ji, 1)
                nw = int(rng.integers(1, 4))
                worst = np.argsort(-cnt)[:nw]
                x_try = best.copy()
                rp = rng.standard_normal((nw, d))
                rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                x_try[worst] = rp
                x_try = x_try + 0.005 * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                steps, tau0, tau1, lr = 1200, 0.012, 0.002, 0.003
                lp_t = min(2.5, tl - 0.5)
            if tl < 6.0:
                steps = min(steps, 400)
                lp_t = min(lp_t, 1.0)
        try:
            x_try = adam_smooth(x_try, steps, tau0, tau1, lr)
        except Exception:
            continue
        if lp_t > 0.3:
            try:
                x_try = lp_polish(x_try, lp_t)
            except Exception:
                pass
        try:
            vv = max_cos(x_try)
        except Exception:
            continue
        if vv < best_val - 1e-12 and np.all(np.isfinite(x_try)):
            best = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
            best_val = vv
            # early exit if huge gain? continue to push further
        if time.time() - t_start >= BUDGET:
            break
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

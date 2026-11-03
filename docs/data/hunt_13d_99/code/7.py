import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def smooth_gd(Y, steps, a0, a1, lr0, lr1, beta=0.9, seed_mom=True):
    n, d = Y.shape
    cur = Y.copy()
    vel = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # exp in stable way
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-18
        Grad = (W @ cur) / rs
        # tangential projection
        dot = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - dot * cur
        # momentum with bias correction
        vel = beta * vel + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = vel / bc
        cur = cur - lr * step
        nr = np.linalg.norm(cur, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        cur /= nr
    return cur

def far_replace(base, idx_list, rng, n_cand=3000):
    X = base.copy()
    n, d = X.shape
    for k in idx_list:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(X, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        j = int(np.argmin(md))
        X[k] = cand[j]
    return X

def lp_polish(x, seconds, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cv = cfull[sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            # keep largest
            ord_ = np.argpartition(-cv, cap)[:cap]
            I = I[ord_]; J = J[ord_]; cv = cv[ord_]
            p = cap
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else [x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cv, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 132.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val

    sigmas = [0.01, 0.03, 0.06, 0.10, 0.16, 0.25, 0.38]
    it = 0
    # temperature schedule for basin hopping
    T0, T1 = 0.004, 0.0003

    while time.time() - t0 < EXPLORE and it < 2000:
        it += 1
        frac = (time.time() - t0) / EXPLORE
        Temp = T0 * (1 - frac) + T1 * frac
        r = rng.random()
        # choose base: mostly current walk, sometimes global best
        if rng.random() < 0.22:
            base = best_x
        else:
            base = cur_x
        # propose
        if r < 0.50:
            sigma = sigmas[rng.integers(0, len(sigmas))]
            noise = rng.standard_normal(size=(n, d)) * sigma
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif r < 0.82:
            # worst-k far replacement
            ppm = per_point_max(base)
            order = np.argsort(-ppm)
            k = int(rng.integers(1, 6))  # 1..5
            idx = order[:k].tolist()
            x0 = far_replace(base, idx, rng, n_cand=3000)
            # small shake to rest to break symmetry
            sj = rng.random() * 0.02
            if sj > 0:
                x0 = x0 + rng.standard_normal(size=(n, d)) * sj
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # subset strong shake of random 8-15 points
            m = int(rng.integers(6, 16))
            sel = rng.choice(n, size=m, replace=False)
            x0 = base.copy()
            sigma = [0.2, 0.35, 0.55][rng.integers(0, 3)]
            x0[sel] += rng.standard_normal(size=(m, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # choose smoothing hyperparams
        pick = rng.random()
        if pick < 0.4:
            a0, a1 = 25.0, 90.0
            lr0, lr1 = 0.08, 0.015
            steps = 700
        elif pick < 0.7:
            a0, a1 = 40.0, 140.0
            lr0, lr1 = 0.05, 0.008
            steps = 900
        else:
            a0, a1 = 15.0, 60.0
            lr0, lr1 = 0.12, 0.02
            steps = 600
        # adapt steps to remaining time
        if time.time() - t0 > EXPLORE - 10:
            steps = min(steps, 400)

        x1 = smooth_gd(x0, steps, a0, a1, lr0, lr1)
        v1 = max_cos(x1)

        if v1 < best_val:
            best_val = v1
            best_x = x1.copy()
            cur_x = x1.copy()
            cur_val = v1
        else:
            # accept into walk?
            if v1 < cur_val or rng.random() < np.exp(-(v1 - cur_val) / max(Temp, 1e-9)):
                cur_x = x1.copy()
                cur_val = v1
            # occasional random restart of walk to best if drifted too far
            if cur_val > best_val + 0.03 and rng.random() < 0.3:
                cur_x = best_x.copy()
                cur_val = best_val

    # final polish of best with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 5:
        # short extra gd at high alpha before LP
        try:
            xf = smooth_gd(best_x, 800, 60.0, 180.0, 0.03, 0.005)
            if max_cos(xf) < best_val:
                best_x = xf
                best_val = max_cos(xf)
        except Exception:
            pass
        remain = BUDGET - (time.time() - t0)
        if remain > 3:
            try:
                best_x = lp_polish(best_x, remain - 1.0)
            except Exception:
                pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

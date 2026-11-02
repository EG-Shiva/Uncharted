import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, radius_init=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 40:
        it += 1
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt_sel = int(np.sum(sel))
        if cnt_sel == 0:
            break
        if cnt_sel > 1500:
            # keep tightest 1500
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def target_quench(X_init, T, steps, lr0, lr1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        p = 2.0 + 5.0 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - T
        W = np.zeros_like(G)
        pos = D > 0
        if np.any(pos):
            W[pos] = p * (D[pos] ** (p - 1.0))
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def contact_counts(x, tol=2e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    return cnt, m

def greedy_hole_fill(X_keep, n_new, rng, n_cand=1500):
    d = X_keep.shape[1]
    cur = X_keep.copy()
    out = np.empty((n_new, d))
    for k in range(n_new):
        Z = rng.standard_normal((n_cand, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ cur.T
        smax = S.max(axis=1)
        best_i = int(np.argmin(smax))
        out[k] = Z[best_i]
        cur = np.vstack([cur, out[k:k+1]])
    return out

def solve():
    t0 = time.time()
    HARD = 155.0
    rng = np.random.default_rng(1)
    np.random.seed(1)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # quick initial jam
    xb = lp_polish(xb, seconds=2.0)
    best_val = max_cos(xb)
    X_best = xb.copy()
    it = 0
    while time.time() - t0 < HARD - 12:
        it += 1
        # target gap
        r = rng.random()
        if it % 10 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(3e-3))))
        elif it % 4 == 0:
            delta = float(np.exp(rng.uniform(np.log(2e-4), np.log(8e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(3e-5), np.log(3e-4))))
        T = best_val - delta
        cnt, cur_m = contact_counts(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        mode = it % 6
        X = X_best.copy()
        if mode == 0:
            n_rep = int(rng.integers(4, 8))
            victims = order_low[:n_rep].copy()
            keep_idx = np.array([i for i in range(n) if i not in set(victims.tolist())])
            X_keep = X[keep_idx]
            new_pts = greedy_hole_fill(X_keep, len(victims), rng, n_cand=1500)
            for k, v in enumerate(victims):
                X[v] = new_pts[k]
            # small jitter on neighbours of hubs
            sig = 0.01
            X += rng.standard_normal(X.shape) * sig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, lr0, lr1 = 500, 0.02, 0.002
        elif mode == 1:
            n_rep = int(rng.integers(2, 5))
            # mix rattlers + random
            cand = np.concatenate([order_low[:3], rng.choice(n, size=n_rep, replace=False)])
            victims = np.unique(cand)[:n_rep]
            keep_idx = np.array([i for i in range(n) if i not in set(victims.tolist())])
            X_keep = X[keep_idx]
            new_pts = greedy_hole_fill(X_keep, len(victims), rng, n_cand=2000)
            for k, v in enumerate(victims):
                X[v] = new_pts[k]
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, lr0, lr1 = 450, 0.025, 0.002
        elif mode == 2:
            # shake hubs
            n_sh = int(rng.integers(5, 12))
            victims = order_high[:n_sh].copy()
            sig = float(np.exp(rng.uniform(np.log(0.02), np.log(0.08))))
            X[victims] += rng.standard_normal((len(victims), d)) * sig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, lr0, lr1 = 400, 0.02, 0.003
        elif mode == 3:
            # global small shake
            sig = float(np.exp(rng.uniform(np.log(0.005), np.log(0.03))))
            X += rng.standard_normal(X.shape) * sig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, lr0, lr1 = 350, 0.015, 0.002
        elif mode == 4:
            # single worst pair break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            v = int(ij[rng.integers(0, 2)])
            # reinsert v into hole
            mask = np.ones(n, dtype=bool); mask[v] = False
            X_keep = X[mask]
            new_pts = greedy_hole_fill(X_keep, 1, rng, n_cand=2500)
            X[v] = new_pts[0]
            sig = 0.008
            X += rng.standard_normal(X.shape) * sig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, lr0, lr1 = 450, 0.02, 0.002
        else:
            n_rep = int(rng.integers(6, 11))
            victims = rng.choice(n, size=n_rep, replace=False)
            keep_idx = np.array([i for i in range(n) if i not in set(victims.tolist())])
            X_keep = X[keep_idx]
            new_pts = greedy_hole_fill(X_keep, len(victims), rng, n_cand=1200)
            for k, v in enumerate(victims):
                X[v] = new_pts[k]
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, lr0, lr1 = 550, 0.03, 0.003

        # anneal toward target
        Xq = target_quench(X, T, steps=steps, lr0=lr0, lr1=lr1)
        # quick check before expensive LP
        vq = max_cos(Xq)
        # only LP if promising (within 2e-3 of best or better than best-1e-5 handled)
        # to save time, skip LP if far worse
        if vq > best_val + 0.004 and (rng.random() > 0.15):
            continue
        time_left = HARD - (time.time() - t0)
        if time_left < 4:
            break
        pol_sec = min(1.2, time_left - 2)
        if pol_sec < 0.3:
            break
        Xp = lp_polish(Xq, seconds=pol_sec)
        vp = max_cos(Xp)
        if vp < best_val - 1e-12:
            best_val = vp
            X_best = Xp.copy()
        # opportunistic: also keep near-best diverse base occasionally
        elif vp < best_val + 0.0005 and rng.random() < 0.05:
            # random restart base to escape basin without losing best
            pass
    # final polish of best within remaining budget (cap 5s)
    rem = HARD - (time.time() - t0)
    if rem > 1.0:
        X_best = lp_polish(X_best, seconds=min(5.0, rem - 0.5))
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_polish(x, steps=380, a0=25.0, a1=420.0, lr=0.003):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr

def lp_polish(X, seconds, radius_init=1e-3, cap=3000):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mval = float(c_all.max())
        thresh = mval - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPLORE_END = t0 + BUDGET - 18.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass

    it = 0
    n, d = 176, 9
    while time.time() < EXPLORE_END:
        it += 1
        cand = best_x.copy()
        mode = it % 6
        if mode == 1:
            cand = cand + 0.025 * rng.standard_normal(cand.shape)
        elif mode == 2:
            cand = cand + 0.09 * rng.standard_normal(cand.shape)
        elif mode == 3 or mode == 5:
            # hub knockout into voids
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            contacts = np.sum(G > M - 5e-4, axis=1)
            per_max = G.max(axis=1)
            # score hubs: contacts primary
            score = contacts.astype(float) * 10.0 + (per_max - M + 0.01) * 100.0
            # add small random tie break
            score = score + 0.5 * rng.standard_normal(n)
            k = 3 + (it % 4)  # 3..6
            if mode == 5:
                k = 2 + (it % 3)
            idx = np.argpartition(-score, k)[:k]
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            cur = cand[mask]
            # order idx randomly for diversity
            order = rng.permutation(idx)
            for j in order:
                trials = rng.standard_normal((768, d))
                trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                mm = (trials @ cur.T).max(axis=1)
                pick = int(np.argmin(mm))
                cand[j] = trials[pick]
                cur = np.vstack([cur, cand[j][None, :]])
            cand = cand + 0.015 * rng.standard_normal(cand.shape)
        elif mode == 4:
            sel = rng.choice(n, size=28, replace=False)
            cand[sel] = cand[sel] + 0.22 * rng.standard_normal((28, d))
        else:
            # global moderate shake + one void reinsert of worst point
            cand = cand + 0.05 * rng.standard_normal(cand.shape)
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            j = int(np.argmax(G.max(axis=1)))
            trials = rng.standard_normal((768, d))
            trials /= np.linalg.norm(trials, axis=1, keepdims=True)
            rest_mask = np.ones(n, dtype=bool); rest_mask[j] = False
            rest = cand[rest_mask]
            mm = (trials @ rest.T).max(axis=1)
            cand[j] = trials[int(np.argmin(mm))]

        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        lr = 0.0025 + 0.001 * ((it // 6) % 3)
        a0 = 25.0 if (it % 2 == 0) else 40.0
        cand_s = smooth_polish(cand, steps=380, a0=a0, a1=420.0, lr=lr)
        v = max_cos(cand_s)
        if v < best_val - 1e-12:
            best_val = v
            best_x = cand_s.copy()
        # safety time check
        if time.time() >= EXPLORE_END:
            break

    rem = BUDGET - (time.time() - t0)
    rem = min(max(rem - 0.5, 0.0), 16.0)
    if rem > 1.0:
        best_x = lp_polish(best_x, seconds=rem, radius_init=1e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

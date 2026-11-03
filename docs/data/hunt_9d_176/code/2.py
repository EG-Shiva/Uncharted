import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def smooth_polish(X, steps=1200, a0=40.0, a1=350.0, lr=0.003):
    x = X.copy()
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
        # clip extreme steps
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


def lp_polish(X, seconds, radius_init=1e-3):
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
        # cap size for speed
        cnt = int(np.count_nonzero(sel))
        if cnt > 3000:
            # keep top 3000
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], 3000)[:3000]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = 3000
        if cnt == 0:
            break
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
    TIME_BUDGET = 168.0
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
        # try to reshape / truncate
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    # also consider record.npy as alternative start
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
    # main basin-hopping loop, reserve ~10s for final polish
    while time.time() - t0 < TIME_BUDGET - 10.0:
        it += 1
        remaining = TIME_BUDGET - 10.0 - (time.time() - t0)
        if remaining < 1.0:
            break
        mode = it % 4
        cand = best_x.copy()
        if mode == 1:
            cand = cand + 0.02 * rng.standard_normal(cand.shape)
        elif mode == 2:
            cand = cand + 0.08 * rng.standard_normal(cand.shape)
        elif mode == 3:
            # knock out k worst points and reinsert in voids
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            k = 1 + ((it // 4) % 3)  # 1..3
            idx = np.argpartition(-per_max, k)[:k]
            mask = np.ones(176, dtype=bool)
            mask[idx] = False
            rest = cand[mask]
            for j in idx:
                trials = rng.standard_normal((256, 9))
                trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                mm = (trials @ rest.T).max(axis=1)
                cand[j] = trials[np.argmin(mm)]
            cand = cand + 0.01 * rng.standard_normal(cand.shape)
        else:
            sel = rng.choice(176, size=24, replace=False)
            cand[sel] = cand[sel] + 0.15 * rng.standard_normal((24, 9))

        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # fast smooth repulsion
        # vary lr/steps slightly for diversity
        lr = 0.0025 + 0.001 * ((it // 4) % 3)
        cand_s = smooth_polish(cand, steps=1000, a0=40.0, a1=350.0, lr=lr)
        v = max_cos(cand_s)
        if v < best_val + 0.002:
            # short LP refinement if time allows
            if time.time() - t0 > TIME_BUDGET - 10.0:
                break
            cand2 = lp_polish(cand_s, seconds=2.2, radius_init=5e-3)
            v2 = max_cos(cand2)
            if v2 < best_val - 1e-12:
                best_val = v2
                best_x = cand2.copy()
        # occasional medium shake of current best without LP to save time
        if it % 20 == 0 and time.time() - t0 < TIME_BUDGET - 12:
            # opportunistic intensification: tiny shake + smooth
            tmp = best_x + 0.005 * rng.standard_normal(best_x.shape)
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            tmp = smooth_polish(tmp, steps=600, a0=80.0, a1=500.0, lr=0.0015)
            vt = max_cos(tmp)
            if vt < best_val + 0.001:
                tmp2 = lp_polish(tmp, seconds=1.5, radius_init=2e-3)
                vtt = max_cos(tmp2)
                if vtt < best_val - 1e-12:
                    best_val = vtt
                    best_x = tmp2.copy()

    # final polish of incumbent with remaining time
    rem = TIME_BUDGET - (time.time() - t0)
    rem = min(max(rem - 0.5, 0.0), 9.0)
    if rem > 1.0:
        best_x = lp_polish(best_x, seconds=rem, radius_init=1e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def adam_lse(x, steps=600, a0=40.0, a1=320.0, lr=0.003):
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
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def hard_squeeze(x, steps=400, lr0=0.002, m0=0.01, m1=2e-4):
    n, d = x.shape
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        lr = lr0 * (1.0 - 0.7 * frac)
        margin = m0 * ((m1 / m0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        mask = G > (M - margin)
        # degree per point
        deg = mask.sum(axis=1, keepdims=True)
        Grad = mask @ x
        # average to avoid hub blowup
        Grad = Grad / np.maximum(deg, 1.0)
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip
        nm = np.linalg.norm(Gt, axis=1, keepdims=True)
        # normalize step scale
        x = x - lr * Gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def lp_polish(X, seconds, radius_init=2e-3, cap=2000):
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
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = cnt; nv = n * d + 1
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    LP_RESERVE = 14.0
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
    # LP-free hopping phase
    while True:
        it += 1
        now = time.time() - t0
        if now > BUDGET - LP_RESERVE:
            break
        # time guard: estimate remaining trials
        mode = it % 5
        cand = best_x.copy()
        if mode == 1:
            cand = cand + 0.015 * rng.standard_normal(cand.shape)
        elif mode == 2:
            cand = cand + 0.05 * rng.standard_normal(cand.shape)
        elif mode == 3:
            # knockout worst 1-2 into voids
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            k = 1 + (it % 2)  # 1 or 2
            idx = np.argpartition(-per, k)[:k]
            mask = np.ones(176, dtype=bool)
            mask[idx] = False
            rest = cand[mask]
            for j in idx:
                trials = rng.standard_normal((512, 9))
                trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                mm = (trials @ rest.T).max(axis=1)
                cand[j] = trials[np.argmin(mm)]
                rest = np.vstack([rest, cand[j][None, :]])
            cand = cand + 0.008 * rng.standard_normal(cand.shape)
        elif mode == 4:
            sel = rng.choice(176, size=16, replace=False)
            cand[sel] = cand[sel] + 0.12 * rng.standard_normal((16, 9))
        else:
            cand = cand + 0.03 * rng.standard_normal(cand.shape)

        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # cheap quench: fewer steps for speed
        lr = 0.0025 + 0.0005 * (it % 3)
        cand = adam_lse(cand, steps=450, a0=40.0, a1=320.0, lr=lr)
        cand = hard_squeeze(cand, steps=300, lr0=0.0015, m0=0.008, m1=2e-4)
        v = max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            best_x = cand.copy()
        # opportunistic micro intensification every 25 iters
        if it % 25 == 0:
            if time.time() - t0 > BUDGET - LP_RESERVE:
                break
            tmp = best_x + 0.004 * rng.standard_normal(best_x.shape)
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            tmp = adam_lse(tmp, steps=300, a0=80.0, a1=500.0, lr=0.0015)
            tmp = hard_squeeze(tmp, steps=300, lr0=0.001, m0=0.003, m1=5e-5)
            vt = max_cos(tmp)
            if vt < best_val - 1e-12:
                best_val = vt
                best_x = tmp.copy()

    # final squeeze of incumbent without LP first
    rem = BUDGET - (time.time() - t0)
    if rem > LP_RESERVE + 1.0:
        # extra hard polish
        best_x = hard_squeeze(best_x, steps=500, lr0=0.0008, m0=0.002, m1=1e-5)
        bv = max_cos(best_x)
        if bv < best_val:
            best_val = bv
        rem = BUDGET - (time.time() - t0)
    # single final LP polish with remaining time
    rem = BUDGET - (time.time() - t0)
    use = min(max(rem - 0.5, 0.0), LP_RESERVE)
    if use > 2.0:
        try:
            xp = lp_polish(best_x, seconds=use, radius_init=1.5e-3, cap=2000)
            vp = max_cos(xp)
            if vp < best_val and np.all(np.isfinite(xp)):
                best_x, best_val = xp, vp
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

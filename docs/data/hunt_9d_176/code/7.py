import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def smooth_polish(x, steps=800, a0=40.0, a1=350.0, lr=0.003):
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


def lp_polish(X, seconds, radius_init=3e-3, cap=2800):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
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
        I = iu0[0][sel]
        J = iu0[1][sel]
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
        if time.time() - start >= seconds:
            break
    return x


def point_stress(x, alpha=150.0):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    M = float(np.max(G))
    E = np.exp(alpha * (G - M))
    np.fill_diagonal(E, 0.0)
    return E.sum(axis=1), M


def knockout_void_reinsert(cand, k, rng, ntrials=1024):
    n, d = cand.shape
    stress, _ = point_stress(cand, alpha=150.0)
    idx = np.argpartition(-stress, k)[:k]
    # order by stress descending
    idx = idx[np.argsort(-stress[idx])]
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    # current set of fixed points
    cur = cand[mask].copy()
    for j in idx:
        trials = rng.standard_normal((ntrials, d))
        trials /= np.linalg.norm(trials, axis=1, keepdims=True)
        dots = trials @ cur.T
        mm = dots.max(axis=1)
        pick = int(np.argmin(mm))
        cand[j] = trials[pick]
        cur = np.vstack([cur, cand[j][None, :]])
    return cand


def solve():
    t0 = time.time()
    BUDGET = 170.0
    FINAL_RESERVE = 11.0
    rng = np.random.default_rng(12345)
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
    while time.time() - t0 < BUDGET - FINAL_RESERVE:
        it += 1
        if time.time() - t0 > BUDGET - FINAL_RESERVE - 0.8:
            break
        mode = it % 5
        cand = best_x.copy()
        if mode == 0:
            cand = cand + 0.015 * rng.standard_normal(cand.shape)
        elif mode == 1:
            cand = cand + 0.06 * rng.standard_normal(cand.shape)
        elif mode == 2:
            k = 2 + (it // 5) % 3  # 2..4
            if it % 2 == 0:
                k = 1 + (it // 5) % 2
            cand = knockout_void_reinsert(cand, k, rng, ntrials=1024)
            cand = cand + 0.008 * rng.standard_normal(cand.shape)
        elif mode == 3:
            # localized cap shake: pick random direction, shake nearest m points harder
            u = rng.standard_normal(9)
            u /= np.linalg.norm(u)
            dots = cand @ u
            m = 12 + int(rng.integers(0, 14))  # 12..25
            sel = np.argpartition(-dots, m)[:m]
            cand[sel] = cand[sel] + 0.09 * rng.standard_normal((m, 9))
            cand = cand + 0.008 * rng.standard_normal(cand.shape)
        else:
            sel = rng.choice(176, size=20, replace=False)
            cand[sel] = cand[sel] + 0.14 * rng.standard_normal((20, 9))

        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # annealed quench, vary params slightly
        lr = 0.0022 + 0.0006 * ((it // 5) % 3)
        cand_s = smooth_polish(cand, steps=750, a0=35.0, a1=320.0, lr=lr)
        v = max_cos(cand_s)
        if v < best_val + 0.005 and v >= best_val + 0.0012:
            # cheap second-stage fine anneal without LP
            cand_s = smooth_polish(cand_s, steps=350, a0=120.0, a1=600.0, lr=0.0012)
            v = max_cos(cand_s)
        if v < best_val + 0.0012:
            if time.time() - t0 > BUDGET - FINAL_RESERVE - 0.5:
                break
            # short LP quench
            cand2 = lp_polish(cand_s, seconds=1.3, radius_init=4e-3)
            v2 = max_cos(cand2)
            if v2 < best_val - 1e-12:
                best_val = v2
                best_x = cand2.copy()
        if it % 25 == 0:
            if time.time() - t0 > BUDGET - FINAL_RESERVE - 2.0:
                break
            tmp = best_x + 0.004 * rng.standard_normal(best_x.shape)
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            tmp = smooth_polish(tmp, steps=500, a0=80.0, a1=550.0, lr=0.0012)
            vt = max_cos(tmp)
            if vt < best_val + 0.0008:
                tmp2 = lp_polish(tmp, seconds=1.2, radius_init=2e-3)
                vtt = max_cos(tmp2)
                if vtt < best_val - 1e-12:
                    best_val = vtt
                    best_x = tmp2.copy()

    rem = BUDGET - (time.time() - t0)
    rem = min(max(rem - 0.6, 0.0), FINAL_RESERVE - 1.0)
    if rem > 1.0:
        # final squeeze: alternate smooth-high-alpha + LP
        try:
            best_x = smooth_polish(best_x, steps=400, a0=150.0, a1=700.0, lr=0.001)
        except Exception:
            pass
        best_x = lp_polish(best_x, seconds=rem, radius_init=1e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

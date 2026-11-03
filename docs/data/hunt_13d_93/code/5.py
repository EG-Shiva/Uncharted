import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _local_opt(x, steps_adam=1100, steps_hard=250, lr0=0.004, tau0=0.015, tau1=0.0025):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    # soft-max Adam
    for s in range(steps_adam):
        frac = s / max(1, steps_adam - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 * (0.3 + 0.7 * (1.0 - frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = G.max()
        # stabilized weights
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        # tangential projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        x = x - step
        x = _normalize(x)
    # hard max-descent on active set
    step_size = 0.003
    cur = _max_cos(x)
    for _ in range(steps_hard):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        margin = 0.0012
        mask = G > (cur_max - margin)
        # grad = sum of active neighbours
        grad = mask.astype(np.float64) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        # avoid division by zero; rattlers have 0 grad -> no move
        direc = np.zeros_like(gt)
        nz = (rn[:, 0] > 1e-12)
        direc[nz] = gt[nz] / rn[nz]
        if not np.any(nz):
            break
        xtry = _normalize(x - step_size * direc)
        vtry = _max_cos(xtry)
        if vtry < cur_max:
            x = xtry
            cur = vtry
            step_size = min(step_size * 1.2, 0.005)
        else:
            step_size *= 0.5
            if step_size < 1e-7:
                break
    return x


def _lp_polish(x, seconds):
    x = _normalize(x.copy())
    n, d = x.shape
    best = _max_cos(x)
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
        y = _normalize(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    rng = np.random.default_rng(7)
    np.random.seed(0)
    best_x = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (93, 13):
                a = _normalize(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _normalize(rng.standard_normal((93, 13)))
        best_val = _max_cos(best_x)
    # quick LP pre-polish check of loaded points is skipped to save time for hopping
    n, d = 93, 13
    # basin hopping
    HOP_END = t0 + 142.0
    trial = 0
    # adapt adam length to save time: shorter early, longer when promising
    while time.time() < HOP_END and time.time() - t0 < TIME_BUDGET - 30:
        mode = trial % 5
        if mode == 0:
            sigma = 0.008
        elif mode == 1:
            sigma = 0.020
        elif mode == 2:
            sigma = 0.045
        elif mode == 3:
            sigma = 0.090
        else:
            sigma = 0.150
        x_try = best_x + sigma * rng.standard_normal((n, d))
        x_try = _normalize(x_try)
        # worst-point re-insertion for larger kicks
        if mode >= 2:
            krep = 1 if mode in (2, 3) else 2
            for _ in range(krep):
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                iw = int(np.argmax(per))
                # best-of-many random candidate far from others
                cand = _normalize(rng.standard_normal((512, d)))
                others = np.delete(x_try, iw, axis=0)
                dots = cand @ others.T
                mdots = dots.max(axis=1)
                ib = int(np.argmin(mdots))
                x_try[iw] = cand[ib]
        # local search
        if mode <= 1:
            x_opt = _local_opt(x_try, steps_adam=900, steps_hard=200)
        elif mode <= 3:
            x_opt = _local_opt(x_try, steps_adam=1300, steps_hard=280)
        else:
            x_opt = _local_opt(x_try, steps_adam=1600, steps_hard=320)
        v = _max_cos(x_opt)
        if v < best_val - 1e-12 and np.all(np.isfinite(x_opt)):
            best_x = x_opt
            best_val = v
        trial += 1
        # safety: if time is running out break
        if trial > 2000:
            break
    # final polish with remaining time
    remain = TIME_BUDGET - (time.time() - t0) - 1.0
    if remain > 5:
        lp_time = min(remain, 28.0)
        try:
            x_pol = _lp_polish(best_x, lp_time)
            vp = _max_cos(x_pol)
            if vp < best_val and np.all(np.isfinite(x_pol)):
                best_x = x_pol
                best_val = vp
        except Exception:
            pass
    else:
        # tiny greedy fallback polish
        pass
    best_x = _normalize(best_x)
    # final safety: remove any NaN / zero
    best_x = np.nan_to_num(best_x)
    best_x = _normalize(best_x)
    return best_x

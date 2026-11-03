import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _local_opt(x, steps_adam=850, steps_hard=200, lr0=0.004, tau0=0.015, tau1=0.0022):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps_adam):
        frac = s / max(1, steps_adam - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 * (0.3 + 0.7 * (1.0 - frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(np.max(G))
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        clip = 0.012
        too = sn > clip
        if np.any(too):
            step = np.where(too, step / (sn / clip + 1e-12), step)
        x = x - step
        x = _normalize(x)
    # hard max descent on active set
    step_size = 0.003
    cur = _max_cos(x)
    for _ in range(steps_hard):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(np.max(G))
        margin = 0.0012
        mask = G > (cur_max - margin)
        grad = mask.astype(np.float64) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-12)
        if not np.any(nz):
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
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


def _lp_polish(x, seconds, r0=5e-4):
    x = _normalize(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 4000:
            idx = np.argpartition(-c_all, 4000)[:4000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 4000
        I = iu[0][sel]
        J = iu[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.4:
            break
    return x


def solve():
    t0 = time.time()
    DEADLINE = 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 93, 13

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _normalize(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _normalize(rng.standard_normal((n, d)))
        best_val = _max_cos(best)

    def hole_reinsert(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        k = int(np.argmax(rowmax))
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        Xr = X[mask]
        M = 20000
        C = _normalize(rng.standard_normal((M, d)))
        D = C @ Xr.T
        worst = D.max(axis=1)
        idx = int(np.argmin(worst))
        Xn = X.copy()
        Xn[k] = C[idx]
        Xn += 0.0012 * rng.standard_normal(Xn.shape)
        return _normalize(Xn)

    # initial squeeze of loaded best
    try:
        if time.time() - t0 < DEADLINE - 10:
            b2 = _lp_polish(best, seconds=4.0, r0=8e-4)
            v2 = _max_cos(b2)
            if v2 < best_val:
                best, best_val = b2, v2
    except Exception:
        pass

    sigmas = [0.008, 0.02, 0.045, 0.09, 0.15, 0.012]
    trial = 0
    HOP_END = t0 + 138.0
    while time.time() < HOP_END and time.time() - t0 < DEADLINE - 30:
        if time.time() - t0 > DEADLINE - 30:
            break
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        try:
            if mode == 0:
                Xt = _normalize(best + sigma * rng.standard_normal(best.shape))
            elif mode == 1:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                worst = np.argsort(rm)[-12:]
                Xt = best.copy() + 0.002 * rng.standard_normal(best.shape)
                Xt[worst] += sigma * rng.standard_normal((len(worst), d))
                Xt = _normalize(Xt)
            elif mode == 2:
                Xt = hole_reinsert(best)
                # extra kick magnitude scaled by sigma
                Xt = _normalize(Xt + (sigma * 0.3) * rng.standard_normal(Xt.shape))
            elif mode == 3:
                Xt = best.copy()
                ksel = rng.choice(n, size=15, replace=False)
                Xt[ksel] += sigma * rng.standard_normal((15, d))
                Xt = _normalize(Xt)
            else:
                Xt = _normalize(best + sigma * rng.standard_normal(best.shape))
        except Exception:
            trial += 1
            continue
        # anneal to nearest basin
        try:
            Xt = _local_opt(Xt, steps_adam=750, steps_hard=160, lr0=0.0045, tau0=0.014, tau1=0.0020)
        except Exception:
            trial += 1
            continue
        if time.time() - t0 > DEADLINE - 25:
            v = _max_cos(Xt)
            if v < best_val:
                best, best_val = Xt, v
            trial += 1
            break
        try:
            v = _max_cos(Xt)
        except Exception:
            trial += 1
            continue
        if v < best_val - 1e-12:
            # tie contacts with LP
            try:
                Xt2 = _lp_polish(Xt, seconds=2.2, r0=4e-4)
                v2 = _max_cos(Xt2)
                if v2 < v:
                    Xt, v = Xt2, v2
            except Exception:
                pass
            if v < best_val:
                best, best_val = Xt, v
        trial += 1

    # final intensive polish on global best
    try:
        remain = DEADLINE - (time.time() - t0) - 1.5
        if remain > 3:
            b3 = _lp_polish(best, seconds=min(remain, 18.0), r0=2e-4)
            v3 = _max_cos(b3)
            if v3 < best_val:
                best, best_val = b3, v3
    except Exception:
        pass

    best = _normalize(best)
    return best

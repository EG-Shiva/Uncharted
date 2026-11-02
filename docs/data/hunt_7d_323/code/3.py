import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, time_budget, radius_init=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    while time.time() - start < time_budget and radius > 1e-13:
        # time left for this polish
        if time.time() - start > time_budget - 0.05:
            break
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 2e-9)
        # select tight pairs, cap to tightest 5000 for speed
        if thresh <= c.min():
            sel_idx = np.argpartition(c, -5000)[-5000:]
            I = iu0[sel_idx]
            J = iu1[sel_idx]
            cb = c[sel_idx]
        else:
            sel = c > thresh
            cnt = int(sel.sum())
            if cnt > 5000:
                sel_idx = np.argpartition(c, -5000)[-5000:]
                I = iu0[sel_idx]
                J = iu1[sel_idx]
                cb = c[sel_idx]
            elif cnt == 0:
                break
            else:
                I = iu0[sel]
                J = iu1[sel]
                cb = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    TOTAL = 172.0
    rng = np.random.default_rng(0)
    # load starting points
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        z = rng.standard_normal((323, 7))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = best_x.shape

    def contacts_info(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - 1e-9
        # pairs at max
        tight = G >= thr
        cnt = tight.sum(axis=1)  # symmetric, diag not counted
        order = np.argsort(-cnt)
        ratt = np.where(cnt == 0)[0]
        return cnt, order, ratt, m

    cnt, order, rattlers, _ = contacts_info(best_x)
    sigmas = [0.008, 0.015, 0.03, 0.06, 0.012, 0.025, 0.05]
    it = 0
    # exploration until reserve for final polish
    while True:
        now = time.time() - t0
        remain_total = TOTAL - now
        # reserve 28s final
        if remain_total < 30.0:
            break
        if now > 144.0:
            break
        # budget per trial
        budget = min(4.0, remain_total - 28.0)
        if budget < 0.8:
            break
        mode = it % 4
        sigma = sigmas[it % len(sigmas)]
        x_try = best_x.copy()
        if mode == 0:
            # global shake boosted on hubs
            noise = rng.standard_normal((n, d)) * sigma
            k = min(12, n)
            hub = order[:k]
            noise[hub] *= 3.0
            if len(rattlers) > 0:
                noise[rattlers] *= 1.5
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 1:
            # worst-point void re-insertion + light shake
            k = min(12, n)
            worst = int(rng.choice(order[:k]))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            M = best_x[mask]  # (n-1,d)
            Nc = 4000
            C = rng.standard_normal((Nc, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ M.T
            smax = S.max(axis=1)
            j = int(np.argmin(smax))
            x_try[worst] = C[j]
            # light shake others
            x_try[mask] = x_try[mask] + rng.standard_normal((n - 1, d)) * 0.005
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 2:
            # local cluster shake
            center = int(rng.integers(n))
            dots = best_x @ best_x[center]
            neigh = np.where(dots > 0.55)[0]
            if len(neigh) < 5:
                neigh = np.argsort(-dots)[:10]
            noise = rng.standard_normal((n, d)) * 0.003
            noise[neigh] += rng.standard_normal((len(neigh), d)) * 0.045
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # pure global moderate shake
            x_try = x_try + rng.standard_normal((n, d)) * sigma
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # polish trial
        try:
            xp, vp = _lp_polish(x_try, time_budget=budget)
        except Exception:
            it += 1
            continue
        if vp < best_val - 5e-13:
            best_x, best_val = xp, vp
            cnt, order, rattlers, _ = contacts_info(best_x)
        it += 1
        # safety: if time almost out break
        if time.time() - t0 > TOTAL - 29.0:
            break

    # final high-precision polish with remaining time
    remain = TOTAL - (time.time() - t0)
    # keep at least a little, at most ~28s
    final_budget = float(np.clip(remain - 1.0, 1.0, 28.0))
    if final_budget > 0.5:
        try:
            # two stages: coarse then fine
            if final_budget > 8:
                best_x, best_val2 = _lp_polish(best_x, time_budget=final_budget - 5.0, radius_init=2e-3)
                best_val = min(best_val, best_val2)
                r2 = TOTAL - (time.time() - t0) - 0.5
                if r2 > 0.5:
                    best_x, best_val2 = _lp_polish(best_x, time_budget=r2, radius_init=5e-4)
                    best_val = min(best_val, best_val2)
            else:
                best_x, best_val2 = _lp_polish(best_x, time_budget=final_budget, radius_init=5e-4)
                best_val = min(best_val, best_val2)
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

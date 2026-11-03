import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.3 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(c_all, -2800)[-2800]
            sel = c_all >= kth
            # subsample if still too many
            cnt2 = int(sel.sum())
            if cnt2 > 3200:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3200]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_power(x, iters, p0, p1, lr):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p0 + (p1 - p0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        # positive part only
        # clip negatives to 0
        np.maximum(G, 0.0, out=G)
        m = float(G.max())
        if m <= 1e-12:
            break
        # W = (G/m)^(p-1)
        W = G / m
        # power; avoid overflow: values in [0,1]
        # use exp((p-1)*log) for large p stability
        # direct power is fine
        np.power(W, p - 1.0, out=W)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        mn = float(np.mean(np.linalg.norm(grad_t, axis=1))) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / mn
        x = x - step * grad_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _deepest_void(Xkeep, rng, n_cand=2000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    vals = dots.max(axis=1)
    bi = int(np.argmin(vals))
    return cand[bi].copy(), float(vals[bi])

def solve():
    t0 = time.time()
    BUDGET = 170.0
    LP_BUDGET = 25.0
    rng = np.random.default_rng(0)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    xbest = cands[bi].copy()
    bbest = float(vals[bi])
    xcur = xbest.copy()
    bcur = bbest

    # quick contact info for surgery targets - recomputed periodically
    it = 0
    # annealing accept threshold
    while True:
        rem = BUDGET - (time.time() - t0) - LP_BUDGET
        if rem < 2.0:
            break
        it += 1
        # choose shake scale cyclically to vary basin distance
        cyc = it % 10
        if cyc in (0, 1):
            sig = 0.02
            s_iters, p0, p1, lr = 220, 18.0, 70.0, 0.025
        elif cyc in (2, 3):
            sig = 0.06
            s_iters, p0, p1, lr = 260, 15.0, 60.0, 0.03
        elif cyc in (4, 5):
            sig = 0.12
            s_iters, p0, p1, lr = 300, 12.0, 55.0, 0.03
        elif cyc == 6:
            sig = 0.20
            s_iters, p0, p1, lr = 320, 10.0, 50.0, 0.035
        else:
            sig = 0.035
            s_iters, p0, p1, lr = 240, 15.0, 65.0, 0.028

        y = xcur + rng.standard_normal((n, d)) * sig
        y /= np.linalg.norm(y, axis=1, keepdims=True)

        # every ~8th hop: overloaded-to-void surgery before smoothing
        if it % 8 == 0:
            try:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                # contacts within 1e-7 of max
                C = (G >= m - 1e-7)
                deg = C.sum(axis=1)
                # pick worst 2-3 hubs
                k = 2 + (it % 2)
                hubs = np.argsort(-deg)[:k]
                # build keep set excluding hubs
                mask = np.ones(n, dtype=bool); mask[hubs] = False
                Xkeep = y[mask]
                for h in hubs:
                    v, _ = _deepest_void(Xkeep, rng, n_cand=1500)
                    y[h] = v
                    # grow keep set
                    Xkeep = np.vstack([Xkeep, v[None, :]])
            except Exception:
                pass
        # every ~11th hop: rattler-style targeted move on current best's structure
        elif it % 11 == 0:
            try:
                G = xbest @ xbest.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                C = (G >= m - 1e-9)
                deg = C.sum(axis=1)
                # copy best and teleport its worst hub to void of rest
                y = xbest.copy()
                y = y + rng.standard_normal((n, d)) * 0.01
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                hubs = np.argsort(-deg)[:2]
                mask = np.ones(n, dtype=bool); mask[hubs] = False
                Xkeep = y[mask]
                for h in hubs:
                    v, _ = _deepest_void(Xkeep, rng, n_cand=1500)
                    y[h] = v
                    Xkeep = np.vstack([Xkeep, v[None, :]])
                s_iters, p0, p1, lr = 300, 14.0, 60.0, 0.03
            except Exception:
                pass

        # cheap collective smooth (no LP)
        try:
            y = _smooth_power(y, iters=s_iters, p0=p0, p1=p1, lr=lr)
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        # acceptance: greedy + small uphill allowance decaying
        allow = 0.0015 * max(0.0, rem / (BUDGET - LP_BUDGET))
        if v < bcur + 1e-12:
            xcur = y; bcur = v
            if v < bbest - 1e-12:
                xbest = y.copy(); bbest = v
        elif v < bcur + allow and (it % 3 == 0):
            # sideways move to different basin
            xcur = y; bcur = v
        # restart from best if wandering too high
        if bcur > bbest + 0.004:
            xcur = xbest.copy(); bcur = bbest
        # time check for loop overhead
        if time.time() - t0 + LP_BUDGET + 1.0 > BUDGET:
            break

    # final squeeze with trust-region LP
    try:
        rem_lp = BUDGET - (time.time() - t0)
        rem_lp = min(max(rem_lp - 1.0, 3.0), LP_BUDGET + 8.0)
        xbest = _lp_polish(xbest, budget=rem_lp)
        # also polish current if different and better chance
    except Exception:
        pass
    xbest = xbest / np.linalg.norm(xbest, axis=1, keepdims=True)
    return xbest

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _anneal(x_init, steps, k0, k1, lr0, lr1, delta, mom=0.7):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        m = float(G[iu].max())
        # window
        Mask = G > (m - delta)
        np.fill_diagonal(Mask, False)
        E = np.zeros_like(G)
        sel = Mask
        if np.any(sel):
            # clip exponent for stability
            arg = k * (G[sel] - m)
            arg = np.clip(arg, -20.0, 0.0)
            E[sel] = np.exp(arg)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        Gt /= deg
        vel = mom * vel + Gt
        x = x - lr * vel
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _farthest_replace(x, idx, rng, trials=2000):
    n, d = x.shape
    keep = np.ones(n, dtype=bool); keep[idx] = False
    Xk = x[keep]
    cand = rng.randn(trials, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # greedy farthest insertion for idx in order
    chosen = []
    cur = list(Xk)
    import numpy as _np
    C = cand.copy()
    for _ in range(len(idx)):
        if len(cur) > 0:
            M = _np.asarray(cur)
            S = C @ M.T
            worst = S.max(axis=1)
            o = _np.argmin(worst)
        else:
            o = 0
        chosen.append(C[o])
        cur.append(C[o])
        # remove chosen to avoid duplicates
        C = _np.delete(C, o, axis=0)
        if len(C) == 0:
            break
    x2 = x.copy()
    for ii, v in zip(idx, chosen):
        x2[ii] = v
    return x2


def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(12345)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)
    it = 0
    noimp = 0
    # reserve final polish
    while time.time() - t0 < BUDGET - 18.0:
        it += 1
        if time.time() - t0 > BUDGET - 18.0:
            break
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        # tight contacts (1e-9 window as in picture)
        cnt_tight = np.sum(Gb > best - 1e-9, axis=1)
        # loose window for ranking variety
        cnt_wide = np.sum(Gb > best - 0.02, axis=1)
        order_lo = np.argsort(cnt_tight + 0.1 * np.argsort(cnt_wide))
        order_hi = np.argsort(-cnt_tight)
        mode = it % 7
        # adapt strength if stuck
        boost = 1.0 + min(noimp / 15.0, 2.0)
        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * (0.008 * boost)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 350, 150, 2000, 0.02, 0.002, 0.06)
        elif mode == 1:
            x += rng.randn(n, d) * (0.025 * boost)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 450, 100, 2500, 0.025, 0.002, 0.08)
        elif mode == 2:
            ksel = 5 if (it % 2 == 0) else 3
            idx = order_lo[:ksel].copy()
            x += rng.randn(n, d) * 0.004
            x[idx] += rng.randn(ksel, d) * (0.10 * boost)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 500, 80, 1800, 0.03, 0.002, 0.10)
        elif mode == 3:
            ksel = 4
            idx = order_lo[:ksel].copy()
            # kill and farthest-point refill
            x = _farthest_replace(x_best, idx, rng, trials=3000)
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 600, 60, 1500, 0.03, 0.002, 0.12)
        elif mode == 4:
            ksel = 3
            idx = order_hi[:ksel].copy()
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * (0.09 * boost)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 450, 120, 2200, 0.025, 0.002, 0.08)
        elif mode == 5:
            # break worst pairs
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            # top pairs
            flat = G[iu]
            top = np.argsort(flat)[::-1][:6]
            victims = set()
            for t in top:
                i, j = iu[0][t], iu[1][t]
                victims.add(i if rng.rand() < 0.5 else j)
                if len(victims) >= 4:
                    break
            idx = np.array(sorted(victims), dtype=int)
            x[idx] += rng.randn(len(idx), d) * (0.12 * boost)
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 500, 90, 2000, 0.028, 0.002, 0.10)
        else:
            # large global shake + refill one point
            x += rng.randn(n, d) * (0.05 * boost)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 400, 150, 2500, 0.02, 0.003, 0.07)
        v = _max_cos(x)
        # decide to LP-polish: if promising or periodic
        thresh = best + (0.008 if mode in (1, 3, 5) else 0.003)
        do_polish = (v < thresh) or (it % 9 == 0 and v < best + 0.015)
        if do_polish:
            rem = BUDGET - 18.0 - (time.time() - t0)
            if rem < 1.0:
                # no time for LP, just check annealed
                if v < best - 1e-12:
                    x_best, best = x / np.linalg.norm(x, axis=1, keepdims=True), v
                    noimp = 0
                else:
                    noimp += 1
                continue
            sec = min(2.2, rem)
            # shorter for bad candidates
            if v > best + 0.003:
                sec = min(1.0, rem)
            y = _lp_polish(x, sec, t0, BUDGET - 18.0)
            vy = _max_cos(y)
            if vy < best - 1e-12:
                x_best, best = y, vy
                noimp = 0
            else:
                # also consider raw annealed if it alone beat best
                if v < best - 1e-12:
                    x_best, best = x, v
                    noimp = 0
                else:
                    noimp += 1
        else:
            if v < best - 1e-12:
                # quick win without LP
                rem = BUDGET - 18.0 - (time.time() - t0)
                sec = min(2.0, rem) if rem > 1.0 else 0.0
                if sec > 0.5:
                    y = _lp_polish(x, sec, t0, BUDGET - 18.0)
                    vy = _max_cos(y)
                    if vy < best:
                        x_best, best = y, vy
                        noimp = 0
                    else:
                        noimp += 1
                else:
                    x_best, best = x, v
                    noimp = 0
            else:
                noimp += 1
    # final polish of best
    rem = max(1.0, 178.0 - (time.time() - t0))
    rem = min(rem, 25.0)
    try:
        x_best = _lp_polish(x_best, rem)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

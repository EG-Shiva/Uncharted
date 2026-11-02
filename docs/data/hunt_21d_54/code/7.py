import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def proj_unit(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    x = np.nan_to_num(x)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def lp_polish(x, seconds, r0=3e-3):
    x = proj_unit(x)
    n, d = x.shape
    if seconds <= 0.2:
        return x
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain < 0.2:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csv = c[sel]
        if p > 2800:
            k = 2800
            idx = np.argpartition(-csv, k)[:k]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(4.0, max(0.5, remain - 0.1))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = proj_unit(y)
        v = max_cos(y)
        if v < best - 1e-13:
            x, best = y, v
            radius = min(radius * 1.5, 0.04)
        else:
            radius *= 0.5
    return x

def smooth_polish(y, steps, lr0=0.04, T0=0.025, T1=0.003):
    y = y.copy()
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (1.0 - 0.5 * f)
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= gn
        y = y - lr * grad
        y = proj_unit(y)
    return y

def optimize_hole(Y, v, steps=350, lr0=0.05, T0=0.02, T1=0.002):
    v = v / (np.linalg.norm(v) + 1e-12)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (1.0 - 0.5 * f)
        g = Y @ v
        m = float(g.max())
        w = np.exp((g - m) / T)
        w /= (w.sum() + 1e-300)
        grad = w @ Y
        grad = grad - float(grad @ v) * v
        gn = float(np.linalg.norm(grad)) + 1e-12
        v = v - lr * grad / gn
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def kick_points(y, idx, rng, strength):
    n, d = y.shape
    for i in idx:
        v = rng.normal(size=d)
        v -= float(v @ y[i]) * y[i]
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            continue
        v /= nv
        ang = float(strength * (0.5 + rng.random()))
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    return y

def solve():
    T0 = time.time()
    LIM = 176.0
    rng = np.random.default_rng(0)
    try:
        b1 = np.load("best.npy")
    except Exception:
        b1 = None
    try:
        b2 = np.load("record.npy")
    except Exception:
        b2 = None
    cands = []
    if b1 is not None:
        cands.append(proj_unit(np.asarray(b1, dtype=float)))
    if b2 is not None:
        cands.append(proj_unit(np.asarray(b2, dtype=float)))
    if len(cands) == 0:
        cands.append(proj_unit(rng.normal(size=(54, 21))))
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = best.shape

    it = 0
    RESERVE = 14.0
    while time.time() - T0 < LIM - RESERVE:
        it += 1
        remain = LIM - RESERVE - (time.time() - T0)
        if remain < 2.0:
            break
        mod = it % 5
        y = best.copy()
        do_lp = False
        lp_time = 4.0
        thresh = 1e-3
        if mod == 1:
            k = int(rng.integers(3, 6))
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.normal(size=(k, d))
            rnd = proj_unit(rnd)
            y[idx] = rnd
            y = smooth_polish(y, steps=700, lr0=0.05, T0=0.03, T1=0.003)
            v = max_cos(y)
            thresh = 0.015
            do_lp = v < best_val + thresh
            lp_time = min(6.0, remain)
        elif mod == 2:
            k = 9
            idx = rng.choice(n, size=k, replace=False)
            strength = float([0.25, 0.4, 0.55][(it // 5) % 3])
            y = kick_points(y, idx, rng, strength)
            nz = rng.normal(size=(n, d)) * (strength * 0.15)
            nz -= (np.sum(nz * y, axis=1, keepdims=True)) * y
            y = proj_unit(y + nz)
            y = smooth_polish(y, steps=380, lr0=0.035, T0=0.02, T1=0.0025)
            v = max_cos(y)
            thresh = 0.008
            do_lp = v < best_val + thresh
            lp_time = min(4.5, remain)
        elif mod == 3:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            wi = int(order[int(rng.integers(0, 3))])
            Y = np.delete(best, wi, axis=0)
            cand = proj_unit(rng.normal(size=(2000, d)))
            scores = (cand @ Y.T).max(axis=1)
            bidx = np.argsort(scores)[:4]
            pick = int(bidx[int(rng.integers(0, len(bidx)))])
            v0 = cand[pick]
            vopt = optimize_hole(Y, v0, steps=350, lr0=0.06, T0=0.02, T1=0.0015)
            y = best.copy()
            y[wi] = vopt
            y = smooth_polish(y, steps=300, lr0=0.025, T0=0.015, T1=0.002)
            v = max_cos(y)
            thresh = 0.006
            do_lp = v < best_val + thresh
            lp_time = min(4.5, remain)
        elif mod == 4:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = flat // n, flat % n
            y = kick_points(y, [ii, jj], rng, 0.6)
            # push them apart explicitly
            for q in (ii, jj):
                other = jj if q == ii else ii
                direction = y[q] - float(y[q] @ y[other]) * y[other]
                # actually move q away from other: subtract component along other
                away = -y[other] + float(y[q] @ y[other]) * y[q]
                away -= float(away @ y[q]) * y[q]
                an = float(np.linalg.norm(away))
                if an > 1e-12:
                    away /= an
                    y[q] = y[q] - 0.08 * away
            y = proj_unit(y)
            y = smooth_polish(y, steps=350, lr0=0.03, T0=0.018, T1=0.002)
            v = max_cos(y)
            thresh = 0.007
            do_lp = v < best_val + thresh
            lp_time = min(4.5, remain)
        else:
            strength = 0.12
            nz = rng.normal(size=(n, d)) * strength
            nz -= (np.sum(nz * y, axis=1, keepdims=True)) * y
            y = proj_unit(y + nz)
            y = smooth_polish(y, steps=450, lr0=0.04, T0=0.022, T1=0.002)
            v = max_cos(y)
            thresh = 0.008
            do_lp = v < best_val + thresh
            lp_time = min(5.0, remain)
        if do_lp:
            lp_time = min(lp_time, LIM - RESERVE - (time.time() - T0))
            if lp_time < 0.8:
                continue
            y2 = lp_polish(y, seconds=lp_time)
            v2 = max_cos(y2)
            if v2 < best_val - 1e-12:
                best = y2
                best_val = v2
    # final polish of incumbent
    left = LIM - (time.time() - T0) - 0.5
    if left > 1.0:
        best = lp_polish(best, seconds=min(left, 12.0), r0=2e-3)
    best = proj_unit(best)
    return best

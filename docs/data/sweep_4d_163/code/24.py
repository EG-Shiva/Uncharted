import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu = iu0
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > (m - window)
        p = int(np.count_nonzero(sel))
        if p == 0:
            break
        if p > 2200:
            idx = np.argpartition(c_all, -2200)[-2200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = 2200
        I = iu[0][sel]
        J = iu[1][sel]
        c_sel = c_all[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 1.5})
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
    return x, best

def _smooth_refine(x, steps, k0, k1, lr0, lr1, mu, noise0, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        v = mu * v - lr * gt
        if noise0 > 0:
            temp = noise0 * (1.0 - frac)
            z = rng.randn(n, d)
            z -= np.sum(z * x, axis=1, keepdims=True) * x
            v += temp * z
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_void(x_keep, rng, n_samples=4000):
    cand = rng.randn(n_samples, 4)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    v = cand[bi].copy()
    # local void maximization: push away from neighbours
    for _ in range(40):
        d = x_keep @ v
        dm = float(d.max())
        w = np.exp(60.0 * (d - dm))
        g = w @ x_keep
        gt = g - float(g @ v) * v
        nr = float(np.linalg.norm(gt))
        if nr < 1e-12:
            break
        gt /= nr
        v = v - 0.06 * gt
        v /= np.linalg.norm(v)
    return v

def _micro_hill(x, best, seconds, t_end, rng):
    # incremental single-point hill climbing, very fast evaluation
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    # active set: points touching near-max
    def active_set(thr):
        rm = G.max(axis=1)
        return np.where(rm > best - thr)[0]
    start = time.time()
    # sigma schedule
    sigmas = [8e-5, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
    for sigma in sigmas:
        if time.time() > t_end or time.time() - start > seconds:
            break
        act = active_set(2e-4)
        if len(act) == 0:
            act = np.arange(n)
        trials = 4000 if sigma > 1e-6 else 6000
        stale = 0
        for t in range(trials):
            if time.time() > t_end or time.time() - start > seconds:
                break
            if rng.rand() < 0.85:
                i = int(act[rng.randint(len(act))])
            else:
                i = int(rng.randint(n))
            z = rng.randn(d)
            z -= float(z @ x[i]) * x[i]
            nz = float(np.linalg.norm(z))
            if nz < 1e-12:
                continue
            z /= nz
            # random magnitude around sigma
            step = sigma * (0.5 + rng.rand())
            xn = x[i] + step * z
            xn /= np.linalg.norm(xn)
            row = x @ xn
            row[i] = -2.0
            # max excluding row/col i
            # G has -2 on diag, so max of rest:
            # compute mask max quickly: max of G without i
            # we can keep current best of rest via precompute? just compute max of G with row/col i masked
            # cheap: Gmax_rest = max of G where indices != i. approximate by global second check
            # exact: compute max over all entries except row i col i
            # do: m1 = row.max(), m2 = G max excluding i (compute via max of G with rows/cols masked is O(n^2) -> too slow)
            # instead maintain: since only row i changes, new global max = max(m1, rest_max) where rest_max = max of G submatrix without i.
            # compute rest_max on fly every ~50 steps only? approximate: rest_max <= best, and if m1 < best and best involved i, need rest.
            # simplest: compute m1, if m1 >= best: reject (no need rest). else compute rest_max via masked max (O(n^2) but only on promising moves, rare).
            m1 = float(row.max())
            if m1 >= best:
                stale += 1
                continue
            # promising: compute rest max
            # mask row/col i
            # G copy masked: use max of G with row i and col i set to -2
            # we can compute quickly: rest = G.max() but if argmax involves i it is overestimate? Actually G still old row i, which may be the max. So need masked.
            # do O(n^2) max with masking via numpy (fast, 26k entries)
            # only executed when m1 < best, which is rare
            Gm = G.copy()
            Gm[i, :] = -2.0
            Gm[:, i] = -2.0
            rest = float(Gm.max())
            newmax = m1 if m1 > rest else rest
            if newmax < best:
                best = float(newmax)
                x[i] = xn
                # update G row/col i
                full = x @ xn  # x now includes xn at i, so entry i is 1.0
                G[i, :] = full
                G[:, i] = full
                G[i, i] = -2.0
                # refresh active set occasionally
                if t % 500 == 499:
                    act = active_set(2e-4)
                    if len(act) == 0:
                        act = np.arange(n)
                stale = 0
            else:
                stale += 1
    return x, best

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng0 = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng0.randn(163, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (163, 4):
        tmp = rng0.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (163, 4) and _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    best_x = xb.copy()
    best_v = _max_cos(best_x)
    n, d = best_x.shape
    best_x, best_v = _lp_polish(best_x, 3.0, T_END - 8.0)
    trial = 0
    while time.time() < T_END - 12.0:
        trial += 1
        rng = np.random.RandomState(1000 + trial * 7919)
        mode = trial % 4
        x = best_x.copy()
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order_worst = np.argsort(-rowmax)
        order_best = np.argsort(rowmax)
        # degree with 1e-7 band
        deg = np.sum(G > (best_v - 1e-7), axis=1)
        order_deg = np.argsort(-deg)
        if mode == 0:
            sigma = 0.012 if (trial % 8 != 0) else 0.03
            x = x + sigma * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 450, 25.0, 280.0, 0.012, 0.002, 0.85, 0.0012
        elif mode == 1:
            kbad = 12
            bad = order_worst[:kbad]
            shake = 0.006 * rng.randn(n, d)
            shake[bad] += 0.085 * rng.randn(kbad, d)
            x = x + shake
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 550, 30.0, 320.0, 0.010, 0.0015, 0.88, 0.0009
        elif mode == 2:
            nrep = 1 if (trial % 2 == 0) else 2
            reps = list(order_worst[:nrep])
            if trial % 8 == 2:
                reps[-1] = int(order_best[0])
            keep = np.ones(n, dtype=bool)
            keep[reps] = False
            for r in reps:
                xk = x[keep]
                x[r] = _best_void(xk, rng, 3500)
                keep[r] = True
            x = x + 0.004 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 600, 20.0, 250.0, 0.012, 0.002, 0.85, 0.0007
        else:
            # jam relief: shake overloaded high-degree points + small global
            kbad = 10
            bad = order_deg[:kbad]
            shake = 0.005 * rng.randn(n, d)
            shake[bad] += 0.06 * rng.randn(kbad, d)
            # also move one rattler to void
            r = int(order_best[0])
            keep = np.ones(n, dtype=bool)
            keep[r] = False
            x[r] = _best_void(x[keep], rng, 2500)
            x = x + shake
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 550, 28.0, 300.0, 0.011, 0.0018, 0.86, 0.0009
        x = _smooth_refine(x, steps, k0, k1, lr0, lr1, mu, nz, rng)
        v_smooth = _max_cos(x)
        if v_smooth < best_v + 0.012 and time.time() < T_END - 10.0:
            lp_sec = 4.0 if v_smooth < best_v else 2.0
            lp_sec = min(lp_sec, T_END - 10.0 - time.time())
            if lp_sec > 0.4:
                x, v_smooth = _lp_polish(x, lp_sec, T_END - 8.0)
        if v_smooth < best_v:
            best_v = float(v_smooth)
            best_x = x.copy()
        if time.time() >= T_END - 10.0:
            break
    # final squeeze: LP + micro hill climbing
    if time.time() < T_END - 1.5:
        sec = max(0.5, min(5.0, T_END - 1.5 - time.time()))
        best_x, best_v = _lp_polish(best_x, sec, T_END - 1.0)
    if time.time() < T_END - 0.8:
        sec = max(0.5, T_END - 0.8 - time.time())
        rngf = np.random.RandomState(999)
        best_x, best_v = _micro_hill(best_x, best_v, min(sec, 6.0), T_END - 0.3, rngf)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

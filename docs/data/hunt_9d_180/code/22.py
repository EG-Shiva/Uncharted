import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t0 = time.time()
    if deadline is None:
        deadline = t0 + seconds
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csub = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps, T0, T1, lr0, mu=0.82):
    n, d = x.shape
    best = x.copy()
    bv = max_cos(x)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 1.0 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s & 63) == 63 or s == steps - 1:
            v = max_cos(x)
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def best_void(others, rng, n_try=8000, n_refine=300):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        frac = k / max(1, n_refine - 1)
        sig = 0.08 * (0.001 / 0.08) ** frac
        prop = y + sig * rng.randn(d)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12:
            continue
        prop /= nrm
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def solve():
    t0 = time.time()
    deadline = t0 + 176.0
    rng = np.random.RandomState(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.randn(180, 9)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        x_best = x0
        b_best = max_cos(x0)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    noimp = 0
    it = 0
    explore_end = t0 + 118.0
    while time.time() < explore_end and time.time() < deadline - 50:
        it += 1
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        if it % 7 == 0:
            x_try = x_cur.copy()
            # bias subset to bad points + random
            order = np.argsort(rowmax)
            nb = 6
            bad = order[-nb:]
            rnd = rng.choice(n, size=6, replace=False)
            idx = np.unique(np.concatenate([bad, rnd]))
            sigma = 0.018 if (it % 14 == 0) else 0.010
            x_try[idx] += sigma * rng.randn(len(idx), d)
            x_try[idx] /= np.linalg.norm(x_try[idx], axis=1, keepdims=True)
            steps = 350
            T0, T1, lr0 = 0.008, 0.0008, 0.030
        else:
            k = 2 if (it % 5 == 0) else 1
            # victims: worst rowmax; occasionally include a random rattler-like point
            victims = list(np.argsort(rowmax)[-k:])
            if it % 11 == 0:
                # replace one victim by lowest-contact point to move rattler
                victims[0] = int(np.argmin(rowmax))
            x_try = x_cur.copy()
            for v in victims:
                others = np.delete(x_try, v, axis=0)
                y, _ = best_void(others, rng, n_try=8000, n_refine=250)
                x_try[v] = y
            steps = 450 if k == 2 else 380
            T0, T1, lr0 = 0.010, 0.0009, 0.028
            if cur_val < b_best + 0.0005 and (it % 3 == 0):
                T0 = 0.005
        x_try_s, v_s = smooth_relax(x_try, steps, T0, T1, lr0)
        # wandering acceptance
        if v_s < b_best - 1e-12:
            b_best = v_s
            x_best = x_try_s.copy()
            x_cur = x_try_s.copy()
            cur_val = v_s
            noimp = 0
        else:
            noimp += 1
            # always wander with current (even if worse) but reset if too stale
            if v_s < cur_val or rng.rand() < 0.25:
                x_cur = x_try_s.copy()
                cur_val = v_s
            if noimp > 30:
                x_cur = x_best.copy()
                cur_val = b_best
                noimp = 0
        if time.time() > explore_end:
            break
    # final polish of best
    remain = deadline - time.time() - 2.0
    if remain > 8.0:
        budget = min(38.0, remain)
        try:
            x_pol = lp_polish(x_best, budget, deadline - 1.0)
            v_pol = max_cos(x_pol)
            if v_pol < b_best:
                x_best = x_pol
                b_best = v_pol
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

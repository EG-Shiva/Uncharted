import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_refine(x, steps, k0, k1, lr0, mu, rng, jitter=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    if jitter > 0:
        x = x + jitter * rng.standard_normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = np.max(G - np.eye(n) * 2.0)
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def _greedy_cap_init(frozen, k, rng, pool_n=3000):
    d = frozen.shape[1]
    pool = rng.standard_normal(size=(pool_n, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    # max dot to frozen
    fd = (pool @ frozen.T).max(axis=1)
    # keep farthest subset to reduce cost
    keep = min(pool_n, 600)
    order = np.argsort(fd)[:keep]
    cand = pool[order]
    cdf = fd[order]
    selected = []
    sel_fd = []
    # first pick: among 3 best random
    first_choice = rng.integers(min(3, len(cand)))
    selected.append(cand[first_choice])
    # mask to avoid re-pick
    used = np.zeros(len(cand), dtype=bool)
    used[first_choice] = True
    for _ in range(1, k):
        S = np.stack(selected)  # (s,d)
        d2 = cand @ S.T
        d2 = d2.max(axis=1)
        dmax = np.maximum(cdf, d2)
        dmax[used] = 1e9
        best5 = np.argsort(dmax)[:5]
        # filter unused
        best5 = [b for b in best5 if not used[b]]
        if len(best5) == 0:
            idx = int(np.argmin(dmax))
        else:
            idx = int(best5[rng.integers(len(best5))])
        selected.append(cand[idx])
        used[idx] = True
    arr = np.stack(selected)
    # small random kick to break symmetry
    arr = arr + 0.01 * rng.standard_normal(size=arr.shape)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr

def _single_reinsert(others, rng, n_cand=4000, steps=150):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(scores)].copy()
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        kk = 40.0
        w = np.exp(kk * (dots - m))
        w /= w.sum()
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.15 * (1.0 - 0.5 * it / steps)
        c = c - step * gt
        nc = np.linalg.norm(c)
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    try:
        x_rec = np.load("record.npy").astype(float)
        x_rec /= np.linalg.norm(x_rec, axis=1, keepdims=True)
        xb_n = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        if _max_cos(x_rec) < _max_cos(xb_n):
            x_best = x_rec.copy()
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val
    n, d = cur_x.shape

    def point_stats(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        worst = G.max(axis=1)
        deg = (G > mx - 5e-4).sum(axis=1)
        return worst, deg, mx

    HOP_END = T0 + 118.0
    it = 0
    while time.time() - T0 < HOP_END and time.time() - T0 < BUDGET - 50:
        it += 1
        worst, deg, mx = point_stats(cur_x)
        y = cur_x.copy()
        mode = it % 7
        if mode in (0, 1, 5):
            # CAP RUIN: remove spatially coherent cluster, greedy refill
            if mode == 0:
                k = int(rng.integers(12, 19))
            elif mode == 1:
                k = int(rng.integers(6, 11))
            else:
                k = int(rng.integers(8, 14))
            # pick cap centre: either nearest neighbours of random point or directional cap
            # bias centre to loose or overloaded regions occasionally
            r = rng.random()
            if r < 0.35:
                # loose centre (few contacts)
                order = np.argsort(deg)
                i0 = int(order[rng.integers(0, 10)])
                dots = y @ y[i0]
                idx = np.argsort(-dots)[:k]
            elif r < 0.6:
                # overloaded centre
                order = np.argsort(-deg)
                i0 = int(order[rng.integers(0, 10)])
                dots = y @ y[i0]
                idx = np.argsort(-dots)[:k]
            else:
                u = rng.standard_normal(d); u /= np.linalg.norm(u)
                proj = y @ u
                idx = np.argsort(-proj)[:k]
            frozen = np.delete(y, idx, axis=0)
            new_arr = _greedy_cap_init(frozen, k, rng, pool_n=3000)
            y[idx] = new_arr
            # annealed smooth refine global
            if k >= 12:
                y = _smooth_refine(y, steps=750, k0=20, k1=320, lr0=0.28, mu=0.88, rng=rng, jitter=0.003)
            else:
                y = _smooth_refine(y, steps=650, k0=25, k1=350, lr0=0.24, mu=0.87, rng=rng, jitter=0.002)
        elif mode == 2:
            # worst-pair teleport: find worst pair, teleport one endpoint to void
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            a, b = np.unravel_index(np.argmax(G), G.shape)
            # pick the one with larger worst-second? just pick b
            victim = int(b) if rng.random() < 0.5 else int(a)
            others = np.delete(y, victim, axis=0)
            c_new = _single_reinsert(others, rng, n_cand=5000, steps=180)
            y[victim] = c_new
            y = _smooth_refine(y, steps=600, k0=30, k1=380, lr0=0.20, mu=0.88, rng=rng, jitter=0.004)
        elif mode == 3:
            # targeted shake of loose + jammed + worst
            idx_loose = np.argsort(deg)[:6]
            idx_over = np.argsort(-deg)[:6]
            idx_w = np.argsort(-worst)[:4]
            idx = np.unique(np.concatenate([idx_loose, idx_over, idx_w]))
            y[idx] += 0.22 * rng.standard_normal(size=(len(idx), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=700, k0=28, k1=320, lr0=0.22, mu=0.88, rng=rng)
        elif mode == 4:
            # moderate global shake from best/cur
            sigma = 0.04 + 0.08 * rng.random()
            base = x_best if rng.random() < 0.6 else cur_x
            y = base + sigma * rng.standard_normal(size=base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=650, k0=25, k1=340, lr0=0.24, mu=0.87, rng=rng)
        else:
            # small exploitative shake of random subset
            kpick = 8
            idx = rng.choice(n, size=kpick, replace=False)
            y[idx] += 0.05 * rng.standard_normal(size=(kpick, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=500, k0=40, k1=420, lr0=0.15, mu=0.90, rng=rng)

        new_val = _max_cos(y)
        if new_val < best_val - 1e-15:
            x_best = y.copy()
            best_val = new_val
            cur_x = y.copy()
            cur_val = new_val
        else:
            elapsed = time.time() - T0
            frac = elapsed / max(1e-9, (HOP_END - T0))
            T = 0.002 * max(0.08, 1.0 - frac)
            T = max(T, 0.00015)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy()
                cur_val = new_val
            # occasional restart to best
            if rng.random() < 0.03:
                cur_x = x_best.copy() + 0.02 * rng.standard_normal(size=x_best.shape)
                cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
                cur_val = _max_cos(cur_x)

    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 5:
        try:
            x_best = _smooth_refine(x_best, steps=300, k0=80, k1=650, lr0=0.07, mu=0.90, rng=rng)
            nv = _max_cos(x_best)
            if nv < best_val:
                best_val = nv
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=remain, t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol
                best_val = pv
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0):
            x_best = x0 if _max_cos(x0) < _max_cos(x_best) else x_best
    except Exception:
        pass
    return x_best

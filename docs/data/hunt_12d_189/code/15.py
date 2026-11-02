import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_opt(x, steps=800, a0=25.0, a1=300.0, eta0=0.05, deadline=None):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        if deadline is not None and s % 80 == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        # geometric anneal
        alpha = a0 * ((a1 / max(a0, 1e-9)) ** frac)
        eta = eta0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W @ x
        wg = (W * G).sum(axis=1, keepdims=True)
        Tang = S - wg * x
        wsum = W.sum(axis=1, keepdims=True) + 1e-12
        step = Tang / wsum
        x = x - eta * step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x

def hole_for_index(y, idx, rng, K=8000):
    d = y.shape[1]
    mask = np.arange(y.shape[0]) != idx
    Xr = y[mask]
    # chunked to save memory/time
    best_c = None
    best_v = 1e9
    best_vec = None
    # two chunks
    for _ in range(2):
        k = K // 2
        cand = rng.normal(size=(k, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = Xr @ cand.T
        cmax = dots.max(axis=0)
        j = int(np.argmin(cmax))
        if cmax[j] < best_v:
            best_v = float(cmax[j])
            best_vec = cand[j].copy()
    return best_vec

def lp_polish(x, deadline, max_pairs=700):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    # precompute iu arrays
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        Gfull = x @ x.T
        c_full = Gfull[iu0]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel_idx = np.where(c_full > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        # cap to closest pairs to keep HiGHS fast
        if sel_idx.size > max_pairs:
            # take largest c values
            part = np.argpartition(c_full[sel_idx], -max_pairs)[-max_pairs:]
            sel_idx = sel_idx[part]
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        p = len(I)
        b_ub = -c_full[sel_idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        c1 = (I[:, None] * d + np.arange(d)).ravel()
        c2 = (J[:, None] * d + np.arange(d)).ravel()
        # interleave? build cols
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cols_I, cols_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
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
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_RESERVE = 20.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (189, 12) and np.all(np.isfinite(c)):
            nrm = np.linalg.norm(c, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            c = c / nrm
            cands.append(c)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        c = rng0.normal(size=(189, 12))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = cands[0]
    best_v = max_cos(best_x)
    for c in cands[1:]:
        v = max_cos(c)
        if v < best_v:
            best_x, best_v = c, v
    best_x = best_x.copy()
    rng = np.random.default_rng(7)
    it = 0
    sigmas = [0.03, 0.06, 0.015, 0.09, 0.04]
    while True:
        now = time.time()
        if now > HARD_END - FINAL_RESERVE - 2.0:
            break
        if it > 120:
            break
        # analyse
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m_best = float(G.max())
        per_max = G.max(axis=1)
        deg = (G > m_best - 0.01).sum(axis=1)
        rattlers = np.where(per_max < m_best - 1e-3)[0]
        overload = np.argsort(deg)[-6:]
        worst = np.argsort(per_max)[-2:]
        y = best_x.copy()
        mode = it % 5
        sigma = sigmas[it % len(sigmas)]
        if mode == 0:
            # teleport rattlers to holes + small noise
            if len(rattlers) > 0:
                # limit to 4 per iter to save time
                sel = rattlers[rng.choice(len(rattlers), size=min(len(rattlers), 4), replace=False)]
                for wi in sel:
                    bv = hole_for_index(y, int(wi), rng, K=6000)
                    y[int(wi)] = bv
            y = y + rng.normal(scale=0.012, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = 300.0
            eta0 = 0.05
            steps = 800
        elif mode == 1:
            noise = rng.normal(scale=sigma, size=y.shape)
            if len(overload) > 0:
                noise[overload] *= 2.2
            if len(rattlers) > 0:
                noise[rattlers] *= 2.0
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = 260.0
            eta0 = 0.045
            steps = 800
        elif mode == 2:
            y = y + rng.normal(scale=0.025, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = 320.0
            eta0 = 0.06
            steps = 700
        elif mode == 3:
            for wi in worst:
                bv = hole_for_index(y, int(wi), rng, K=8000)
                y[int(wi)] = bv
            y = y + rng.normal(scale=0.01, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = 300.0
            eta0 = 0.04
            steps = 850
        else:
            # random subset shake
            k = 20
            idx = rng.choice(y.shape[0], size=k, replace=False)
            y[idx] = y[idx] + rng.normal(scale=0.12, size=(k, y.shape[1]))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = 250.0
            eta0 = 0.05
            steps = 750
        # time check before heavy work
        if time.time() > HARD_END - FINAL_RESERVE - 1.0:
            break
        iter_deadline = HARD_END - FINAL_RESERVE
        y = smooth_opt(y, steps=steps, a0=30.0, a1=a1, eta0=eta0, deadline=iter_deadline - 2.0)
        if time.time() > HARD_END - FINAL_RESERVE - 0.5:
            # evaluate without LP if no time
            v = max_cos(y)
            if v < best_v - 1e-12:
                best_x, best_v = y, v
            it += 1
            continue
        lp_dead = min(time.time() + 1.8, HARD_END - FINAL_RESERVE)
        y = lp_polish(y, deadline=lp_dead, max_pairs=700)
        v = max_cos(y)
        if v < best_v - 1e-12 and np.all(np.isfinite(y)):
            best_x, best_v = y, v
        it += 1
    # final polish
    remain = HARD_END - time.time()
    if remain > 3.0:
        fin_dead = time.time() + min(remain - 0.8, FINAL_RESERVE)
        best_x = lp_polish(best_x, deadline=fin_dead, max_pairs=900)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

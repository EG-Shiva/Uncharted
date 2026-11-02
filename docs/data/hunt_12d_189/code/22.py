import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_opt(x, steps, a0, a1, eta0, t_deadline=None):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        if t_deadline is not None and s % 100 == 0 and time.time() > t_deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        eta = eta0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W @ x
        wg = (W * G).sum(axis=1, keepdims=True)
        wsum = W.sum(axis=1, keepdims=True) + 1e-12
        Tang = S - wg * x
        step = Tang / wsum
        x = x - eta * step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x

def lp_polish_capped(x, seconds, max_pairs=1400):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 7e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_full = Gfull[iu0, iu1]
        m = c_full.max()
        thresh = m - max(20.0 * radius, 2e-9)
        # select tight pairs, cap to max_pairs tightest
        if np.count_nonzero(c_full > thresh) > max_pairs:
            part = np.argpartition(c_full, -max_pairs)[-max_pairs:]
            sel = np.zeros(c_full.shape, dtype=bool)
            sel[part] = True
        else:
            sel = c_full > thresh
        I = iu0[sel]; J = iu1[sel]
        c_sel = c_full[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def deepest_hole(Xr, rng, K):
    d = Xr.shape[1]
    cand = rng.normal(size=(K, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = Xr @ cand.T
    cmax = dots.max(axis=0)
    bi = int(np.argmin(cmax))
    return cand[bi], float(cmax[bi])

def solve():
    t0 = time.time()
    BUDGET = 168.0
    FINAL_RESERVE = 18.0
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
            cands.append(c / nrm)
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
            best_x, best_v = c.copy(), v
    best_x = best_x.copy()
    rng = np.random.default_rng(0)
    it = 0
    sigmas = [0.035, 0.06, 0.02, 0.09, 0.045]
    while True:
        now = time.time()
        remain_total = BUDGET - (now - t0)
        if remain_total < FINAL_RESERVE + 3:
            break
        if it > 100:
            break
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m_best = G.max()
        per_max = G.max(axis=1)
        deg = (G > m_best - 0.01).sum(axis=1)
        rattlers = np.where(per_max < m_best - 5e-4)[0]
        overload = np.argsort(deg)[-6:]
        worst2 = np.argsort(per_max)[-2:]
        y = best_x.copy()
        mode = it % 5
        if mode == 4:
            # hole re-insertion of 2 worst
            for wi in worst2:
                mask = np.arange(189) != wi
                Xr = y[mask]
                hv, _ = deepest_hole(Xr, rng, K=7000)
                y[wi] = hv
            y = y + rng.normal(scale=0.012, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, a0, a1, eta0 = 450, 40.0, 300.0, 0.04
        elif mode == 3:
            noise = rng.normal(scale=0.02, size=y.shape)
            if len(rattlers) > 0:
                noise[rattlers] += rng.normal(scale=0.12, size=(len(rattlers), 12))
            noise[overload] += rng.normal(scale=0.10, size=(len(overload), 12))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, a0, a1, eta0 = 500, 25.0, 250.0, 0.05
        else:
            sigma = sigmas[it % len(sigmas)]
            noise = rng.normal(scale=sigma, size=y.shape)
            if len(rattlers) > 0:
                noise[rattlers] *= 2.5
            noise[overload] *= 1.8
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1v = 220.0 + (it % 3) * 50.0
            steps, a0, a1, eta0 = 400, 30.0, a1v, 0.04
        deadline = t0 + BUDGET - FINAL_RESERVE
        y = smooth_opt(y, steps=steps, a0=a0, a1=a1, eta0=eta0, t_deadline=deadline)
        v_smooth = max_cos(y)
        # LP only if promising to save time
        do_lp = (v_smooth < best_v + 0.002)
        if do_lp:
            now2 = time.time()
            rem = BUDGET - FINAL_RESERVE - (now2 - t0)
            if rem > 1.5:
                y = lp_polish_capped(y, seconds=min(1.8, rem), max_pairs=1400)
        v = max_cos(y)
        if np.all(np.isfinite(y)) and v < best_v - 1e-12:
            best_x, best_v = y, v
        it += 1
    remain = BUDGET - (time.time() - t0)
    if remain > 2.5:
        best_x = lp_polish_capped(best_x, seconds=min(remain - 1.0, FINAL_RESERVE), max_pairs=1600)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

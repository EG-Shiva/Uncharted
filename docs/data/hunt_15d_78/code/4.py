import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_refine(y, steps=450, lr0=0.04, tau0=0.01, tau1=0.0007):
    n, d = y.shape
    y = y.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 * (1.0 - 0.7 * frac)
        G = y @ y.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # softmax weights, avoid overflow
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        S = W.sum() / 2.0
        if not np.isfinite(S) or S == 0:
            break
        Grad = (W / S) @ y
        proj = (Grad * y).sum(axis=1, keepdims=True)
        Gt = Grad - proj * y
        # clip tangential step for stability
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # optional clip very large grads
        # Gt = Gt / np.maximum(1.0, gn * 5.0)
        y = y - lr * Gt
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y = y / nrm
    return y

def lp_polish(x, seconds, rinit=0.005):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = rinit
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 78, 15
    best = None
    best_val = 1e9
    for f in ["best.npy", "record.npy"]:
        try:
            a = np.load(f)
            if a.shape == (78, 15):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # exploration: basin hopping
    it = 0
    # reserve ~28s for final polish
    while time.time() - t0 < BUDGET - 28.0:
        it += 1
        elapsed = time.time() - t0
        if elapsed > BUDGET - 30:
            break
        cyc = it % 12
        if cyc == 0:
            sigma = 0.12
        elif cyc in (1, 2):
            sigma = 0.05
        elif cyc in (3, 4, 5):
            sigma = 0.022
        else:
            sigma = 0.009
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        # contact counts with loose window to find hubs / loose tail
        cnt_loose = (G > best_val - 1e-3).sum(axis=1)
        cnt_tight = (G > best_val - 1e-7).sum(axis=1)
        over = np.argsort(-cnt_loose)[:8]
        ratt = np.where(cnt_tight == 0)[0]
        rand_idx = rng.choice(n, size=6, replace=False)
        # pick subset to kick hard
        kick = np.unique(np.concatenate([over[:4], rand_idx, ratt]) if len(ratt) > 0 else np.concatenate([over[:4], rand_idx]))
        noise = rng.standard_normal((n, d)) * (sigma * 0.3)
        noise[kick] += rng.standard_normal((len(kick), d)) * sigma
        cand = best + noise
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        # occasionally fully randomize rattlers to place in holes
        if len(ratt) > 0 and (it % 5 == 0):
            cand[ratt] = rng.standard_normal((len(ratt), d))
            cand[ratt] /= np.linalg.norm(cand[ratt], axis=1, keepdims=True)
        # smooth repulsion pre-opt
        steps = 350 if sigma > 0.03 else 500
        cand2 = smooth_refine(cand, steps=steps, lr0=0.04, tau0=0.008, tau1=0.0006)
        v2 = max_cos(cand2)
        # only LP-polish if promising (almost always early, stricter later)
        thresh = best_val + (0.015 if sigma > 0.03 else 0.008)
        if v2 > thresh:
            continue
        # short LP polish to reach local optimum
        remain = BUDGET - 28.0 - (time.time() - t0)
        if remain < 1.0:
            break
        sec = min(2.2, remain)
        # larger initial radius for larger kicks
        rinit = 0.008 if sigma > 0.04 else 0.005
        cand3 = lp_polish(cand2, seconds=sec, rinit=rinit)
        v3 = max_cos(cand3)
        if v3 < best_val - 1e-12:
            best = cand3
            best_val = v3
        # early stop if time low
        if time.time() - t0 > BUDGET - 28.0:
            break
        # safety: cap iterations
        if it > 80:
            break
    # final long polish of best
    remain = BUDGET - (time.time() - t0)
    remain = max(1.0, min(30.0, remain - 1.0))
    best = lp_polish(best, seconds=remain, rinit=0.003)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

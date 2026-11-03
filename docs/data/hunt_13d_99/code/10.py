import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def joint_anneal(cur, steps, a0, a1, lr0, lr1, beta, rng, noise=0.0):
    n, d = cur.shape
    x = cur.copy()
    mom = np.zeros_like(x)
    best = x.copy()
    best_s = max_cos(x)
    la = np.log(a1 / a0) / max(steps, 1)
    for s in range(steps):
        alpha = a0 * np.exp(la * s)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.einsum('ij,ij->i', Grad, x)[:, None]
        Gt = Grad - proj * x
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        lr = lr0 + (lr1 - lr0) * (s / max(steps - 1, 1))
        if noise > 0:
            nn = rng.standard_normal(size=(n, d)) * (noise * lr)
            nn -= (nn * x).sum(axis=1, keepdims=True) * x
            step = step + nn
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(x)
            if sc < best_s:
                best_s = sc
                best = x.copy()
    return best, best_s

def subset_anneal(cur, free_idx, steps, alpha, lr, beta, rng):
    n, d = cur.shape
    x = cur.copy()
    k = len(free_idx)
    mom = np.zeros((k, d))
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.einsum('ij,ij->i', Grad, x)[:, None]
        Gt = Grad - proj * x
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        gfree = Gt[free_idx]
        mom = beta * mom + (1.0 - beta) * gfree
        bc = 1.0 - beta ** (s + 1)
        x[free_idx] = x[free_idx] - lr * (mom / bc)
        x[free_idx] /= np.linalg.norm(x[free_idx], axis=1, keepdims=True)
    return x

def greedy_reseed(base, idx, rng, n_cand=2500):
    n, d = base.shape
    x = base.copy()
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ x.T
        dots[:, k] = -2.0
        md = dots.max(axis=1)
        x[k] = cand[np.argmin(md)]
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 170.0
    EXPLORE_UNTIL = 132.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    pop = [(best_val, best_x.copy())]
    it = 0
    while time.time() - t0 < EXPLORE_UNTIL and it < 400:
        it += 1
        r = rng.random()
        if r < 0.7 or len(pop) == 1:
            base = best_x
        else:
            base = pop[rng.integers(0, len(pop))][1]
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        # contact counts at current max
        m0 = rowmax.max()
        contacts = (G > m0 - 1e-9).sum(axis=1)
        score_pts = rowmax + 0.002 * contacts
        worst_order = np.argsort(score_pts)
        mode = rng.random()
        if mode < 0.50:
            # cluster around worst hub: worst point + its nearest neighbours
            w = worst_order[-1]
            neigh = np.argsort(G[w])[-12:]
            kk = int(rng.integers(5, 11))
            # pick kk among worst global + neighbours
            cand_pool = np.concatenate([worst_order[-8:], neigh])
            cand_pool = np.unique(cand_pool)
            if len(cand_pool) > kk:
                idx = rng.choice(cand_pool, size=kk, replace=False)
            else:
                idx = cand_pool
            x0 = greedy_reseed(base, idx, rng, n_cand=2500)
            # subset relax frozen background
            x0 = subset_anneal(x0, idx, steps=int(rng.integers(250, 600)), alpha=float(rng.uniform(30, 70)), lr=float(rng.uniform(0.05, 0.12)), beta=0.9, rng=rng)
            a0 = float(rng.uniform(18, 35)); a1 = float(rng.uniform(80, 160))
            steps = int(rng.integers(1200, 2400)); lr0 = float(rng.uniform(0.03, 0.07)); lr1 = float(rng.uniform(0.005, 0.02))
            x1, s1 = joint_anneal(x0, steps, a0, a1, lr0, lr1, 0.9, rng, noise=0.0005 if rng.random() < 0.4 else 0.0)
        elif mode < 0.78:
            # kick worst k with correlated gaussian
            kk = int(rng.integers(3, 10))
            idx = worst_order[-kk:]
            sig = float(rng.uniform(0.05, 0.35))
            x0 = base.copy()
            x0[idx] = x0[idx] + rng.standard_normal(size=(kk, d)) * sig
            # occasionally also jitter neighbours slightly
            if rng.random() < 0.5:
                x0 += rng.standard_normal(size=(n, d)) * (sig * 0.12)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            a0 = float(rng.uniform(20, 40)); a1 = float(rng.uniform(70, 150))
            steps = int(rng.integers(1000, 2200)); lr0 = float(rng.uniform(0.02, 0.06)); lr1 = float(rng.uniform(0.005, 0.02))
            x1, s1 = joint_anneal(x0, steps, a0, a1, lr0, lr1, 0.9, rng, noise=0.0)
        else:
            # global shake + long anneal
            sig = float(rng.uniform(0.02, 0.15))
            x0 = base + rng.standard_normal(size=(n, d)) * sig
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            a0 = float(rng.uniform(15, 30)); a1 = float(rng.uniform(60, 130))
            steps = int(rng.integers(1400, 2600)); lr0 = float(rng.uniform(0.02, 0.05)); lr1 = float(rng.uniform(0.004, 0.015))
            x1, s1 = joint_anneal(x0, steps, a0, a1, lr0, lr1, 0.9, rng, noise=0.0)
        s1 = max_cos(x1)
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        elif len(pop) < 5 and s1 < pop[-1][0] - 1e-11:
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        elif rng.random() < 0.03 and len(pop) < 5:
            # diversity: far in L2 but decent score
            dmin = min([float(((x1 - q) ** 2).sum()) for _, q in pop])
            if dmin > 1.0 and s1 < best_val + 0.01:
                pop.append((s1, x1.copy()))
                pop = sorted(pop, key=lambda z: z[0])[:5]
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 6:
        best_x = lp_polish(best_x, remain * 0.8)
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

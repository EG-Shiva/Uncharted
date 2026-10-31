import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_opt(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k1 != k0 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / denom if mu > 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = x / nrm
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def farthest_samples(Xkeep, rng, nc=2000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        base = np.load("best.npy")
    except Exception:
        try:
            base = np.load("record.npy")
        except Exception:
            base = rng.standard_normal((n, d))
    base = np.asarray(base, dtype=float)
    if base.shape != (n, d):
        base = rng.standard_normal((n, d))
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    base_val = max_cos(base)
    best_x = base.copy()
    best_v = base_val
    # population of diverse candidates (keep top 6)
    pop = [(best_v, best_x.copy())]
    def add_pop(v, x):
        pop.append((v, x.copy()))
        pop.sort(key=lambda t: t[0])
        del pop[6:]
    # Phase A: random basins with fast smooth repulsion
    # reserve times
    phaseA_end = t0 + 75.0
    trial = 0
    while time.time() < phaseA_end and time.time() < DEADLINE - 60:
        trial += 1
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        k0 = float(rng.choice([5.0, 8.0, 12.0, 15.0]))
        k1 = float(rng.choice([120.0, 200.0, 300.0, 450.0]))
        lr0 = float(rng.choice([0.05, 0.08, 0.12, 0.18]))
        lr1 = float(rng.choice([0.002, 0.004, 0.007, 0.01]))
        mu = float(rng.choice([0.80, 0.85, 0.9, 0.93]))
        noise = float(rng.choice([0.0, 0.005, 0.01, 0.02]))
        iters = int(rng.choice([1200, 1600, 2200, 2800]))
        x = smooth_opt(x0, iters, k0, k1, lr0, lr1, mu, noise, rng)
        v = max_cos(x)
        if v < pop[-1][0] or len(pop) < 6:
            add_pop(v, x)
        if v < best_v:
            best_v, best_x = v, x.copy()
        # early exit if already beating record by margin to go polish
        if best_v < 0.8485:
            break
    # Phase B: kick overloaded points -> void reinsert + relax (topology changes)
    phaseB_end = t0 + 115.0
    biter = 0
    while time.time() < phaseB_end and time.time() < DEADLINE - 55:
        biter += 1
        # pick base from pop (bias to best but keep diversity)
        idx_pick = int(rng.integers(len(pop)))
        if rng.random() < 0.5:
            idx_pick = 0
        bx = pop[idx_pick][1].copy()
        G = bx @ bx.T
        np.fill_diagonal(G, -2.0)
        maxper = G.max(axis=1)
        k = int(rng.integers(2, 9))
        worst = np.argsort(-maxper)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[worst] = False
        Xkeep = bx[keep_mask]
        # reinsert one by one into voids
        newpts = []
        cur = Xkeep
        for _ in range(k):
            nc = 1500 if rng.random() < 0.7 else 4000
            p = farthest_samples(cur, rng, nc=nc)
            newpts.append(p)
            cur = np.vstack([cur, p.reshape(1, -1)])
        x = bx.copy()
        x[worst] = np.array(newpts)
        # short relax with sharp focus
        k0 = 12.0
        k1 = float(rng.choice([200.0, 350.0]))
        lr0 = float(rng.choice([0.06, 0.1]))
        lr1 = 0.003
        mu = 0.9
        noise = float(rng.choice([0.0, 0.008]))
        iters = int(rng.choice([600, 900, 1300]))
        x = smooth_opt(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        v = max_cos(x)
        if v < pop[-1][0] or len(pop) < 6:
            add_pop(v, x)
        if v < best_v:
            best_v, best_x = v, x.copy()
    # Phase C: LP polish top candidates (prefer non-record basins)
    # sort pop, polish up to 3 with remaining time
    pop.sort(key=lambda t: t[0])
    # deduplicate record basin: if candidate extremely close to base, deprioritize
    # move best non-record to front if within 1e-3
    polished_best_v = best_v
    polished_best_x = best_x.copy()
    # build polish order: take up to 4 best with value within 0.02 of best
    order = []
    for v, x in pop:
        if v < best_v + 0.02:
            order.append((v, x))
        if len(order) >= 4:
            break
    # ensure global best first
    order.sort(key=lambda t: t[0])
    for v, x in order:
        if time.time() > DEADLINE - 8:
            break
        # budget per polish: divide remaining
        remain = DEADLINE - time.time()
        budget = min(22.0, remain - 5.0)
        if budget < 4:
            break
        y = lp_polish(x.copy(), budget)
        vy = max_cos(y)
        if vy < polished_best_v:
            polished_best_v, polished_best_x = vy, y.copy()
        if vy < best_v - 1e-12:
            best_v, best_x = vy, y.copy()
    # final choice: polished best if better, else incumbent best
    out = polished_best_x if polished_best_v <= best_v + 1e-12 else best_x
    # also if we never beat record, return the better of out and base (out is >=)
    # ensure valid
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    # safety: replace any nan
    if not np.all(np.isfinite(out)):
        out = base.copy()
    return out

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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
        if res.status != 0 or res.x is None:
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

def hole_candidate(others, rng, M=8000):
    cand = rng.normal(size=(M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[np.argmin(worst)]

def adam_smooth(x, steps, T0, T1, lr0, lr1, rng=None, jitter=0.0):
    x = x.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute bias denom powers
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = (G - cmax) / T
        np.clip(E, -30.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (it + 1)
        c2 = 1.0 - b2 ** (it + 1)
        mhat = m1 / c1
        vhat = m2 / c2
        upd = mhat / (np.sqrt(vhat) + eps)
        rd = np.sum(upd * x, axis=1, keepdims=True)
        upd_t = upd - rd * x
        if jitter > 0 and rng is not None:
            nz = rng.normal(size=x.shape) * jitter
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            upd_t = upd_t + nz
        x = x - lr * upd_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 147.0
    rng = np.random.default_rng(12345)
    n = 153
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
        except Exception:
            continue
        a = np.asarray(a, dtype=float)
        if a.shape == (n, 4) and np.all(np.isfinite(a)):
            nr = np.linalg.norm(a, axis=1, keepdims=True)
            if np.all(nr > 1e-12):
                a = a / nr
                cands.append((max_cos(a), a))
    if len(cands) == 0:
        x0 = rng.normal(size=(n, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda t: t[0])
        x0 = cands[0][1].copy()
    best = x0.copy()
    best_val = max_cos(best)
    # initial quick polish to reach local optimum
    if DEADLINE - time.time() > 10:
        best = lp_polish(best, budget=3.0)
        best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val
    Tb = 0.0022
    it = 0
    noimp = 0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 25:
        it += 1
        # contact analysis on cur
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        cv = float(Gc.max())
        tight = Gc > cv - 1e-7
        # exclude diagonal (already -2)
        tight_deg = tight.sum(axis=1)
        loose_deg = (Gc > cv - 0.005).sum(axis=1)
        rattlers = np.where(tight_deg == 0)[0]
        order_hub = np.argsort(-loose_deg)
        u = rng.random()
        y = cur.copy()
        if len(rattlers) > 0 and u < 0.35:
            nr = 1 if (len(rattlers) == 1 or rng.random() < 0.7) else 2
            ridx = rng.choice(rattlers, size=min(nr, len(rattlers)), replace=False)
            for r in ridx:
                others = np.delete(y, r, axis=0)
                y[r] = hole_candidate(others, rng, M=8000)
            y += rng.normal(size=y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, jit = 600, 0.022, 0.0022, 0.006, 0.0006, 0.0
        elif u < 0.56:
            # hub hole-reinsertion: replace most overloaded point
            hub = int(order_hub[0] if rng.random() < 0.8 else rng.choice(order_hub[:10]))
            others = np.delete(y, hub, axis=0)
            y[hub] = hole_candidate(others, rng, M=8000)
            y += rng.normal(size=y.shape) * 0.0025
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, jit = 650, 0.025, 0.0025, 0.006, 0.0005, 0.0
        elif u < 0.76:
            # cluster shake around random center + its nearest neighbours
            c = int(rng.integers(n))
            dots = cur[c] @ cur.T
            nn = np.argsort(-dots)[:int(rng.integers(6, 13))]
            y[nn] += rng.normal(size=(len(nn), 4)) * 0.16
            y += rng.normal(size=y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, jit = 650, 0.025, 0.003, 0.007, 0.0006, 0.15
        elif u < 0.90:
            sig = 0.007 + 0.008 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, jit = 500, 0.018, 0.0022, 0.005, 0.0005, 0.0
        else:
            sig = 0.035 + 0.03 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1, jit = 550, 0.030, 0.004, 0.007, 0.0007, 0.25
        ys = adam_smooth(y, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1, rng=rng, jitter=jit)
        v = max_cos(ys)
        accept = False
        if v < cur_val:
            accept = True
        else:
            if rng.random() < np.exp((cur_val - v) / Tb):
                accept = True
        if accept:
            cur, cur_val = ys, v
            noimp = 0 if v < best_val else noimp + 1
        else:
            noimp += 1
        if accept and v < best_val + 0.0025 and time.time() < EXPLORE_END - 2:
            z = lp_polish(ys, budget=0.9)
            vz = max_cos(z)
            if vz < best_val - 1e-12:
                best, best_val = z, vz
                cur, cur_val = z.copy(), vz
                noimp = 0
            elif vz < cur_val:
                cur, cur_val = z, vz
        else:
            if v < best_val - 1e-12:
                best, best_val = ys, v
                noimp = 0
        if noimp > 35 or cur_val > best_val + 0.012:
            cur = best.copy() + rng.normal(size=best.shape) * (0.006 + 0.004 * rng.random())
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = max_cos(cur)
            noimp = 0
            Tb = max(0.0011, Tb * 0.985)
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 3:
        best = lp_polish(best, budget=min(remaining, 22.0))
        best_val = max_cos(best)
        # tiny final adam touch + repolish if time left
        if DEADLINE - time.time() > 8:
            yf = best + rng.normal(size=best.shape) * 0.0015
            yf /= np.linalg.norm(yf, axis=1, keepdims=True)
            yf = adam_smooth(yf, steps=300, T0=0.008, T1=0.0015, lr0=0.003, lr1=0.0003)
            yf = lp_polish(yf, budget=DEADLINE - time.time() - 2.0)
            vf = max_cos(yf)
            if vf < best_val:
                best, best_val = yf, vf
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

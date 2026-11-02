import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _anneal(x, steps, t0, t1, lr0, lr1, deadline=None, check_every=200):
    n, d = x.shape
    for k in range(steps):
        if deadline is not None and (k % check_every == 0) and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        t = t0 + (t1 - t0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        # stable softmax weights
        E = np.exp(t * (G - M))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            continue
        Gr = (E @ x) / S
        # tangent projection (reuse factor absorbed in lr; scale up)
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        # scale: Gr was divided by S (~2*sum), multiply to keep step size meaningful
        # E@x/S has magnitude ~ 1/(n) roughly; rescale by n to make lr intuitive
        x = x - (lr * n) * Gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _largest_holes(X_fixed, n_holes, rng, n_sample=4000):
    # find vectors minimizing max dot to X_fixed
    n, d = X_fixed.shape
    cand = rng.standard_normal(size=(n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_fixed.T  # (sample, n)
    m = dots.max(axis=1)
    idx = np.argpartition(m, n_holes)[:n_holes]
    # sort by quality
    idx = idx[np.argsort(m[idx])]
    return cand[idx]

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    HARD_END = t_start + 170.0
    EXPLORE_END = t_start + 132.0
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            z = rng.standard_normal(size=(76, 15))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            return z
    x0 = np.asarray(x0, dtype=np.float64)
    # ensure shape (76,15)
    if x0.shape != (76, 15):
        # try to reshape/truncate deterministically
        flat = x0.ravel()
        need = 76 * 15
        if flat.size < need:
            extra = rng.standard_normal(size=(need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(76, 15)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best = x0.copy()
    best_val = _max_cos(best)
    cur_val = best_val
    n, d = 76, 15
    cyc = 0
    # quick pre-anneal params variety
    while time.time() < EXPLORE_END and time.time() < HARD_END - 30:
        cyc += 1
        mode = cyc % 5
        Xt = best.copy()
        if mode == 0:
            # global small shake
            sigma = 0.025
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, t0a, t1a, lr0, lr1 = 700, 80.0, 600.0, 0.25, 0.02
        elif mode == 1:
            # subset strong shake
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.18
            Xt[idx] += sigma * rng.standard_normal(size=(k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, t0a, t1a, lr0, lr1 = 900, 60.0, 800.0, 0.30, 0.015
        elif mode == 2:
            # rattler reinsertion
            rm, G = _row_max(Xt)
            order = np.argsort(rm)
            n_rep = 3 if cyc % 10 != 2 else 6
            rep_idx = order[:n_rep]
            keep = np.ones(n, dtype=bool); keep[rep_idx] = False
            X_fixed = Xt[keep]
            holes = _largest_holes(X_fixed, n_rep, rng, n_sample=5000)
            Xt[rep_idx] = holes
            # tiny jitter to break symmetry
            Xt += 0.005 * rng.standard_normal(size=Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, t0a, t1a, lr0, lr1 = 1000, 70.0, 900.0, 0.28, 0.015
        elif mode == 3:
            # jammed-point shake (high contacts)
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            M = G.max()
            cnt = (G > M - 0.002).sum(axis=1)
            # pick most jammed
            order = np.argsort(-cnt)
            k = 5
            idx = order[:k]
            sigma = 0.10
            Xt[idx] += sigma * rng.standard_normal(size=(k, d))
            # also jitter one rattler
            rm = G.max(axis=1)
            ridx = np.argsort(rm)[:1]
            Xt[ridx] += 0.05 * rng.standard_normal(size=(1, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, t0a, t1a, lr0, lr1 = 900, 60.0, 700.0, 0.30, 0.02
        else:
            # random direction large hole + global medium shake
            k = 4
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.12
            Xt[idx] += sigma * rng.standard_normal(size=(k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, t0a, t1a, lr0, lr1 = 800, 90.0, 1000.0, 0.22, 0.012
        # anneal with deadline guard
        Xt = _anneal(Xt, steps, t0a, t1a, lr0, lr1, deadline=min(EXPLORE_END, HARD_END - 30))
        v = _max_cos(Xt)
        if v < best_val - 1e-12:
            best, best_val = Xt, v
        # early exit if we beat record by clear margin and time is short? continue to polish
        if best_val < 0.1675:
            break
        if time.time() > EXPLORE_END:
            break
    # final selection: ensure best is kept
    # short high-sharpness refinement without LP (fast) on best
    if time.time() < HARD_END - 28:
        remain = HARD_END - 28 - time.time()
        # cap steps to fit
        Xt = _anneal(best.copy(), 600, 200.0, 1500.0, 0.12, 0.008, deadline=HARD_END - 28)
        v = _max_cos(Xt)
        if v < best_val:
            best, best_val = Xt, v
    # LP polish with remaining budget (max ~25s)
    lp_budget = max(2.0, min(25.0, HARD_END - time.time() - 2.0))
    if lp_budget > 2.5:
        try:
            Xp = _lp_polish(best, lp_budget)
            vp = _max_cos(Xp)
            if vp <= best_val:
                best = Xp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: if somehow worse than input, return input
    try:
        if _max_cos(best) > _max_cos(x0) + 1e-12:
            best = x0
    except Exception:
        pass
    return best

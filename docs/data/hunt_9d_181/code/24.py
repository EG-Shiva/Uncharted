import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, init_radius=0.007, max_pairs=1500):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        rem = budget - (now - t0)
        if rem < 0.2 or radius < 1e-13:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(25.0 * radius, 2e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > max_pairs:
            # keep tightest
            part = np.argpartition(c_all[sel_idx], -max_pairs)[-max_pairs:]
            sel_idx = sel_idx[part]
        if len(sel_idx) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        csel = c_all[sel_idx]
        p = len(sel_idx)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.04)
        else:
            radius *= 0.5
    return x

def _repulse(x, steps, alpha, lr, max_step=0.03):
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = float(np.linalg.norm(step, axis=1).max())
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _pinch(x, steps, width, lr, max_step=0.03):
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = G - (m - width)
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = float(np.linalg.norm(step, axis=1).max())
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _deepest_hole(Xkeep, rng, n_cand=2000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    worst = dots.max(axis=1)
    return cand[int(np.argmin(worst))]

def solve():
    t0 = time.time()
    LIMIT = 171.0
    rng = np.random.default_rng(12345)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best_val = np.inf
    x_best = None
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    x_best = x_best.copy()
    ruin_cycle = [40, 18, 60, 25, 35, 12, 50, 22, 30, 15]
    it = 0
    # precompute nothing
    while time.time() - t0 < LIMIT - 26.0:
        it += 1
        remain_outer = LIMIT - 26.0 - (time.time() - t0)
        if remain_outer < 3.0:
            break
        y = None
        # 1 in 7: fresh random start for new basin / balanced graph
        if it % 7 == 0:
            y = rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # fast global spread
            y = _repulse(y, steps=180, alpha=22.0, lr=0.006)
        else:
            y = x_best.copy()
            k = ruin_cycle[(it - 1) % len(ruin_cycle)] + int(rng.integers(-3, 4))
            k = max(8, min(n - 5, k))
            # contact counts to target hubs + random
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.005).sum(axis=1)
            # hubs: top k//2, random: rest
            order = np.argsort(-cnt, kind="stable")
            nhub = k // 2
            victims = set(order[:nhub].tolist())
            # add random indices to fill k
            while len(victims) < k:
                victims.add(int(rng.integers(0, n)))
            # occasionally also kill a rattler-adjacent region: add neighbors of random point
            if rng.random() < 0.4:
                c0 = int(rng.integers(0, n))
                dots = y[c0] @ y.T
                nn = np.argsort(-dots)[:6]
                for v in nn[:2]:
                    if len(victims) >= k + 2:
                        break
                    victims.add(int(v))
            victims = np.array(sorted(victims), dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            # sequential deepest-hole reinsertion
            for v in victims:
                h = _deepest_hole(Xkeep, rng, n_cand=2000)
                h = h + 0.005 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])
            # extra jitter on reinserted to break symmetry
            y[victims] += 0.01 * rng.standard_normal(size=(len(victims), d))
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
        # smoothing: alternate regimes
        if it % 2 == 0:
            y = _repulse(y, steps=220, alpha=26.0, lr=0.005)
            y = _pinch(y, steps=140, width=0.03, lr=0.25)
        else:
            y = _repulse(y, steps=160, alpha=16.0, lr=0.007)
            y = _pinch(y, steps=180, width=0.045, lr=0.22)
        # short LP polish
        rem = LIMIT - 26.0 - (time.time() - t0)
        if rem < 1.5:
            break
        budget = min(4.0, rem)
        y = _lp_polish(y, budget=budget, init_radius=0.007)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
    # final polish
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        x_best = _lp_polish(x_best, budget=rem, init_radius=0.004, max_pairs=2000)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

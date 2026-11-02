import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=0.005, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x, steps, s0, s1, lr):
    n = x.shape[0]
    for it in range(steps):
        s = s0 + (s1 - s0) * (it / max(1, steps - 1))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        gmax = G.max()
        # exp(s*(G-gmax)), diag -> 0
        W = np.exp(s * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        Grad = (W @ x) / (S * 0.5)
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip huge steps
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        tmp = rng.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z))
    best = best.copy()
    best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val

    it = 0
    # main basin-hopping, leave ~18s for final polish
    while time.time() < HARD_END - 18.0:
        it += 1
        r = rng.rand()
        if r < 0.30:
            sigma = 0.003 + rng.rand() * 0.010
        elif r < 0.70:
            sigma = 0.015 + rng.rand() * 0.035
        else:
            sigma = 0.05 + rng.rand() * 0.07
        x = cur + rng.randn(n, d) * sigma
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        # choose victims: focus on worst + random
        Gtmp = x @ x.T
        np.fill_diagonal(Gtmp, -2.0)
        rowmax = Gtmp.max(axis=1)
        order = np.argsort(-rowmax)
        if rng.rand() < 0.75:
            k = int(rng.choice([1, 1, 2, 2, 3]))
        else:
            k = 1
        used = set()
        for t in range(k):
            if time.time() > HARD_END - 18.0:
                break
            if rng.rand() < 0.65:
                # worst-biased
                idx_pool = order[:8]
                victim = int(idx_pool[rng.randint(0, len(idx_pool))])
            else:
                victim = int(rng.randint(0, n))
            if victim in used:
                continue
            used.add(victim)
            mask = np.ones(n, dtype=bool)
            mask[victim] = False
            others = x[mask]
            nc = 2500
            cand = rng.randn(nc, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ others.T
            m = dots.max(axis=1)
            x[victim] = cand[int(np.argmin(m))]

        # annealed soft-max repulsion
        s0 = 30.0 + rng.rand() * 20.0
        s1 = 140.0 + rng.rand() * 140.0
        steps = 180 + rng.randint(0, 150)
        lr = 0.015 + rng.rand() * 0.025
        x = _smooth(x, steps, s0, s1, lr)
        if time.time() > HARD_END - 18.0:
            v = _max_cos(x)
            if v < best_val:
                best, best_val = x, v
            break
        # short LP polish
        x = _lp_polish(x, seconds=1.1, r0=0.006, t_deadline=HARD_END - 18.0)
        v = _max_cos(x)
        if v < best_val - 1e-12:
            best, best_val = x.copy(), v
            cur, cur_val = x.copy(), v
            # extra exploit if promising: longer polish
            if time.time() < HARD_END - 25.0:
                x2 = _lp_polish(x, seconds=1.0, r0=0.002, t_deadline=HARD_END - 18.0)
                v2 = _max_cos(x2)
                if v2 < best_val:
                    best, best_val = x2.copy(), v2
                    cur, cur_val = x2.copy(), v2
        elif v < cur_val:
            cur, cur_val = x.copy(), v
        else:
            # annealing wander
            Temp = 0.0015 * (0.995 ** (it // 3)) + 1e-4
            if rng.rand() < np.exp(-(v - cur_val) / Temp) * 0.5:
                cur, cur_val = x.copy(), v
            # occasionally restart from best
            if rng.rand() < 0.08:
                cur, cur_val = best.copy(), best_val

    # final intensive polish of best
    remain = HARD_END - time.time() - 1.0
    if remain > 2.0:
        # two-stage: medium radius then tiny
        half = remain * 0.6
        best = _lp_polish(best, seconds=half, r0=0.004, t_deadline=HARD_END - 0.5)
        remain2 = HARD_END - time.time() - 0.5
        if remain2 > 1.0:
            best = _lp_polish(best, seconds=remain2, r0=0.001, t_deadline=HARD_END - 0.2)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

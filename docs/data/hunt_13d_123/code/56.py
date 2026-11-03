import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _descend(v0, steps, b0, b1, l0, l1, mu=0.85):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / np.maximum(gn, 1.0)
        vel = mu * vel + gt
        v = v - lr * vel
        v = _norm(v)
    return v

def _deepest_hole(Ykeep, rng, n_cand=5000):
    n, d = Ykeep.shape[0], Ykeep.shape[1]
    C = rng.randn(n_cand, d)
    C = _norm(C)
    scores = (C @ Ykeep.T).max(axis=1)
    return C[int(np.argmin(scores))]

def _lp_polish(x, seconds, p_cap=1500):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c[idx], p_cap - 1)[:p_cap]
            ns = np.zeros_like(sel)
            ns[idx[part]] = True
            sel = ns
        I = iu0[sel]; J = iu1[sel]
        cs = c[sel]
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return bx

def _greedy(x, seconds, rng, t_end):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    sigma = 0.0015
    it = 0
    while time.time() < t_end - 0.5:
        it += 1
        if it % 3 == 0:
            i = int(rng.randint(n))
        else:
            rm = G.max(axis=1)
            w = np.argpartition(-rm, 6)[:6]
            i = int(w[rng.randint(len(w))])
        r = rng.randn(d)
        r = r - (r @ x[i]) * x[i]
        rn = np.linalg.norm(r)
        if rn < 1e-12:
            continue
        r /= rn
        step = sigma * (0.5 + rng.rand())
        cand = x[i] + step * r
        cand /= (np.linalg.norm(cand) + 1e-18)
        dots = x @ cand
        dots[i] = -2.0
        md = float(dots.max())
        if md < best:
            x[i] = cand
            gcol = x @ cand
            G[i, :] = gcol; G[:, i] = gcol
            G[i, i] = -2.0
            best = float(G.max())
        if it % 800 == 0:
            sigma *= 0.97
            if sigma < 2e-4:
                sigma = 2e-4
            if time.time() >= t_end - 0.5:
                break
    return x

def solve():
    t0 = time.time()
    T_HOP = t0 + 115.0
    T_GREEDY = t0 + 150.0
    T_LP = t0 + 172.0
    rng = np.random.RandomState(42)
    n, d = 123, 13
    xb = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < bv:
                    bv = v; xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.randn(n, d))
        bv = _max_cos(xb)
    x_best = xb.copy()
    best_val = bv
    trial = 0
    while time.time() < T_HOP and trial < 400:
        trial += 1
        mode = trial % 6
        xt = x_best.copy()
        if mode == 0:
            xt = _norm(xt + 0.008 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 500, 45.0, 500.0, 0.007, 0.0006
        elif mode == 1:
            rm = _row_max(x_best)
            widx = np.argsort(-rm)[:5]
            xt[widx] += 0.06 * rng.randn(len(widx), d)
            xt = _norm(xt + 0.002 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 600, 30.0, 500.0, 0.009, 0.0007
        elif mode == 2:
            k = 1
            idx = rng.choice(n, k, replace=False)
            xt[idx] = rng.randn(k, d)
            xt = _norm(xt + 0.002 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 650, 28.0, 450.0, 0.010, 0.0007
        elif mode == 3:
            # rattler / low-degree to deepest hole
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 1e-9).sum(axis=1)
            li = int(np.argmin(deg))
            mask = np.ones(n, dtype=bool); mask[li] = False
            hole = _deepest_hole(x_best[mask], rng, n_cand=4000)
            xt[li] = hole + 1e-3 * rng.randn(d)
            xt = _norm(xt + 0.002 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 600, 32.0, 500.0, 0.009, 0.0007
        elif mode == 4:
            xt = _norm(xt + 0.025 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 650, 25.0, 420.0, 0.010, 0.0008
        else:
            k = 6
            idx = rng.choice(n, k, replace=False)
            xt[idx] += 0.04 * rng.randn(k, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 550, 35.0, 550.0, 0.008, 0.0006
        if time.time() >= T_HOP:
            break
        v = _descend(xt, steps, b0, b1, l0, l1)
        sv = _max_cos(v)
        if sv < best_val:
            best_val = sv
            x_best = v.copy()
    if time.time() < T_GREEDY - 3:
        x_best = _greedy(x_best, 0, rng, T_GREEDY)
        bv2 = _max_cos(x_best)
        if bv2 < best_val:
            best_val = bv2
    if time.time() < T_LP - 4:
        sec = min(T_LP - time.time() - 1.0, 14.0)
        if sec > 3:
            xp = _lp_polish(x_best, sec, p_cap=1500)
            pv = _max_cos(xp)
            if pv < best_val:
                x_best = xp
                best_val = pv
    x_best = _norm(x_best)
    return x_best

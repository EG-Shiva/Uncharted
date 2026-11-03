import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _descend(v0, steps, b0, b1, l0, l1, mu=0.8):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # softmax weights, avoid overflow
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / rs
        # normalize step scale
        sc = np.sqrt(np.sum(gt * gt)) + 1e-12
        # keep lr as-is (already small)
        vel = mu * vel + gt
        v = v - lr * vel
        v = _norm(v)
    return v

def _lp_polish(x, seconds, p_cap=1200):
    t_end = time.time() + seconds
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and radius > 1e-14 and it < 60:
        it += 1
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.flatnonzero(sel)
            part = np.argpartition(-c[idx], p_cap - 1)[:p_cap]
            ns = np.zeros_like(sel)
            ns[idx[part]] = True
            sel = ns
        I = iu0[sel]; J = iu1[sel]; cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d)).reshape(p, d)
        cJ = (J[:, None] * d + np.arange(d)).reshape(p, d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            # pull back to best if radius collapsed
            if radius < 1e-7:
                x = bx.copy()
    return bx

def solve():
    t0 = time.time()
    HARD = 172.0  # leave margin under 180
    n, d = 123, 13
    rng = np.random.RandomState(1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.randn(n, d))
        best_val = _max_cos(best)
    x_best = best.copy()

    # quick hole sampler reused
    def deepest_hole(x_keep, n_cand=4000):
        cand = _norm(rng.randn(n_cand, d))
        dots = cand @ x_keep.T
        md = dots.max(axis=1)
        return cand[int(np.argmin(md))]

    trial = 0
    # exploration budget: stop early to reserve final polish
    while time.time() - t0 < 125.0 and trial < 300:
        trial += 1
        remain = HARD - (time.time() - t0)
        if remain < 32:
            break
        mode = trial % 5
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        if mode == 0:
            xt = x_best + 0.006 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 700, 50.0, 600.0, 0.006, 0.0006
        elif mode == 1:
            xt = x_best.copy()
            k = 2 if (trial % 2 == 0) else 3
            worst = order[:k]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            Y = x_best[keep]
            for wi in worst:
                xt[wi] = deepest_hole(Y, n_cand=3000) + 0.002 * rng.randn(d)
            xt = xt + 0.003 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 800, 40.0, 600.0, 0.007, 0.0007
        elif mode == 2:
            # kick hubs harder
            xt = x_best + 0.004 * rng.randn(n, d)
            w = order[:8]
            xt[w] += 0.05 * rng.randn(len(w), d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 800, 35.0, 550.0, 0.008, 0.0007
        elif mode == 3:
            xt = x_best + 0.018 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 900, 30.0, 500.0, 0.009, 0.0008
        else:
            # low-degree / rattler reinsertion
            deg_win = 0.004
            m = float(G.max())
            deg = (G > m - deg_win).sum(axis=1)
            low = np.argsort(deg)[:2]
            xt = x_best.copy()
            keep = np.ones(n, dtype=bool); keep[low] = False
            Y = x_best[keep]
            for wi in low:
                xt[wi] = deepest_hole(Y, n_cand=3000) + 0.002 * rng.randn(d)
            xt = xt + 0.003 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 800, 40.0, 650.0, 0.007, 0.0006

        v = _descend(xt, steps, b0, b1, l0, l1)
        sv = _max_cos(v)
        if sv < best_val:
            best_val = sv
            x_best = v.copy()
            # short LP snap on strict improvement if time allows
            if HARD - (time.time() - t0) > 35:
                vp = _lp_polish(v, 2.5)
                pv = _max_cos(vp)
                if pv < best_val:
                    best_val = pv
                    x_best = vp.copy()
        elif sv < best_val + 0.002 and (trial % 7 == 0):
            # occasional polish of promising basin
            if HARD - (time.time() - t0) > 40:
                vp = _lp_polish(v, 1.5)
                pv = _max_cos(vp)
                if pv < best_val:
                    best_val = pv
                    x_best = vp.copy()
    # final polish with remaining time
    remain = HARD - (time.time() - t0)
    if remain > 6:
        x_best = _lp_polish(x_best, min(remain - 2, 26.0))
    x_best = _norm(x_best)
    return x_best

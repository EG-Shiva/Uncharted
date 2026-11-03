import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(v0, steps, b0, b1, l0, l1, mu=0.82):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        # normalize step scale
        vel = mu * vel + gt
        v = v - lr * vel
        v = _norm(v)
    return v

def _hole_replace(x, k, rng, n_cand=20000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    worst = np.argsort(-rm)[:k]
    keep = np.ones(n, dtype=bool)
    keep[worst] = False
    Y = x[keep]
    S = rng.randn(n_cand, d)
    S = _norm(S)
    dots = S @ Y.T
    md = dots.max(axis=1)
    order = np.argsort(md)
    # greedy diverse pick among best holes
    chosen = []
    for idx in order:
        c = S[idx]
        ok = True
        for j in chosen:
            if float(c @ S[j]) > 0.90:
                ok = False
                break
        if ok:
            chosen.append(int(idx))
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
                if len(chosen) >= k:
                    break
    xn = x.copy()
    for wi, ci in zip(worst, chosen):
        xn[wi] = S[ci]
    return _norm(xn)

def _lp_polish(x, seconds, t_dead):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_dead:
        if radius < 1e-14:
            break
        Gall = x @ x.T
        c = Gall[iu0, iu1]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        # cap constraints
        if cnt > 1600:
            idx = np.where(sel)[0]
            part = np.argpartition(-c[idx], 1600 - 1)[:1600]
            ns = np.zeros_like(sel)
            ns[idx[part]] = True
            sel = ns
            cnt = 1600
        I = iu0[sel]; J = iu1[sel]; cs = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d)).reshape(p, d)
        cJ = (J[:, None] * d + np.arange(d)).reshape(p, d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    HOP_END = t0 + 125.0
    rng = np.random.RandomState(0)
    n, d = 123, 13
    xb = None; bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _mc(a)
                if v < bv:
                    bv = v; xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.randn(n, d)); bv = _mc(xb)
    x_best = xb.copy(); best_val = bv
    # quick hole-fix for rattler first
    try:
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        deg = (G > m0 - 1e-9).sum(axis=1)
        # move zero/low contact points to deepest holes
        low = np.where(deg <= 2)[0]
        if len(low) > 0:
            k = min(len(low), 3)
            cand = _hole_replace(x_best, k, rng)
            cand = _anneal(cand, 600, 40.0, 500.0, 0.008, 0.0008)
            sv = _mc(cand)
            if sv < best_val:
                best_val = sv; x_best = cand.copy()
    except Exception:
        pass
    trial = 0
    while time.time() < HOP_END and trial < 400:
        trial += 1
        mode = trial % 7
        try:
            if mode == 0:
                xt = _hole_replace(x_best, 1, rng, n_cand=15000)
                xt = _norm(xt + 0.004 * rng.randn(n, d))
                steps, b0, b1, l0, l1 = 500, 40.0, 500.0, 0.008, 0.0008
            elif mode == 1:
                xt = _hole_replace(x_best, 2, rng, n_cand=15000)
                xt = _norm(xt + 0.004 * rng.randn(n, d))
                steps, b0, b1, l0, l1 = 600, 35.0, 550.0, 0.009, 0.0008
            elif mode == 2:
                xt = _hole_replace(x_best, 3, rng, n_cand=12000)
                xt = _norm(xt + 0.005 * rng.randn(n, d))
                steps, b0, b1, l0, l1 = 600, 30.0, 450.0, 0.010, 0.0009
            elif mode == 3:
                xt = _norm(x_best + 0.008 * rng.randn(n, d))
                steps, b0, b1, l0, l1 = 450, 50.0, 500.0, 0.006, 0.0006
            elif mode == 4:
                xt = _norm(x_best + 0.022 * rng.randn(n, d))
                steps, b0, b1, l0, l1 = 600, 30.0, 450.0, 0.010, 0.0008
            elif mode == 5:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                w = np.argsort(-rm)[:8]
                xt = x_best.copy() + 0.004 * rng.randn(n, d)
                xt[w] += 0.06 * rng.randn(len(w), d)
                xt = _norm(xt)
                steps, b0, b1, l0, l1 = 550, 35.0, 500.0, 0.009, 0.0008
            else:
                xt = _norm(x_best + 0.045 * rng.randn(n, d))
                steps, b0, b1, l0, l1 = 700, 25.0, 400.0, 0.012, 0.0010
            # shorten if low on time
            if HOP_END - time.time() < 8:
                steps = min(steps, 250)
            v = _anneal(xt, steps, b0, b1, l0, l1)
            sv = _mc(v)
            if sv < best_val:
                best_val = sv
                x_best = v.copy()
        except Exception:
            continue
        if time.time() >= HOP_END:
            break
    # final LP snap, strictly time-capped
    remain = T_END - time.time()
    if remain > 6:
        try:
            x_best = _lp_polish(x_best, min(remain - 1.0, 38.0), T_END - 0.5)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

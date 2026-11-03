import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_anneal(v0, steps, beta0, beta1, lr0, lr1, mu=0.85):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        vel = mu * vel + gt
        v = v - lr * vel
        v = _normalize(v)
    return v

def _replace_worst(x, k, rng, n_samples=30000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = G.max(axis=1)
    worst = np.argsort(-row_max)[:k]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[worst] = False
    Y = x[keep_mask]
    S = rng.randn(n_samples, d)
    S = _normalize(S)
    dots = S @ Y.T
    worst_dot = dots.max(axis=1)
    order = np.argsort(worst_dot)
    chosen = []
    chosen_vecs = []
    for idx in order:
        cand = S[idx]
        ok = True
        for cv in chosen_vecs:
            if float(cand @ cv) > 0.92:
                ok = False
                break
        if ok:
            chosen.append(idx)
            chosen_vecs.append(cand)
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        for idx in order:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= k:
                    break
    x_new = x.copy()
    for ii, si in zip(worst, chosen):
        x_new[ii] = S[si] + 1e-3 * rng.randn(d)
    return _normalize(x_new)

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.RandomState(0)
    n, d = 123, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(n, d)
    xb = np.array(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m = min(xb.size, n * d)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb = _normalize(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    trial = 0
    lp_calls = 0
    # exploration phase, reserve time for final polish
    while time.time() - t0 < 125 and trial < 200:
        trial += 1
        mode = trial % 6
        if mode == 0:
            xt = _replace_worst(x_best, 1, rng)
            xt = xt + 0.004 * rng.randn(n, d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1200, 40.0, 500.0, 0.008, 0.0008
        elif mode == 1:
            xt = _replace_worst(x_best, 2, rng)
            xt = xt + 0.004 * rng.randn(n, d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1400, 40.0, 600.0, 0.008, 0.0008
        elif mode == 2:
            xt = x_best + 0.008 * rng.randn(n, d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1000, 50.0, 500.0, 0.006, 0.0006
        elif mode == 3:
            xt = x_best + 0.020 * rng.randn(n, d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1400, 30.0, 500.0, 0.010, 0.0008
        elif mode == 4:
            xt = x_best + 0.040 * rng.randn(n, d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1600, 25.0, 400.0, 0.012, 0.0010
        else:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            widx = np.argsort(-rm)[:10]
            xt = x_best.copy()
            xt = xt + 0.005 * rng.randn(n, d)
            xt[widx] = xt[widx] + 0.05 * rng.randn(len(widx), d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1300, 35.0, 550.0, 0.009, 0.0008
        if time.time() - t0 > 122:
            break
        v = _smooth_anneal(xt, steps, b0, b1, l0, l1, mu=0.85)
        sv = _max_cos(v)
        need_lp = False
        lp_sec = 0.0
        if sv < best_val - 1e-7:
            need_lp = True
            lp_sec = 6.0
        elif sv < best_val + 0.003 and lp_calls < 18:
            # only polish promising ones
            if (trial % 2 == 0) or (sv < best_val + 0.0005):
                need_lp = True
                lp_sec = 3.0
        if need_lp:
            remain = 128 - (time.time() - t0)
            if remain < 4:
                break
            lp_sec = min(lp_sec, remain)
            vp = _lp_polish(v, lp_sec)
            lp_calls += 1
            pv = _max_cos(vp)
            if pv < best_val:
                best_val = pv
                x_best = vp.copy()
                # intensify around new best
                if remain > 10:
                    vp2 = _lp_polish(x_best, min(4.0, remain - 3))
                    pv2 = _max_cos(vp2)
                    if pv2 < best_val:
                        best_val = pv2
                        x_best = vp2.copy()
        # opportunistic: if smooth alone beats best, keep it
        if sv < best_val:
            best_val = sv
            x_best = v.copy()
    remain = 165 - (time.time() - t0)
    if remain > 5:
        x_best = _lp_polish(x_best, min(remain, 28.0))
    x_best = _normalize(x_best)
    return x_best

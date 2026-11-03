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

def _anneal(v0, steps, beta0, beta1, lr0, lr1, mu=0.85):
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
        # softmax weights, focus on near-max pairs
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        vel = mu * vel + gt
        v = v - lr * vel
        # project to sphere
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        v = v / nrm
    return v

def _holes_for(Y, k, rng, n_samples=25000):
    d = Y.shape[1]
    S = rng.randn(n_samples, d)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ Y.T
    worst = dots.max(axis=1)
    order = np.argsort(worst)
    chosen = []
    for idx in order:
        cand = S[idx]
        ok = True
        for cv in chosen:
            if float(cand @ cv) > 0.90:
                ok = False
                break
        if ok:
            chosen.append(cand)
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        for idx in order:
            cand = S[idx]
            # avoid duplicate
            dup = False
            for cv in chosen:
                if np.array_equal(cand, cv):
                    dup = True
                    break
            if not dup:
                chosen.append(cand)
                if len(chosen) >= k:
                    break
    return np.array(chosen)

def _lp_polish(x, seconds, t_end=None):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 60:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = iu_c[sel]
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
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_LIMIT = 170.0  # leave margin under 180
    rng = np.random.RandomState(12345)
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
    x_cur = xb.copy()
    cur_val = best_val

    trial = 0
    # exploration budget: stop early enough for final polish
    explore_end = t0 + 132.0
    while time.time() < explore_end and trial < 300:
        trial += 1
        mode = trial % 7
        # choose kick strength and re-insert count
        if mode == 0:
            k, noise = 1, 0.004
            steps, b0, b1, l0, l1 = 700, 40.0, 600.0, 0.008, 0.0008
        elif mode == 1:
            k, noise = 2, 0.004
            steps, b0, b1, l0, l1 = 800, 40.0, 700.0, 0.008, 0.0008
        elif mode == 2:
            k, noise = 3, 0.003
            steps, b0, b1, l0, l1 = 800, 30.0, 600.0, 0.009, 0.0008
        elif mode == 3:
            k, noise = 1, 0.012
            steps, b0, b1, l0, l1 = 700, 50.0, 550.0, 0.006, 0.0006
        elif mode == 4:
            k, noise = 2, 0.020
            steps, b0, b1, l0, l1 = 800, 30.0, 500.0, 0.010, 0.0008
        elif mode == 5:
            k, noise = 1, 0.040
            steps, b0, b1, l0, l1 = 900, 25.0, 400.0, 0.012, 0.0010
        else:
            k, noise = 1, 0.006
            steps, b0, b1, l0, l1 = 750, 35.0, 650.0, 0.009, 0.0007

        # base = current (wandering) mostly, sometimes best
        base = x_cur if (trial % 3 != 0) else x_best
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        worst = np.argsort(-row_max)[:k]
        keep = np.ones(n, dtype=bool); keep[worst] = False
        Y = base[keep]
        holes = _holes_for(Y, k, rng)
        xt = base.copy()
        for ii, h in zip(worst, holes):
            xt[ii] = h + 1e-3 * rng.randn(d)
        # extra shake: stronger on worst, light elsewhere
        xt = xt + noise * rng.randn(n, d)
        if mode == 6:
            # hub-targeted extra kick on top-8 overloaded
            w8 = np.argsort(-row_max)[:8]
            xt[w8] += 0.03 * rng.randn(len(w8), d)
        xt = _normalize(xt)

        if time.time() > explore_end - 2.0:
            break
        v = _anneal(xt, steps, b0, b1, l0, l1, mu=0.85)
        sv = _max_cos(v)
        if sv < best_val:
            best_val = sv
            x_best = v.copy()
        # wandering acceptance: always move if better than cur, else small prob
        if sv < cur_val:
            x_cur, cur_val = v.copy(), sv
        elif sv < cur_val + 0.004 and rng.rand() < 0.15:
            x_cur, cur_val = v.copy(), sv
        elif rng.rand() < 0.02:
            # random restart from best to avoid drift
            x_cur, cur_val = x_best.copy(), best_val

    remain = T_LIMIT - (time.time() - t0)
    if remain > 6.0:
        # final trust-region LP to freeze topology at local optimum
        x_best = _lp_polish(x_best, min(remain - 1.0, 32.0))
    x_best = _normalize(x_best)
    return x_best

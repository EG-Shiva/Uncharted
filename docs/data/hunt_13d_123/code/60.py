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

def _anneal(v0, steps, beta0, beta1, lr0, lr1):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    mu = 0.85
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        # softmax focus on near-max pairs
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        vel = mu * vel + gt
        v = v - lr * vel
        v = _norm(v)
    return v

def _holes(Y, k, rng, n_samples=12000):
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
            if float(cand @ cv) > 0.92:
                ok = False
                break
        if ok:
            chosen.append(cand)
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        for idx in order:
            cand = S[idx]
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

def _lp_polish(x, seconds, t_end):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and (time.time() < t_end):
        if radius < 1e-14:
            break
        it += 1
        if it > 40:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1400:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], 1400 - 1)[:1400]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = len(I)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(2.0, seconds / 6.0)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() > t_end - 1.0:
            break
    return bx

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    EXP_END = t0 + 132.0
    rng = np.random.RandomState(7)
    n, d = 123, 13
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

    trial = 0
    while time.time() < EXP_END and trial < 250:
        trial += 1
        mode = trial % 6
        if mode == 0:
            k, noise, steps, b0, b1, l0, l1 = 1, 0.004, 350, 40.0, 600.0, 0.008, 0.0008
        elif mode == 1:
            k, noise, steps, b0, b1, l0, l1 = 2, 0.005, 350, 40.0, 600.0, 0.008, 0.0008
        elif mode == 2:
            k, noise, steps, b0, b1, l0, l1 = 3, 0.004, 400, 30.0, 550.0, 0.009, 0.0008
        elif mode == 3:
            k, noise, steps, b0, b1, l0, l1 = 2, 0.015, 400, 35.0, 500.0, 0.009, 0.0007
        elif mode == 4:
            k, noise, steps, b0, b1, l0, l1 = 1, 0.035, 450, 25.0, 400.0, 0.011, 0.0010
        else:
            k, noise, steps, b0, b1, l0, l1 = 4, 0.006, 400, 30.0, 500.0, 0.009, 0.0008

        base = x_best
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        # contact degree for rattler targeting
        m = float(G.max())
        deg = (G > m - 1e-9).sum(axis=1)
        worst = np.argsort(-row_max)[:k]
        # occasionally force rattler (min degree) into the set
        if trial % 4 == 0:
            r = int(np.argmin(deg))
            if r not in worst:
                worst = np.array(list(worst[:-1]) + [r]) if k > 1 else np.array([r])
        keep = np.ones(n, dtype=bool); keep[worst] = False
        Y = base[keep]
        holes = _holes(Y, k, rng)
        xt = base.copy()
        for ii, h in zip(worst, holes):
            xt[ii] = h + 1e-3 * rng.randn(d)
        xt = xt + noise * rng.randn(n, d)
        if mode == 5:
            w8 = np.argsort(-row_max)[:8]
            xt[w8] += 0.025 * rng.randn(len(w8), d)
        xt = _norm(xt)
        if time.time() > EXP_END - 3.0:
            break
        v = _anneal(xt, steps, b0, b1, l0, l1)
        sv = _max_cos(v)
        if sv < best_val:
            best_val = sv
            x_best = v.copy()
            # early exit if clearly a new record with margin
            if best_val < 0.2720840 and time.time() > t0 + 40:
                pass
        if time.time() > EXP_END:
            break

    if time.time() < T_END - 7.0:
        secs = min(T_END - time.time() - 1.0, 28.0)
        x_best = _lp_polish(x_best, secs, T_END - 1.0)
    x_best = _norm(x_best)
    return x_best

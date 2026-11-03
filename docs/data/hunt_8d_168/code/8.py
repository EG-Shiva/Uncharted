import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

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
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
    n, d = 168, 8
    try:
        best_f = np.load("best.npy").astype(float)
    except Exception:
        best_f = None
    try:
        rec_f = np.load("record.npy").astype(float)
    except Exception:
        rec_f = None
    cands = []
    if best_f is not None and best_f.shape == (n, d):
        cands.append(best_f / np.linalg.norm(best_f, axis=1, keepdims=True))
    if rec_f is not None and rec_f.shape == (n, d):
        cands.append(rec_f / np.linalg.norm(rec_f, axis=1, keepdims=True))
    if not cands:
        c = rng.normal(size=(n, d)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)

    def soft_run(x_init, steps, a0, a1, lr0, lr1, mu, normed=True):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if s % 400 == 0 and time.time() - t0 > BUDGET - 38:
                break
            frac = s / max(steps - 1, 1)
            alpha = a0 + (a1 - a0) * frac
            if lr0 > 0 and lr1 > 0:
                lr = lr0 * (lr1 / lr0) ** frac
            else:
                lr = lr0
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            if normed:
                S = W.sum(axis=1, keepdims=True) + 1e-12
                A = (W @ x) / S
            else:
                A = (W @ x) / 30.0
            dot = np.sum(A * x, axis=1, keepdims=True)
            At = A - dot * x
            vel = mu * vel + At
            x = x - lr * vel
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        return x

    def refine_hole(Y, p, iters=250, beta=80.0, lr0=0.15, lr1=0.005):
        p = p / np.linalg.norm(p)
        for it in range(iters):
            frac = it / max(iters - 1, 1)
            lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
            g = Y @ p
            m = g.max()
            w = np.exp(beta * (g - m))
            s = w.sum() + 1e-12
            A = (w @ Y) / s
            At = A - float(A @ p) * p
            p = p - lr * At
            nr = np.linalg.norm(p)
            if nr == 0:
                break
            p /= nr
        return p

    def find_holes(Y, k, n_probe=24000):
        m = Y.shape[0]
        P = rng.normal(size=(n_probe, d))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        # chunk to save memory
        best_s = np.empty(n_probe)
        CH = 6000
        for a in range(0, n_probe, CH):
            b = min(a + CH, n_probe)
            D = P[a:b] @ Y.T
            best_s[a:b] = D.max(axis=1)
        order = np.argsort(best_s)  # smallest max-dot = biggest hole
        sel = []
        for idx in order:
            if len(sel) >= k:
                break
            p = P[idx]
            ok = True
            for q in sel:
                if abs(float(p @ q)) > 0.93:
                    ok = False
                    break
            if ok:
                sel.append(p)
            # if too strict, allow after many tries
            if len(sel) == 0 and idx > 5000:
                sel.append(p)
        # fallback fill if strict suppression removed too many
        if len(sel) < k:
            for idx in order:
                if len(sel) >= k:
                    break
                p = P[idx]
                dup = False
                for q in sel:
                    if np.allclose(p, q):
                        dup = True
                        break
                if not dup:
                    # allow closer now
                    sel.append(p)
                    if len(sel) >= k:
                        break
        holes = []
        for p in sel[:k]:
            holes.append(refine_hole(Y, p))
        return np.array(holes)

    def contact_counts(X, thr=0.004):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        return (G > m - thr).sum(axis=1), m

    trial = 0
    # quick diverse seeds
    for _ in range(2):
        if time.time() - t0 > BUDGET - 45:
            break
        x0 = rng.normal(size=(n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        xopt = soft_run(x0, 2200, 35.0, 170.0, 0.07, 0.005, 0.85, normed=True)
        v = _max_cos(xopt)
        if v < best_val:
            best_val = v; best_x = xopt
        trial += 1

    while time.time() - t0 < BUDGET - 38:
        trial += 1
        r = rng.random()
        normed = bool(rng.random() < 0.7)
        if r < 0.18:
            # global jitter
            scale = 10.0 ** float(rng.uniform(-2.4, -0.7))
            x0 = best_x + rng.normal(size=(n, d)) * scale
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1200, 2800))
            a0 = float(rng.uniform(30, 60)); a1 = float(rng.uniform(110, 210))
            lr0 = float(rng.uniform(0.03, 0.08)); lr1 = float(rng.uniform(0.002, 0.009))
            mu = float(rng.uniform(0.7, 0.9))
            xopt = soft_run(x0, steps, a0, a1, lr0, lr1, mu, normed=normed)
        elif r < 0.55:
            # delete worst-k and hole-fill (topology change)
            cnt, m = contact_counts(best_x, thr=0.004)
            k = int(rng.integers(3, 13))
            # mostly worst, sometimes random
            if rng.random() < 0.75:
                order = np.argsort(-cnt)
                pool = order[:max(25, 2 * k)]
                sel = rng.choice(pool, size=k, replace=False)
            else:
                sel = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[sel] = False
            Y = best_x[keep]
            try:
                holes = find_holes(Y, k)
            except Exception:
                holes = rng.normal(size=(k, d))
                holes /= np.linalg.norm(holes, axis=1, keepdims=True)
            x0 = np.empty((n, d))
            x0[keep] = Y
            x0[~keep] = holes
            x0 += rng.normal(size=(n, d)) * 0.004
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1500, 3200))
            a0 = float(rng.uniform(30, 60)); a1 = float(rng.uniform(120, 230))
            lr0 = float(rng.uniform(0.04, 0.09)); lr1 = float(rng.uniform(0.002, 0.008))
            mu = float(rng.uniform(0.75, 0.9))
            xopt = soft_run(x0, steps, a0, a1, lr0, lr1, mu, normed=normed)
        elif r < 0.78:
            # random subset re-init then hole-guided repair
            k = int(rng.integers(4, 16))
            sel = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[sel] = False
            Y = best_x[keep]
            # half holes, half random
            kh = max(1, k // 2)
            try:
                holes = find_holes(Y, kh)
            except Exception:
                holes = rng.normal(size=(kh, d))
                holes /= np.linalg.norm(holes, axis=1, keepdims=True)
            x0 = best_x.copy()
            x0[sel[:kh]] = holes
            xr = rng.normal(size=(k - kh, d)); xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            if k - kh > 0:
                x0[sel[kh:]] = xr
            x0 += rng.normal(size=(n, d)) * 0.005
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1500, 3200))
            a0 = float(rng.uniform(30, 60)); a1 = float(rng.uniform(120, 220))
            lr0 = float(rng.uniform(0.04, 0.09)); lr1 = float(rng.uniform(0.002, 0.008))
            mu = float(rng.uniform(0.75, 0.9))
            xopt = soft_run(x0, steps, a0, a1, lr0, lr1, mu, normed=normed)
        else:
            # heavy kick on overloaded + jitter
            cnt, m = contact_counts(best_x, thr=0.004)
            order = np.argsort(-cnt)
            pool = order[:30]
            k = int(rng.integers(4, 12))
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            x0 = best_x.copy()
            s_big = float(rng.uniform(0.15, 0.55))
            x0[sel] += rng.normal(size=(len(sel), d)) * s_big
            x0 += rng.normal(size=(n, d)) * 0.008
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1400, 3000))
            a0 = float(rng.uniform(30, 70)); a1 = float(rng.uniform(120, 230))
            lr0 = float(rng.uniform(0.03, 0.07)); lr1 = float(rng.uniform(0.002, 0.007))
            mu = float(rng.uniform(0.75, 0.9))
            xopt = soft_run(x0, steps, a0, a1, lr0, lr1, mu, normed=normed)
        v = _max_cos(xopt)
        if v < best_val - 1e-12:
            best_val = v
            best_x = xopt
            if time.time() - t0 < BUDGET - 50:
                xopt2 = soft_run(best_x, 1000, 90.0, 230.0, 0.018, 0.002, 0.8, normed=True)
                v2 = _max_cos(xopt2)
                if v2 < best_val:
                    best_val = v2
                    best_x = xopt2
        if time.time() - t0 > BUDGET - 38:
            break

    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5:
        best_x = _lp_polish(best_x, min(remain, 36.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

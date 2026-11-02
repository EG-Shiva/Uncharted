import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_squeeze(x, end_time, r0=0.004, cap=1200):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    itn = 0
    while time.time() < end_time and radius > 1e-13 and itn < 60:
        itn += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            order = np.argsort(-c[sel])[:cap]
            I = I[order]; J = J[order]; p = cap
        cb = (x[I] * x[J]).sum(axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq,
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.03)
        else:
            if v < m + 1e-12:
                x = y
            radius *= 0.5
    return xb, best

def _adam_push(x, target, t0_, t1_, steps, lr0):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        temp = t0_ * (t1_ / t0_) ** frac
        lr = lr0 * (0.35 + 0.65 * (1.0 - frac))
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        D = (G - target) / temp
        dm = float(D.max())
        W = np.exp(D - dm)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / sn)
        x = x - step * scale
        x = _norm_rows(x)
        if not np.all(np.isfinite(x)):
            break
    return x

def _find_hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.randn(n_cand, d)
    cand = _norm_rows(cand)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def solve():
    t_start = time.time()
    DEAD = t_start + 168.0
    SEARCH_END = t_start + 150.0
    rng = np.random.RandomState(7)
    n, d = 55, 21
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=float))
        except Exception:
            pass
    best_val = 1e9
    x_best = None
    for c in cands:
        try:
            if c.shape != (55, 21):
                continue
            c = _norm_rows(c)
            v = _max_cos(c)
            if v < best_val:
                best_val = v; x_best = c.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(x_best)
    if time.time() < DEAD - 10:
        xb, vb = _lp_squeeze(x_best, min(time.time() + 1.5, DEAD - 8), r0=8e-4)
        if vb < best_val:
            x_best, best_val = xb, vb
    base = x_best.copy()
    it = 0
    while time.time() < SEARCH_END:
        it += 1
        if it % 25 == 0:
            base = x_best.copy()
        cur = base.copy()
        mode = it % 7
        if mode in (0, 1, 4):
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            cnt = (G > mx - 1e-3).sum(axis=1)
            if mode == 0: k = 1
            elif mode == 1: k = 2
            else: k = 3
            victims = np.argsort(-cnt)[:k]
            r = rng.rand()
            if r < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            elif r < 0.40:
                # target loose points (few contacts) for diversity
                victims = np.argsort(cnt)[:k]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = cur[keep]
            for vi in victims:
                h = _find_hole(Y, rng, n_cand=3000)
                cur[vi] = h
                Y = np.vstack([Y, h[None, :]])
            sigma = 0.008 + rng.rand() * 0.015
            cur = _norm_rows(cur + sigma * rng.randn(n, d))
        elif mode == 2:
            sigma = 0.02 + rng.rand() * 0.05
            cur = _norm_rows(cur + sigma * rng.randn(n, d))
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            thr = float(G.max()) - 2e-3
            cnt = (G > thr).sum(axis=1)
            ov = np.argsort(-cnt)[:6]
            cur[ov] += (sigma * 1.2) * rng.randn(len(ov), d)
            cur = _norm_rows(cur)
        elif mode == 3:
            sigma = 0.004 + rng.rand() * 0.010
            cur = _norm_rows(cur + sigma * rng.randn(n, d))
        elif mode == 5:
            sigma = 0.07 + rng.rand() * 0.07
            cur = _norm_rows(cur + sigma * rng.randn(n, d))
        else:
            # subset shake: pick random 8-14 points and reshuffle locally
            k = 8 + int(rng.rand() * 7)
            idx = rng.choice(n, size=k, replace=False)
            cur[idx] = _norm_rows(cur[idx] + 0.15 * rng.randn(k, d))
        delta = 2e-4 + rng.rand() * 7e-4
        target = best_val - delta
        lt0 = rng.uniform(-2.2, -1.6)
        lt1 = rng.uniform(-3.2, -2.5)
        t0_ = 10.0 ** lt0; t1_ = 10.0 ** lt1
        if t1_ > t0_:
            t0_, t1_ = t1_, t0_
        steps = 300 + int(rng.rand() * 400)
        lr0 = 0.005 + rng.rand() * 0.011
        cur = _adam_push(cur, target, t0_, t1_, steps, lr0)
        v_pre = _max_cos(cur)
        if v_pre > best_val + 0.004 and (it % 5 != 0):
            if rng.rand() < 0.04:
                base = cur.copy()
            continue
        if SEARCH_END - time.time() < 0.6:
            break
        secs = 0.6
        if v_pre < best_val + 0.0005:
            secs = 0.9
        if v_pre < best_val:
            secs = 1.3
        end = min(time.time() + secs, SEARCH_END)
        polished, v = _lp_squeeze(cur, end, r0=0.005)
        if v < best_val - 1e-12:
            best_val = v; x_best = polished.copy(); base = polished.copy()
            if time.time() < SEARCH_END - 2.0 and (v_pre - v) > 1e-6:
                end2 = min(time.time() + 0.7, SEARCH_END)
                sq, vs = _lp_squeeze(x_best, end2, r0=8e-4)
                if vs < best_val:
                    best_val = vs; x_best = sq.copy(); base = sq.copy()
        else:
            if v < _max_cos(base) and rng.rand() < 0.30:
                base = polished.copy()
        if it > 350:
            break
    rem = DEAD - time.time()
    if rem > 1.2:
        end = time.time() + min(rem - 0.5, 12.0)
        sq, vs = _lp_squeeze(x_best, end, r0=8e-4)
        if vs < best_val:
            x_best = sq
    x_best = _norm_rows(x_best)
    x_best[~np.isfinite(x_best)] = 0.01
    x_best = _norm_rows(x_best)
    return x_best

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

def _lp_polish(x, deadline, r0=0.005, max_it=60):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = float(r0)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1200:
            o = np.argsort(-c[sel])[:1200]
            I = I[o]; J = J[o]
            cb = c[sel][o]
            p = 1200
        else:
            cb = c[sel]
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
        if time.time() >= deadline:
            break
    return xb, best

def _adam_quench(x, steps, t0, t1, lr0):
    n, d = x.shape
    m = np.zeros_like(x); vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        temp = t0 * (t1 / t0) ** frac
        lr = lr0 * (0.35 + 0.65 * (1.0 - frac))
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        D = (G - 0.066) / temp
        # shift for stability using row-wise? global max
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
        sc = np.minimum(1.0, 0.05 / sn)
        x = x - step * sc
        x = _norm_rows(x)
    return x

def _find_hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.randn(n_cand, d)
    cand = _norm_rows(cand)
    md = (cand @ Y.T).max(axis=1)
    return cand[np.argmin(md)]

def solve():
    t0t = time.time()
    DEAD = t0t + 168.0
    SEARCH_END = t0t + 150.0
    rng = np.random.RandomState(20261101)
    n, d = 55, 21
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(best)
    base = best.copy()
    it = 0
    while time.time() < SEARCH_END:
        it += 1
        if it % 25 == 0:
            base = best.copy()
        cur = base.copy()
        mode = it % 6
        if mode in (0, 1, 4):
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            cnt = (G > mx - 1e-3).sum(axis=1)
            k = 1 if mode == 0 else (2 if mode == 1 else 3)
            victims = np.argsort(-cnt)[:k]
            if rng.rand() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
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
            cur[ov] += sigma * rng.randn(len(ov), d)
            cur = _norm_rows(cur)
        elif mode == 3:
            sigma = 0.005 + rng.rand() * 0.012
            cur = _norm_rows(cur + sigma * rng.randn(n, d))
        else:
            sigma = 0.08 + rng.rand() * 0.08
            cur = _norm_rows(cur + sigma * rng.randn(n, d))
        lt0 = rng.uniform(-2.3, -1.6)
        lt1 = rng.uniform(-3.3, -2.5)
        ta0 = 10.0 ** lt0; ta1 = 10.0 ** lt1
        if ta1 > ta0:
            ta0, ta1 = ta1, ta0
        steps = 350 + int(rng.rand() * 400)
        lr0 = 0.005 + rng.rand() * 0.012
        cur = _adam_quench(cur, steps, ta0, ta1, lr0)
        v_pre = _max_cos(cur)
        # only polish near-misses to save time
        thresh = best_val + 0.002
        if v_pre < best_val:
            thresh = best_val + 0.0005
        if v_pre > thresh and (it % 4 != 0):
            if rng.rand() < 0.03:
                base = cur.copy()
            continue
        if time.time() > SEARCH_END - 0.6:
            break
        secs = 0.6
        if v_pre < best_val + 0.0005:
            secs = 0.9
        if v_pre < best_val:
            secs = 1.2
        end = min(time.time() + secs, SEARCH_END)
        pol, v = _lp_polish(cur, end, r0=0.005)
        if v < best_val - 1e-12:
            best_val = v; best = pol.copy(); base = pol.copy()
        else:
            if v < _max_cos(base) and rng.rand() < 0.25:
                base = pol.copy()
        if it > 500:
            break
    rem = DEAD - time.time()
    if rem > 1.5 and best is not None:
        end = time.time() + min(rem - 0.5, 12.0)
        sq, vs = _lp_polish(best, end, r0=1e-3, max_it=80)
        if vs < best_val:
            best = sq; best_val = vs
    best = _norm_rows(best)
    best[~np.isfinite(best)] = 0.01
    best = _norm_rows(best)
    return best

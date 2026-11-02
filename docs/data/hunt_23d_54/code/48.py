import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        sel = D > -w
        W = np.zeros_like(G)
        # exp in [-k*w, 0]
        W[sel] = np.exp(k * D[sel])
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _void_best(kept, rng, n_try=2000):
    d = kept.shape[1]
    C = rng.standard_normal((n_try, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        c = c_all[sel]
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1200:
            # keep tightest 1200 to keep LP fast
            idx = np.argsort(-c)[:1200]
            c = c[idx]; I = I[idx]; J = J[idx]; p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_blocks = np.empty((p, 2 * d + 1), dtype=int)
        cols_blocks[:, :d] = I[:, None] * d + ar
        cols_blocks[:, d:2*d] = J[:, None] * d + ar
        cols_blocks[:, 2*d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1))
        vals_blocks[:, :d] = x[J]
        vals_blocks[:, d:2*d] = x[I]
        vals_blocks[:, 2*d] = -1.0
        vals = vals_blocks.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    HARD = 176.0
    FINAL_RESERVE = 22.0
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        try:
            r = min(xb.size, tmp.size)
            tmp.ravel()[:r] = xb.ravel()[:r]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    # quick PCA outlier info cache updated periodically
    it = 0
    tol0 = 0.004
    while time.time() - t0 < HARD - FINAL_RESERVE:
        it += 1
        remain = HARD - FINAL_RESERVE - (time.time() - t0)
        if remain < 2.0:
            break
        steps = 500
        if remain < 15:
            steps = 300
        elif remain < 40:
            steps = 400
        mode = it % 6
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        ppm = G.max(axis=1)
        wide = G > (best - 0.015)
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)
        x = x_best.copy()
        if mode == 0:
            sig = float(rng.uniform(0.006, 0.014))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=80, k1=2500, lr0=0.012, lr1=0.0008, w0=0.05, w1=0.005)
        elif mode == 1:
            sig = float(rng.uniform(0.015, 0.035))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2000, lr0=0.015, lr1=0.001, w0=0.06, w1=0.006)
        elif mode == 2:
            ksel = int(rng.integers(1, 4))
            idx = order_worst[:ksel]
            keep = np.ones(n, dtype=bool); keep[idx] = False
            fixed = x[keep]
            for j in idx:
                nv = _void_best(fixed, rng, n_try=1500)
                x[j] = nv + rng.standard_normal(d) * 0.015
                x[j] /= np.linalg.norm(x[j])
                fixed = np.vstack([fixed, x[j][None, :]])
            x = x + rng.standard_normal((n, d)) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=70, k1=2800, lr0=0.014, lr1=0.0008, w0=0.055, w1=0.005)
        elif mode == 3:
            ksel = int(rng.integers(2, 5))
            idx = order_lo[:ksel]
            keep = np.ones(n, dtype=bool); keep[idx] = False
            fixed = x[keep]
            for j in idx:
                nv = _void_best(fixed, rng, n_try=1500)
                x[j] = nv + rng.standard_normal(d) * 0.02
                x[j] /= np.linalg.norm(x[j])
                fixed = np.vstack([fixed, x[j][None, :]])
            x = _anneal(x, steps=steps, k0=70, k1=2500, lr0=0.014, lr1=0.0008, w0=0.055, w1=0.005)
        elif mode == 4:
            # PCA outlier targeted move
            try:
                xc = x_best - x_best.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(xc, full_matrices=False)
                P = x_best @ Vt[:2].T  # (n,2)
                score = np.linalg.norm(P, axis=1)
                ksel = int(rng.integers(2, 5))
                idx = np.argsort(-score)[:ksel]
                sig = float(rng.uniform(0.03, 0.12))
                for j in idx:
                    if rng.random() < 0.5:
                        # reinsert to void
                        mask = np.ones(n, dtype=bool); mask[j] = False
                        nv = _void_best(x[mask], rng, n_try=1200)
                        x[j] = nv
                    else:
                        x[j] = x[j] + sig * rng.standard_normal(d)
                        x[j] /= np.linalg.norm(x[j])
            except Exception:
                x = x + 0.02 * rng.standard_normal((n, d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2200, lr0=0.013, lr1=0.0009, w0=0.06, w1=0.0055)
        else:
            ksel = int(rng.integers(1, 4))
            idx = rng.choice(n, size=ksel, replace=False)
            R = rng.standard_normal((ksel, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            x[idx] = R
            x = _anneal(x, steps=steps, k0=50, k1=1800, lr0=0.016, lr1=0.001, w0=0.07, w1=0.006)
        v = _max_cos(x)
        # acceptance: strict improvement or small sideways early
        frac = (time.time() - t0) / max(HARD - FINAL_RESERVE, 1.0)
        tol = tol0 * (1.0 - frac)  # decays to 0
        if v < best - 1e-12:
            best = v
            x_best = x.copy()
            # confirm/refine with short LP if time allows (rare)
            if HARD - (time.time() - t0) > FINAL_RESERVE + 6:
                try:
                    y = _lp_polish(x_best, 1.5, deadline=t0 + HARD - FINAL_RESERVE)
                    vy = _max_cos(y)
                    if vy < best:
                        best = vy
                        x_best = y
                except Exception:
                    pass
        elif v < best + tol and rng.random() < 0.15:
            # sideways drift to different basin: move base without updating best
            # keep hopping from this point for a while by temporary replacement?
            # implement as occasional base shift if not too bad
            pass
        # loop guard
    # final polish of best
    rem = HARD - (time.time() - t0) - 0.5
    if rem > 2:
        rem = min(rem, FINAL_RESERVE)
        try:
            x_best = _lp_polish(x_best, rem)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

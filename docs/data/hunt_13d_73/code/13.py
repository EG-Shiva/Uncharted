import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
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
        m0 = min(xb.shape[0], n)
        m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()
    X_cur = xb.copy()
    cur_val = best_val

    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    def smooth_adam(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        W = np.empty((n, n))
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            e = np.exp(s * (c - cmax))
            ss = e.sum()
            if not np.isfinite(ss) or ss == 0:
                w = np.zeros_like(e)
                w[c >= cmax - 1e-12] = 1.0
                w /= w.sum()
            else:
                w = e / ss
            W.fill(0.0)
            W[I0, J0] = w
            W[J0, I0] = w
            grad = W @ X
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 *= b1; m1 += (1 - b1) * gt
            v1 *= b2; v1 += (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
            b1p *= b1; b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X -= lr * step
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if time.time() >= DEADLINE - 32.0:
                break
        return X

    def find_hole(X_keep, n_cand=7000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ X_keep.T
        md = dots.max(axis=1)
        return C[int(np.argmin(md))]

    sigmas = [0.008, 0.02, 0.045, 0.09, 0.18, 0.32]
    it = 0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 34.0:
        it += 1
        use_best = rng.random() < 0.70
        base = X_best if use_best else X_cur
        sigma = sigmas[it % len(sigmas)]
        mode = it % 7
        try:
            if mode in (0, 1, 5):
                # global Gaussian kick + smooth
                Nt = rng.standard_normal((n, d))
                X_try = base + sigma * Nt
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                # vary sharpness by iteration
                if mode == 5:
                    X_try = smooth_adam(X_try, 550, 40.0, 500.0, 0.008, 0.0004)
                else:
                    X_try = smooth_adam(X_try, 650, 30.0, 400.0, 0.01, 0.0005)
            elif mode == 2:
                # coherent cap shake: pick random cap, shove it
                u = rng.standard_normal(d); u /= np.linalg.norm(u)
                dots_u = base @ u
                thr = np.quantile(dots_u, 0.75)
                mask = dots_u >= thr
                if mask.sum() < 3:
                    mask = np.zeros(n, dtype=bool); mask[np.argsort(-dots_u)[:8]] = True
                Nt = rng.standard_normal((n, d)) * 0.3
                shove = rng.standard_normal(d); shove /= np.linalg.norm(shove)
                Nt[mask] += (sigma * 3.0) * shove[None, :]
                Nt[~mask] += (sigma * 0.4) * rng.standard_normal((int((~mask).sum()), d))
                X_try = base + Nt
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                X_try = smooth_adam(X_try, 600, 35.0, 450.0, 0.009, 0.0004)
            else:
                # kill worst 1-3 and reinsert into holes
                k = 1 if mode == 3 else (2 if mode == 4 else 3)
                if it % 25 == 0:
                    k = 3
                rm = _row_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                if rng.random() < 0.35:
                    victims[0] = int(rng.integers(0, n))
                X_try = base.copy()
                keep = np.ones(n, dtype=bool); keep[victims] = False
                X_keep = X_try[keep]
                for v in victims:
                    h = find_hole(X_keep, n_cand=7000)
                    # small jitter
                    h = h + 1e-3 * rng.standard_normal(d)
                    h /= np.linalg.norm(h)
                    X_try[v] = h
                    X_keep = X_try[np.ones(n, dtype=bool)] if False else np.delete(X_try, v, axis=0) if False else X_try[keep | (np.arange(n) == v)]
                    # update keep set for next victim hole search
                    X_keep = X_try[np.array([i not in victims[victims.tolist().index(int(v))+1:] if i != v else True for i in range(n)])] if k > 1 else X_try[keep | (np.arange(n) == v)]
                    # simpler: recompute hole basis as all except remaining victims not yet placed
                    # just use current X_try excluding not-yet-placed is complex; use all except v
                    # fallback: use all points except v
                    others = np.delete(X_try, int(v), axis=0)
                    # refine once: already have h, keep it
                    _ = others
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                X_try = smooth_adam(X_try, 500, 30.0, 350.0, 0.009, 0.0005)
        except Exception:
            continue
        try:
            v_try = _max_cos(X_try)
        except Exception:
            continue
        if v_try < best_val - 1e-15:
            best_val = v_try
            X_best = X_try.copy()
            X_cur = X_try.copy()
            cur_val = v_try
        elif v_try < cur_val:
            X_cur = X_try.copy()
            cur_val = v_try
        elif rng.random() < 0.06 and v_try < cur_val + 0.015:
            X_cur = X_try.copy()
            cur_val = v_try
        # occasional restart from best if wandering too far
        if cur_val > best_val + 0.02:
            X_cur = X_best.copy()
            cur_val = best_val

    # final polish of best
    try:
        remain = DEADLINE - time.time()
        if remain > 8.0:
            X_best = _lp_polish(X_best, min(remain - 1.0, 32.0), t_end=DEADLINE - 1.0)
            bv = _max_cos(X_best)
            best_val = bv
    except Exception:
        pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # tiny shrink to enforce strict feasibility (no-op for score, guards NaN)
    X_best = np.where(np.isfinite(X_best), X_best, rng.standard_normal((n, d)))
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

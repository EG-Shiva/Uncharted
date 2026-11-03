import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, r0=2e-3, rmax=0.05):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    while time.time() - t0 < seconds and radius > 1e-14:
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cf, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).ravel(),
                               (J[:, None] * d + np.arange(d)).ravel(),
                               np.full(p, n * d)]).reshape(p, 2 * d + 1).ravel() if False else None
        # build cols directly
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best - 1e-15:
            X = Y; best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
        if time.time() - t0 >= seconds:
            break
    return X

def solve():
    t_start = time.time()
    BUDGET = 174.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (162, 8):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (162, 8):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        r0 = np.random.default_rng(0)
        a = r0.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    X_best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    n, d = 162, 8
    rng = np.random.default_rng(20261103)

    SHIFT = 0.5

    def eg_fn(Yf, alpha):
        Y = Yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = alpha * (G - SHIFT)
        np.clip(M, -60.0, 20.0, out=M)
        Emat = np.exp(M)
        E = float(Emat.sum() * 0.5)
        GradX = alpha * (Emat @ X)
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradY = (GradX - dot * X) / nr
        return E, GradY.ravel()

    def staged_opt(X_init, alphas, maxiters):
        Yf = X_init.ravel().copy()
        for alpha, mi in zip(alphas, maxiters):
            if time.time() - t_start > BUDGET - 35:
                break
            try:
                res = minimize(lambda y: eg_fn(y, alpha), Yf, jac=True,
                               method="L-BFGS-B",
                               options={"maxiter": mi, "ftol": 1e-12, "gtol": 1e-5, "maxfun": mi * 20})
                Yf = res.x
            except Exception:
                break
        Y = Yf.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def refine_single(y0, X_keep, alpha=90.0, steps=300, lr0=0.08):
        y = y0 / np.linalg.norm(y0)
        for it in range(steps):
            s = X_keep @ y
            # weights focused on nearest
            m = float(s.max())
            w = np.exp(alpha * (s - m))
            sw = w.sum()
            if not np.isfinite(sw) or sw <= 0:
                break
            g = (w[:, None] * X_keep).sum(axis=0) / sw
            # tangent projection (descent on max dot)
            gt = g - float(g @ y) * y
            ng = float(np.linalg.norm(gt))
            if ng < 1e-14:
                break
            lr = lr0 * (1.0 - 0.7 * it / steps) / (1.0 + ng * 5.0)
            y = y - lr * (gt / ng + 0.3 * gt)
            y /= np.linalg.norm(y)
        return y

    def deep_reinsert(Xcur, idx_list, n_samp=22000, alpha_ref=90.0):
        Xnew = Xcur.copy()
        for w in idx_list:
            keep = np.delete(Xnew, w, axis=0)
            C = rng.standard_normal(size=(n_samp, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ keep.T
            scores = S.max(axis=1)
            b = int(np.argmin(scores))
            y0 = C[b]
            # try a few of the best as starts, keep best refined
            order = np.argpartition(scores, 8)[:8]
            yb = y0; sb = float(scores[b])
            # refine top 3 starts cheaply and pick best true score after refine
            best_y = refine_single(y0, keep, alpha=alpha_ref)
            best_s = float((keep @ best_y).max())
            for oi in order[1:4]:
                yc = refine_single(C[oi], keep, alpha=alpha_ref, steps=150)
                sc = float((keep @ yc).max())
                if sc < best_s:
                    best_s = sc; best_y = yc
            # enforce separation among already inserted (already in keep)
            Xnew[w] = best_y
        return Xnew

    def per_stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        thr = float(G.max()) - 0.006
        cnt = (G > thr).sum(axis=1)
        return G, pm, cnt

    trial = 0
    no_improve = 0
    while time.time() - t_start < 132:
        trial += 1
        if time.time() - t_start > BUDGET - 42:
            break
        mode = trial % 6
        X_try = X_best.copy()
        if mode == 0:
            # single worst reinsert
            _, pm, _ = per_stats(X_best)
            w = [int(np.argmax(pm))]
            # occasionally also rattler
            if trial % 12 == 0:
                G, _, cnt = per_stats(X_best)
                w.append(int(np.argmin(cnt)))
            X_try = deep_reinsert(X_try, w)
            alphas, mis = [60, 140, 220], [60, 60, 50]
        elif mode == 1:
            # multi overload + random
            _, pm, _ = per_stats(X_best)
            order = np.argsort(-pm)
            k = int(rng.integers(4, 8))
            # 70% worst, 30% random
            nw = max(2, k - 2)
            idx = list(order[:nw]) + list(rng.choice(n, k - nw, replace=False))
            # de-dup
            idx = list(dict.fromkeys([int(v) for v in idx]))[:k]
            X_try = deep_reinsert(X_try, idx)
            alphas, mis = [25, 70, 150, 220], [70, 70, 60, 40]
        elif mode == 2:
            sigma = 0.012 if no_improve < 10 else 0.03
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            alphas, mis = [80, 170], [55, 55]
        elif mode == 3:
            k = int(rng.integers(8, 16))
            idx = rng.choice(n, k, replace=False)
            X_try[idx] += 0.16 * rng.standard_normal(size=(k, d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            alphas, mis = [20, 55, 130, 210], [80, 80, 60, 40]
        elif mode == 4:
            G, _, cnt = per_stats(X_best)
            low = list(np.argsort(cnt)[:3])
            # include a high-contact point to relieve jam
            high = [int(np.argsort(cnt)[-1])]
            idx = low + high
            X_try = deep_reinsert(X_try, idx)
            alphas, mis = [30, 90, 180], [70, 70, 60]
        else:
            sigma = 0.07 + 0.03 * rng.random()
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            alphas, mis = [25, 80, 160], [70, 70, 60]

        X_opt = staged_opt(X_try, alphas, mis)
        v = _max_cos(X_opt)
        if v < best_val - 1e-12:
            X_best = X_opt; best_val = v
            no_improve = 0
            if time.time() - t_start < 105:
                X_lp = _lp_polish(X_best, 2.0)
                v2 = _max_cos(X_lp)
                if v2 < best_val:
                    X_best, best_val = X_lp, v2
                elif v2 < best_val + 1e-12:
                    pass
        else:
            no_improve += 1

    remain = BUDGET - (time.time() - t_start) - 2.0
    if remain > 5:
        # use larger initial radius if still unbalanced
        X_best = _lp_polish(X_best, remain, r0=4e-3)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best

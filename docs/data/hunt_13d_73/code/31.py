import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (X @ X.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_cand(c0, S, alpha=220.0, steps=70, lr=0.08):
    c = c0.copy()
    for _ in range(steps):
        d = S @ c
        m = d.max()
        e = np.exp(alpha * (d - m))
        s = e.sum()
        if s == 0 or not np.isfinite(s):
            break
        w = e / s
        g = w @ S
        gt = g - (g @ c) * c
        c = c - lr * gt
        n = np.linalg.norm(c)
        if n == 0 or not np.isfinite(n):
            break
        c /= n
    return c

def _block_opt(B0, F, maxiter=120, alpha=240.0):
    k, d = B0.shape
    m = F.shape[0]
    iu = np.triu_indices(k, 1) if k > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    Y0 = B0.ravel().copy()
    def fun(Yf):
        Ym = Yf.reshape(k, d)
        rn = np.linalg.norm(Ym, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Ym / rn
        Dbf = X @ F.T
        if k > 1:
            Dbb = (X @ X.T)[iu]
            dots = np.concatenate([Dbf.ravel(), Dbb])
        else:
            dots = Dbf.ravel()
        M = dots.max()
        e = np.exp(alpha * (dots - M))
        S = e.sum()
        loss = M + np.log(S) / alpha
        p = e / S
        nbf = Dbf.size
        pbf = p[:nbf].reshape(k, m)
        gX = pbf @ F
        if k > 1:
            pbb = p[nbf:]
            W = np.zeros((k, k))
            W[iu] = pbb
            W = W + W.T
            gX = gX + W @ X
        dg = (gX * X).sum(axis=1, keepdims=True)
        gXt = gX - dg * X
        gY = (gXt / rn).ravel()
        return float(loss), gY.astype(float)
    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 8})
        Ym = res.x.reshape(k, d)
    except Exception:
        Ym = Y0.reshape(k, d)
    Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
    return Ym

def _relax1s(X0, time_budget=1.0, t_end=1e30):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    steps = 220
    s0, s1 = 90.0, 280.0
    lr0, lr1 = 0.02, 0.004
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    start = time.time()
    for k in range(steps):
        if time.time() - start > time_budget or time.time() > t_end:
            break
        frac = k / (steps - 1)
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        sumE = E.sum() * 0.5
        if sumE == 0 or not np.isfinite(sumE):
            break
        W = E / sumE
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1; m1 += (1 - b1) * gt
        v1 *= b2; v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
        b1p *= b1; b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X -= lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    LOOP_END = t0 + 150.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(n, d))
    if xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    cap_sizes = [10, 12, 14, 16, 11, 13, 15, 10, 16, 12]
    it = 0
    NB = 80000
    while time.time() < LOOP_END and time.time() < HARD_END - 8.0:
        it += 1
        if time.time() > HARD_END - 8.0:
            break
        # analyze incumbent for center choice
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        rowmax = G.max(axis=1)
        counts = (G >= cmax - 1e-7).sum(axis=1)
        idx_rattler = int(np.argmin(counts))
        idx_hot = int(np.argmax(counts))
        idx_rand = int(rng.integers(0, n))
        # PCA outlier on top-2 axes
        try:
            _, _, Vh = np.linalg.svd(Xbest - Xbest.mean(axis=0, keepdims=True), full_matrices=False)
            P = Xbest @ Vh[:2].T
            Pc = P - P.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(Pc, axis=1)
            idx_pca = int(np.argmax(dists))
        except Exception:
            idx_pca = int(rng.integers(0, n))
        mode = it % 4
        if mode == 0:
            cidx = idx_rattler
        elif mode == 1:
            cidx = idx_rand
        elif mode == 2:
            cidx = idx_hot
        else:
            cidx = idx_pca
        center = Xbest[cidx].copy()
        k = int(cap_sizes[(it - 1) % len(cap_sizes)])
        dots_c = Xbest @ center
        order = np.argsort(-dots_c)
        del_idx = order[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        F = Xbest[keep_mask].copy()
        # greedy rebuild
        S = F.copy()
        Bnew = []
        ok = True
        for v in range(k):
            if time.time() > HARD_END - 8.0 or time.time() > LOOP_END:
                ok = False
                break
            C = rng.normal(size=(NB, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            dots = C @ S.T
            md = dots.max(axis=1)
            # best few
            nb = 5
            part = np.argpartition(md, nb - 1)[:nb]
            best_c = None
            best_s = 1e30
            for bi in part:
                c0 = C[int(bi)]
                cr = _refine_cand(c0, S)
                s_val = float((S @ cr).max())
                if s_val < best_s:
                    best_s = s_val
                    best_c = cr
            if best_c is None:
                best_c = C[int(part[0])]
            Bnew.append(best_c)
            S = np.vstack([S, best_c[None, :]])
        if not ok or len(Bnew) != k:
            continue
        B0 = np.array(Bnew)
        # block optimization (frozen obstacles)
        if time.time() < HARD_END - 6.0:
            try:
                Bopt = _block_opt(B0, F, maxiter=100)
            except Exception:
                Bopt = B0
        else:
            Bopt = B0
        Xtry = Xbest.copy()
        Xtry[del_idx] = Bopt
        # 1-second smooth relaxation (unfrozen)
        remain = HARD_END - time.time()
        if remain < 2.0:
            break
        budg = min(1.0, remain - 1.0)
        if budg > 0.15:
            try:
                Xtry = _relax1s(Xtry, time_budget=budg, t_end=HARD_END - 7.0)
            except Exception:
                pass
        cand = _max_cos(Xtry)
        # LP polish only if within 0.001 or better
        if cand <= best_val + 0.001:
            t_left = HARD_END - time.time()
            if t_left > 7.5:
                try:
                    Xtry2 = _lp_polish(Xtry, seconds=min(6.0, t_left - 1.0), t_end=HARD_END - 0.5)
                    cand2 = _max_cos(Xtry2)
                except Exception:
                    cand2 = cand
                else:
                    if cand2 < cand:
                        Xtry, cand = Xtry2, cand2
        if cand < best_val:
            Xbest = Xtry / np.linalg.norm(Xtry, axis=1, keepdims=True)
            best_val = float(cand)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

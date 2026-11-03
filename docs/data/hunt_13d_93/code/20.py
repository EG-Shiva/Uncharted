import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _smooth(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T1 != T0 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum(axis=1, keepdims=True) + 1e-12
        Gr = (E @ X) / S
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Ft = Gr - dot * X
        V = mu * V + (1.0 - mu) * Ft
        dotv = (V * X).sum(axis=1, keepdims=True)
        Vt = V - dotv * X
        X = X - lr * Vt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hard(X, iters, step0, K):
    best = _maxcos(X)
    step = step0
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    m = len(iu0)
    K = int(min(K, m))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < m:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(m)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        G2 = Xt @ Xt.T
        np.fill_diagonal(G2, -2.0)
        v = float(G2.max())
        if v < best:
            X = Xt
            best = v
            step = min(step * 1.3, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def _pair_push(X, iters, eps0=0.002):
    best = _maxcos(X)
    eps = eps0
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        ii = k // n
        jj = k % n
        if ii == jj:
            eps *= 0.7
            continue
        xi = X[ii].copy()
        xj = X[jj].copy()
        di = xj - float(xj @ xi) * xi
        dj = xi - float(xi @ xj) * xj
        ni = float(np.linalg.norm(di))
        nj = float(np.linalg.norm(dj))
        if ni < 1e-12 or nj < 1e-12:
            eps *= 0.5
            continue
        di /= ni
        dj /= nj
        Y = X.copy()
        Y[ii] = xi - eps * di
        Y[jj] = xj - eps * dj
        Y[ii] /= np.linalg.norm(Y[ii])
        Y[jj] /= np.linalg.norm(Y[jj])
        G2 = Y @ Y.T
        np.fill_diagonal(G2, -2.0)
        v = float(G2.max())
        if v < best:
            X = Y
            best = v
            eps *= 1.05
        else:
            eps *= 0.7
            if eps < 1e-12:
                eps = eps0 * 0.1
    return X, best

def _find_void(others, rng, n_cand=8000):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    worst = D.max(axis=1)
    v = C[int(np.argmin(worst))].copy()
    lr = 0.05
    tau = 0.008
    for _ in range(200):
        dots = others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.05, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    v /= np.linalg.norm(v)
    return v

def _lp_polish(x, seconds, r_init=2e-4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-13:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = max(25.0 * radius, 5e-8)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(-c_all, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        csub = c_all[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            colsI = I[:, None] * d + ar
            colsJ = J[:, None] * d + ar
            cols = np.concatenate([colsI, colsJ, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            r_eq = np.repeat(np.arange(n), d)
            c_eq = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (r_eq, c_eq)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)
    # quick break of isolated bottleneck
    try:
        best, best_val = _pair_push(best, 300, 0.002)
        best, bv = _hard(best, 300, 0.003, 150)
        if bv < best_val:
            best_val = bv
        else:
            best_val = _maxcos(best)
    except Exception:
        pass

    def hole_reinsert(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        k = int(np.argmax(rm))
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        Xr = X[mask]
        v = _find_void(Xr, rng, n_cand=8000)
        Xn = X.copy()
        Xn[k] = v
        Xn += 0.001 * rng.standard_normal(Xn.shape)
        return _norm(Xn)

    sigmas = [0.02, 0.01, 0.005, 0.03, 0.008, 0.015]
    trial = 0
    while time.time() - t0 < 122.0 and trial < 60:
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        try:
            if mode == 0:
                Xt = _norm(best + sigma * rng.standard_normal(best.shape))
            elif mode == 1:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                worst = np.argsort(rm)[-12:]
                Xt = best.copy() + 0.002 * rng.standard_normal(best.shape)
                Xt[worst] += sigma * rng.standard_normal((len(worst), d))
                Xt = _norm(Xt)
            elif mode == 2:
                Xt = hole_reinsert(best)
            elif mode == 3:
                Xt = _norm(best + sigma * 0.5 * rng.standard_normal(best.shape))
                # random rotation of a small subset to change topology
                sub = rng.choice(n, size=8, replace=False)
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                Xt[sub] = _norm(Xt[sub] @ Q.T)
            else:
                Xt = _norm(best + sigma * rng.standard_normal(best.shape))
        except Exception:
            trial += 1
            continue
        # explore with annealed soft-max
        try:
            Xt = _smooth(Xt, steps=900, T0=0.012, T1=0.0008, lr0=0.012, lr1=0.0008, mu=0.9)
        except Exception:
            trial += 1
            continue
        if time.time() - t0 > DEADLINE - 3:
            v = _maxcos(Xt)
            if v < best_val:
                best, best_val = Xt, v
            break
        try:
            Xt2, v2 = _hard(Xt, 350, 0.002, 80)
            # contact count decides extra push
            G = Xt2 @ Xt2.T
            np.fill_diagonal(G, -2.0)
            mc = float(G.max())
            nc = int((G > mc - 1e-9).sum() // 2)
            if nc < 25:
                Xt2, v2 = _pair_push(Xt2, 250, 0.0015)
                Xt3, v3 = _hard(Xt2, 250, 0.001, 200)
                Xt2, v2 = (Xt3, v3) if v3 < v2 else (Xt2, v2)
            else:
                Xt3, v3 = _hard(Xt2, 200, 0.0008, 250)
                if v3 < v2:
                    Xt2, v2 = Xt3, v3
        except Exception:
            trial += 1
            continue
        if v2 < best_val - 1e-12:
            best, best_val = Xt2, v2
        trial += 1

    # final intensive polish with remaining time
    try:
        remain = DEADLINE - (time.time() - t0)
        if remain > 8:
            # alternate hard at multiple scales
            for K, s0, it in [(60, 0.0015, 600), (150, 0.0008, 600), (300, 0.0004, 500)]:
                if time.time() - t0 > DEADLINE - 6:
                    break
                b, bv = _hard(best, it, s0, K)
                if bv < best_val:
                    best, best_val = b, bv
                b2, bv2 = _pair_push(best, 200, 0.0008)
                if bv2 < best_val:
                    best, best_val = b2, bv2
        remain = DEADLINE - (time.time() - t0)
        if remain > 5:
            # LP squeeze, tiny radii to tie bulk
            lb, lv = _lp_polish(best, seconds=max(1.0, remain - 2.0), r_init=2e-4)
            if lv < best_val:
                best, best_val = lb, lv
            # one more hard after LP to clean isolated pair
            if time.time() - t0 < DEADLINE - 2:
                b, bv = _hard(best, 400, 0.0005, 200)
                if bv < best_val:
                    best, best_val = b, bv
    except Exception:
        pass
    best = _norm(best)
    return best

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

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _rand_sphere(rng, n, d):
    A = rng.standard_normal(size=(n, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _find_void(others, rng, n_cand=6000):
    d = others.shape[1]
    cand = _rand_sphere(rng, n_cand, d)
    dots = cand @ others.T
    m = dots.max(axis=1)
    v = cand[int(np.argmin(m))].copy()
    lr = 0.05
    tau = 0.008
    for _ in range(250):
        dots = others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = w.sum()
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

def _smooth_opt(X, steps, T0, T1, lr0, lr1, mu, iu0, iu1):
    n, d = X.shape
    V = np.zeros_like(X)
    W = np.empty((n, n), dtype=np.float64)
    for k in range(steps):
        f = k / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        c = G[iu0, iu1]
        m = c.max()
        w = np.exp((c - m) / T)
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            w = np.zeros_like(w)
            w[int(np.argmax(c))] = 1.0
            s = 1.0
        w /= s
        W.fill(0.0)
        W[iu0, iu1] = w
        W[iu1, iu0] = w
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        grad_t = grad - dot * X
        V = mu * V + (1.0 - mu) * grad_t
        dotv = np.einsum('ij,ij->i', V, X)[:, None]
        Vt = V - dotv * X
        X = X - lr * Vt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hard_polish(X, iters, step0, K, iu0, iu1):
    best = _max_cos(X)
    step = step0
    n, d = X.shape
    K = min(K, len(iu0))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < len(c):
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(len(c))
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        ng = np.linalg.norm(gt)
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        v = _max_cos(Xt)
        if v < best:
            X = Xt
            best = v
            step = min(step * 1.3, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def _lp_polish(x, seconds, r_init=2e-4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thresh = max(25.0 * radius, 5e-08)
        sel = cfull > m - thresh
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints if too many (keep tightest 2500)
        if p > 2500:
            order = np.argpartition(-cfull[sel], 2500)[:2500]
            I, J = I[order], J[order]
            p = 2500
            sel_idx = np.where(sel)[0][order]
            csub = cfull[sel_idx]
        else:
            csub = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        # vals built above has wrong size if concatenated incorrectly; rebuild correctly
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq,
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best_x)
    # stage 0: fast squeeze of loaded best with tiny-temperature smooth + hard + small LP
    try:
        X = best_x.copy()
        X = _smooth_opt(X, steps=1200, T0=8e-04, T1=6e-05, lr0=0.03, lr1=0.004, mu=0.8, iu0=iu0, iu1=iu1)
        X, v = _hard_polish(X, iters=300, step0=0.003, K=500, iu0=iu0, iu1=iu1)
        if v < best_val:
            best_val = v
            best_x = X.copy()
    except Exception:
        pass
    # small LP squeeze if time allows
    if time.time() - t0 < BUDGET - 30:
        try:
            remain = min(6.0, BUDGET - (time.time() - t0) - 25)
            if remain > 1:
                Xp = _lp_polish(best_x, seconds=remain, r_init=1e-4)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best_val = vp
                    best_x = Xp
        except Exception:
            pass

    trial = 0
    # basin hopping until ~ t0+140
    while time.time() - t0 < 138.0:
        if time.time() - t0 > BUDGET - 32:
            break
        trial += 1
        mode = trial % 6
        Xt = None
        try:
            if mode == 1:
                sig = 0.008 + 0.004 * rng.random()
                Xt = best_x + sig * rng.standard_normal(size=best_x.shape)
                Xt = _norm(Xt)
                Xp = _smooth_opt(Xt, steps=1600, T0=0.002, T1=8e-05, lr0=0.05, lr1=0.004, mu=0.8, iu0=iu0, iu1=iu1)
                Xp, vp = _hard_polish(Xp, iters=200, step0=0.002, K=400, iu0=iu0, iu1=iu1)
            elif mode == 2:
                sig = 0.025
                Xt = best_x.copy()
                nb = 14
                idx = rng.choice(n, size=nb, replace=False)
                Xt[idx] += sig * rng.standard_normal(size=(nb, d))
                Xt = _norm(Xt)
                Xp = _smooth_opt(Xt, steps=1800, T0=0.003, T1=1e-04, lr0=0.05, lr1=0.004, mu=0.8, iu0=iu0, iu1=iu1)
                Xp, vp = _hard_polish(Xp, iters=200, step0=0.0025, K=500, iu0=iu0, iu1=iu1)
            elif mode == 3:
                rm = _row_max(best_x)
                wi = int(np.argmax(rm))
                others = np.delete(best_x, wi, axis=0)
                v = _find_void(others, rng, n_cand=7000)
                Xt = best_x.copy()
                Xt[wi] = v
                Xp = _smooth_opt(Xt, steps=1800, T0=0.0025, T1=8e-05, lr0=0.05, lr1=0.003, mu=0.85, iu0=iu0, iu1=iu1)
                Xp, vp = _hard_polish(Xp, iters=250, step0=0.002, K=500, iu0=iu0, iu1=iu1)
            elif mode == 4:
                rm = _row_max(best_x)
                k = 2
                worst = np.argpartition(-rm, k)[:k]
                Xt = best_x.copy()
                for wi in worst:
                    oidx = [i for i in range(n) if i != wi]
                    others = Xt[oidx]
                    vv = _find_void(others, rng, n_cand=4000)
                    Xt[wi] = vv
                Xp = _smooth_opt(Xt, steps=1500, T0=0.002, T1=8e-05, lr0=0.04, lr1=0.003, mu=0.8, iu0=iu0, iu1=iu1)
                Xp, vp = _hard_polish(Xp, iters=200, step0=0.002, K=500, iu0=iu0, iu1=iu1)
            elif mode == 5:
                rm = _row_max(best_x)
                k = 5
                worst = np.argpartition(-rm, k)[:k]
                Xt = best_x.copy()
                Xt[worst] = _rand_sphere(rng, k, d)
                Xp = _smooth_opt(Xt, steps=2000, T0=0.004, T1=1.2e-04, lr0=0.06, lr1=0.004, mu=0.8, iu0=iu0, iu1=iu1)
                Xp, vp = _hard_polish(Xp, iters=200, step0=0.003, K=400, iu0=iu0, iu1=iu1)
            else:
                sig = 0.06
                Xt = best_x.copy()
                nb = 25
                idx = rng.choice(n, size=nb, replace=False)
                Xt[idx] += sig * rng.standard_normal(size=(nb, d))
                Xt = _norm(Xt)
                Xp = _smooth_opt(Xt, steps=1600, T0=0.005, T1=1.5e-04, lr0=0.06, lr1=0.005, mu=0.75, iu0=iu0, iu1=iu1)
                Xp, vp = _hard_polish(Xp, iters=180, step0=0.003, K=400, iu0=iu0, iu1=iu1)
        except Exception:
            continue
        if not np.all(np.isfinite(Xp)):
            continue
        v = float(vp) if 'vp' in locals() else _max_cos(Xp)
        if v < best_val - 1e-12:
            best_val = v
            best_x = Xp.copy()
            # quick tiny LP squeeze on new best when promising
            try:
                if time.time() - t0 < BUDGET - 35:
                    Xq = _lp_polish(best_x, seconds=1.5, r_init=5e-05)
                    vq = _max_cos(Xq)
                    if vq < best_val:
                        best_val = vq
                        best_x = Xq
            except Exception:
                pass
        # periodic LP touch every 15 trials
        if trial % 15 == 0 and (time.time() - t0) < 120:
            try:
                Xq = _lp_polish(best_x, seconds=1.0, r_init=5e-05)
                vq = _max_cos(Xq)
                if vq < best_val:
                    best_val = vq
                    best_x = Xq
            except Exception:
                pass

    remain = BUDGET - (time.time() - t0)
    if remain > 4:
        try:
            # final multi-scale polish: smooth with very small T then LP with tiny radius
            X = _smooth_opt(best_x, steps=800, T0=3e-04, T1=2e-05, lr0=0.015, lr1=0.002, mu=0.85, iu0=iu0, iu1=iu1)
            X, v = _hard_polish(X, iters=300, step0=0.001, K=600, iu0=iu0, iu1=iu1)
            if v < best_val:
                best_val = v
                best_x = X.copy()
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        try:
            Xf = _lp_polish(best_x, seconds=max(remain - 1.0, 1.0), r_init=8e-05)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best_x = Xf
        except Exception:
            pass
    best_x = _norm(best_x)
    for i in range(n):
        nrm = float(np.linalg.norm(best_x[i]))
        if not np.isfinite(nrm) or nrm < 1e-12:
            best_x[i] = _rand_sphere(rng, 1, d)[0]
    best_x = _norm(best_x)
    return best_x

import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _lp_polish(x, seconds):
    x = _norm(x.astype(float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:5000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)

    iu = np.triu_indices(n, 1)
    M = len(iu[0])

    def run_lbfgs(Y0, t, tau, maxiter, gtol):
        y0 = Y0.ravel().copy()
        def fun_grad(y):
            Y = y.reshape(n, d)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            X = Y / r
            # all pairs
            G = X @ X.T
            c = G[iu]
            z = (c - t) / tau
            s = np.empty_like(z)
            sig = np.empty_like(z)
            big = z > 20.0
            small = z < -20.0
            mid = ~(big | small)
            s[big] = z[big]
            sig[big] = 1.0
            e_small = np.exp(z[small])
            s[small] = e_small
            sig[small] = e_small
            if np.any(mid):
                zm = z[mid]
                s[mid] = np.log1p(np.exp(zm))
                sig[mid] = 1.0 / (1.0 + np.exp(-zm))
            f = float(np.sum(s * s))
            # anchor radial scale
            rr = r[:, 0]
            reg = 1e-4 * float(np.sum((rr - 1.0) ** 2))
            f = f + reg
            W = (2.0 * s * sig) / tau
            Wfull = np.zeros((n, n), dtype=float)
            Wfull[iu] = W
            Wfull = Wfull + Wfull.T
            grad_X = Wfull @ X
            dot = np.sum(grad_X * X, axis=1, keepdims=True)
            grad_Y = (grad_X - dot * X) / r
            # reg grad: 2e-4*(r-1)*X
            grad_Y = grad_Y + (2e-4 * (rr - 1.0))[:, None] * X
            return f, grad_Y.ravel()
        try:
            res = minimize(fun_grad, y0, method='L-BFGS-B', jac=True,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol, 'maxfun': maxiter * 3})
            Yopt = res.x.reshape(n, d)
            return Yopt
        except Exception:
            return Y0

    targets = [0.4223, 0.4220, 0.4217, 0.4213]
    taus = [1e-3, 3e-4, 1e-4]
    scales = [0.05, 0.10, 0.15]
    lp_calls = 0

    # homotopy budget: leave 25s for final LP + margin
    for ti, t in enumerate(targets):
        if time.time() - t0 > 135:
            break
        for rep, sc in enumerate(scales):
            if time.time() - t0 > 135:
                break
            Y0 = best + sc * rng.standard_normal((n, d))
            cur_Y = Y0
            for k, tau in enumerate(taus):
                if time.time() - t0 > 138:
                    break
                # fewer iters for large tau, more for small tau
                if k == 0:
                    mi, gt = 80, 1e-5
                elif k == 1:
                    mi, gt = 100, 1e-6
                else:
                    mi, gt = 120, 1e-7
                cur_Y = run_lbfgs(cur_Y, t, tau, mi, gt)
                Xcur = _norm(cur_Y)
                v = _max_cos(Xcur)
                # promote only on true improvement (allows temp violations in chain)
                if v < best_val:
                    best_val = v
                    best = Xcur.copy()
                    # keep chain on improved point (already)
                # early exit if feasible below target (near-zero penalty)
                # feasible means max <= t + small tol
                if v <= t + 1e-6:
                    best_val = v
                    best = Xcur.copy()
                    break
            # end tau loop
        # at most two short LP squeezes interleaved
        if lp_calls < 2 and (ti == 1 or ti == 2):
            if time.time() - t0 < 130:
                try:
                    cand = _lp_polish(best, 3.0)
                    v = _max_cos(cand)
                    if v < best_val:
                        best_val = v
                        best = cand
                    lp_calls += 1
                except Exception:
                    pass

    # final 20s LP squeeze if time permits
    remain = 170 - (time.time() - t0)
    if remain > 1:
        fin_sec = min(20.0, remain)
        try:
            cand = _lp_polish(best, fin_sec)
            v = _max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
        except Exception:
            pass
    best = _norm(best)
    return best

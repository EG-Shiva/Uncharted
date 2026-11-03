import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _smooth_refine(x, steps, T0, T1, lr0, lr1, momentum=0.85):
    n = x.shape[0]
    v = np.zeros_like(x)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sum(W)) / n + 1e-12
        grad = grad / s
        v = momentum * v + (1.0 - momentum) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _find_hole(Xk, rng, n_samples=20000):
    d = Xk.shape[1]
    Z = rng.randn(n_samples, d)
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=150, T=0.008, lr=0.02):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _lbfgs_smooth(x_init, T, maxiter=80):
    n, d = x_init.shape
    iu = np.triu_indices(n, 1)
    y0 = x_init.ravel().copy()
    def fun_grad(y):
        Y = y.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        E = np.exp((c - m) / T)
        S = float(E.sum())
        f = m + T * np.log(S + 1e-300)
        W = E / (S + 1e-300)
        # symmetric weight matrix applied as W_full @ X
        # build grad via accumulation (avoid full matrix)
        # grad_X[i] = sum_{j != i} w_{ij} x_j
        # use sparse accumulation with iu
        grad_X = np.zeros_like(X)
        # for each pair k: contributes Wk * x_j to i and Wk * x_i to j
        # vectorized via bincount-like using numpy add
        Xi = X[iu[0]]
        Xj = X[iu[1]]
        # contributions
        # grad_X[ I ] += W * Xj ; grad_X[ J ] += W * Xi
        # use np.add.at
        np.add.at(grad_X, iu[0], (W[:, None] * Xj))
        np.add.at(grad_X, iu[1], (W[:, None] * Xi))
        # chain to Y: (grad - (grad.X)X)/r
        dot = np.sum(grad_X * X, axis=1, keepdims=True)
        grad_Y = (grad_X - dot * X) / r
        return float(f), grad_Y.ravel()
    try:
        res = minimize(fun_grad, y0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxiter*2})
        Y = res.x.reshape(n, d)
        X = _norm(Y)
        return X
    except Exception:
        return x_init

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
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
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        if deadline - time.time() < 0.4:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(3)
    n_target, d_target = 131, 9
    best_x = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n_target, d_target):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.randn(n_target, d_target))
        best_s = _max_cos(best_x)
    x_best = best_x.copy()
    score_best = best_s
    x_cur = best_x.copy()
    score_cur = best_s

    HOP_END = t0 + 122.0
    trial = 0
    while time.time() < HOP_END:
        trial += 1
        remain = HOP_END - time.time()
        if remain < 1.0:
            break
        # annealing temperature
        prog = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Temp = 0.0004 * (1 - prog) + 0.00002
        r = rng.rand()
        try:
            if r < 0.30:
                sigma = 0.015 if (trial % 2 == 0) else 0.055
                if trial % 31 == 0:
                    sigma = 0.22
                elif trial % 13 == 0:
                    sigma = 0.11
                x_try = x_cur + sigma * rng.randn(*x_cur.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=220, T0=0.018, T1=0.006, lr0=0.013, lr1=0.0022)
                x_try = _smooth_refine(x_try, steps=200, T0=0.006, T1=0.0018, lr0=0.004, lr1=0.001)
            elif r < 0.55:
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                # pick 1 or 2 points
                k = 1 if rng.rand() < 0.6 else 2
                cand_pool = order[:8]
                chosen = []
                perm = rng.permutation(len(cand_pool))[:k]
                for q in perm:
                    chosen.append(int(cand_pool[q]))
                x_try = x_cur.copy()
                mask_keep = np.ones(len(x_try), dtype=bool)
                for cidx in chosen:
                    mask_keep[cidx] = False
                Xk = x_try[mask_keep]
                for cidx in chosen:
                    z, _ = _find_hole(Xk, rng, n_samples=16000)
                    z = _refine_hole(z, Xk, steps=120, T=0.008, lr=0.02)
                    x_try[cidx] = z
                    Xk = np.vstack([Xk, z[None, :]])
                x_try = x_try + 0.003 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=260, T0=0.014, T1=0.005, lr0=0.011, lr1=0.002)
                x_try = _smooth_refine(x_try, steps=200, T0=0.005, T1=0.0018, lr0=0.004, lr1=0.001)
            elif r < 0.70:
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                worst = order[:16]
                x_try = x_cur.copy()
                x_try[worst] = x_try[worst] + 0.08 * rng.randn(len(worst), x_try.shape[1])
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=220, T0=0.016, T1=0.005, lr0=0.012, lr1=0.002)
                x_try = _smooth_refine(x_try, steps=200, T0=0.005, T1=0.0018, lr0=0.004, lr1=0.001)
            elif r < 0.85:
                # rattler + global jitter: identify rattlers w.r.t tight threshold
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = np.sum(G > m - 1e-9, axis=1)
                ratt = np.where(cnt == 0)[0]
                x_try = x_cur.copy()
                if len(ratt) > 0:
                    # randomize a few rattlers into holes
                    k = min(len(ratt), 1 + int(rng.randint(3)))
                    sel_r = ratt[rng.permutation(len(ratt))[:k]]
                    mask = np.ones(len(x_try), dtype=bool)
                    mask[sel_r] = False
                    Xk = x_try[mask]
                    for cidx in sel_r:
                        z, _ = _find_hole(Xk, rng, n_samples=12000)
                        z = _refine_hole(z, Xk, steps=100, T=0.01, lr=0.02)
                        x_try[cidx] = z
                        Xk = np.vstack([Xk, z[None, :]])
                x_try = x_try + 0.008 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=240, T0=0.015, T1=0.005, lr0=0.012, lr1=0.002)
                x_try = _smooth_refine(x_try, steps=200, T0=0.005, T1=0.0018, lr0=0.004, lr1=0.001)
            else:
                sigma = 0.03 + 0.02 * rng.rand()
                x_try = x_cur + sigma * rng.randn(*x_cur.shape)
                x_try = _norm(x_try)
                # L-BFGS at moderate T directly
                x_try = _lbfgs_smooth(x_try, T=0.006, maxiter=60)
                x_try = _smooth_refine(x_try, steps=150, T0=0.004, T1=0.0018, lr0=0.003, lr1=0.001)

            # opportunistic L-BFGS squeeze if close to best
            s_fast = _max_cos(x_try)
            if s_fast < score_best + 0.002 and (trial % 4 == 0 or s_fast < score_best):
                x_try = _lbfgs_smooth(x_try, T=0.0035, maxiter=50)
                # second stage colder
                if _max_cos(x_try) < score_best + 0.001:
                    x_try = _lbfgs_smooth(x_try, T=0.0015, maxiter=50)
            s = _max_cos(x_try)
            if s < score_best:
                score_best = s
                x_best = x_try.copy()
                x_cur = x_try.copy()
                score_cur = s
            else:
                if s < score_cur or rng.rand() < np.exp((score_cur - s) / max(Temp, 1e-9)):
                    x_cur = x_try.copy()
                    score_cur = s
                # occasional restart from best
                if trial % 50 == 0 and rng.rand() < 0.5:
                    x_cur = x_best.copy()
                    score_cur = score_best
        except Exception:
            continue

    # final chain: L-BFGS cold then LP
    try:
        x_best = _lbfgs_smooth(x_best, T=0.004, maxiter=80)
        x_best = _lbfgs_smooth(x_best, T=0.0018, maxiter=80)
        x_best = _lbfgs_smooth(x_best, T=0.0009, maxiter=60)
    except Exception:
        pass
    remain = BUDGET - (time.time() - t0)
    if remain > 6.0:
        try:
            x_best = _lp_polish(x_best, time.time() + remain - 1.0)
        except Exception:
            pass
    x_best = _norm(x_best)
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(best_x)
    return x_best

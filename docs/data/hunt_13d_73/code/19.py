import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _find_hole(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _adam_refine(X0, steps, s0, s1, lr0, lr1):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    n = X.shape[0]
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        # softmax weights, cap exponent for stability
        D = s * (G - cmax)
        # clip to avoid overflow (D<=0)
        np.clip(D, -60.0, 0.0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        sumE = E.sum() * 0.5
        if sumE <= 0 or not np.isfinite(sumE):
            continue
        W = E / sumE
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1
        m1 += (1 - b1) * gt
        v1 *= b2
        v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X -= lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lbfgs_polish(X0, alphas=(120.0, 350.0), maxiter_each=60):
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y = X0.ravel().copy()
    for alpha in alphas:
        def fun(Yflat):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            dots = G[iu0]
            M = dots.max()
            e = np.exp(alpha * (dots - M))
            S = e.sum()
            loss = M + np.log(S) / alpha
            p = e / S
            W = np.zeros((n, n))
            W[iu0] = p
            W = W + W.T
            grad_X = W @ X
            dg = (grad_X * X).sum(axis=1, keepdims=True)
            grad_Xt = grad_X - dg * X
            grad_Y = (grad_Xt / rn).ravel()
            return float(loss), grad_Y.astype(float)
        try:
            res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter_each, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 10})
            Y = res.x
        except Exception:
            break
    Ym = Y.reshape(n, d)
    Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
    return Ym

def _lp_polish(x, seconds, t_end=1e30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
    EXPLORE_END = t0 + 132.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = rng.normal(size=(n, d))
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(Xb.shape[0], n); m1 = min(Xb.shape[1], d)
        tmp[:m0, :m1] = Xb[:m0, :m1]
        Xb = tmp
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    X_cur = Xb.copy()
    cur_val = best_val
    sigmas = [0.012, 0.03, 0.07, 0.15, 0.28]
    it = 0
    Temp = 0.0018
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 35.0:
        it += 1
        use_best = rng.random() < 0.75
        base = X_best if use_best else X_cur
        mode = it % 10
        try:
            if mode in (0, 3, 7, 9):
                k = 1
                if mode == 0:
                    k = 2
                if it % 25 == 0:
                    k = 3
                rm = _rowwise_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                if rng.random() < 0.35:
                    victims[0] = int(rng.integers(0, n))
                    if k > 1 and rng.random() < 0.5:
                        victims[1] = int(rng.integers(0, n))
                X_try = base.copy()
                cur_keep_full = base.copy()
                for idx in victims:
                    mask = np.ones(n, dtype=bool); mask[idx] = False
                    # use current evolving set for hole search
                    others = X_try[mask]
                    h = _find_hole(others, rng, n_cand=4000)
                    X_try[idx] = h
                X_try += 0.006 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                steps = 700
                s0, s1 = 40.0, 350.0
                lr0, lr1 = 0.01, 0.002
            elif mode == 5:
                X_try = base.copy()
                vi = int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool); mask[vi] = False
                h = _find_hole(base[mask], rng, n_cand=5000)
                X_try[vi] = h
                X_try += 0.008 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                steps = 600
                s0, s1 = 50.0, 400.0
                lr0, lr1 = 0.01, 0.002
            else:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                X_try = base + sigma * rng.normal(size=base.shape)
                if rng.random() < 0.12:
                    vi = int(rng.integers(0, n))
                    X_try[vi] = base[vi] + 0.5 * rng.normal(size=(d,))
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                steps = 500
                s0, s1 = 40.0, 300.0
                lr0, lr1 = 0.012, 0.002
        except Exception:
            continue
        try:
            X_opt = _adam_refine(X_try, steps=steps, s0=s0, s1=s1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        try:
            v = _max_cos(X_opt)
        except Exception:
            continue
        if v < best_val - 1e-12:
            # quick L-BFGS squeeze on promising newcomer
            try:
                if v < best_val + 0.002 and time.time() < EXPLORE_END - 5:
                    X_opt2 = _lbfgs_polish(X_opt, alphas=(150.0, 400.0), maxiter_each=40)
                    v2 = _max_cos(X_opt2)
                    if v2 < v:
                        X_opt, v = X_opt2, v2
            except Exception:
                pass
            if v < best_val - 1e-12:
                X_best = X_opt.copy()
                best_val = float(v)
                X_cur = X_opt.copy()
                cur_val = float(v)
                if best_val < 0.19:
                    break
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / Temp):
            X_cur = X_opt.copy()
            cur_val = float(v)
    # final polish
    remain = DEADLINE - time.time() - 1.5
    if remain > 5.0:
        try:
            # sharp L-BFGS then LP
            tb = min(12.0, remain * 0.4)
            t_start = time.time()
            # only one L-BFGS pass to save time
            X_best = _lbfgs_polish(X_best, alphas=(120.0, 350.0, 600.0), maxiter_each=50)
            best_val = _max_cos(X_best)
            left = min(remain - (time.time() - t_start), DEADLINE - time.time() - 0.8)
            if left > 3.0:
                X_best = _lp_polish(X_best, seconds=min(left, 28.0), t_end=DEADLINE - 0.8)
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # sanitize
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best /= nr
    return X_best

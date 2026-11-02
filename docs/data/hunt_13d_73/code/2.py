import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def random_on_sphere(rng, n, d):
    Y = rng.normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def find_hole(X_keep, rng, n_cand=6000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def smooth_opt(X0, alphas=(80.0, 250.0), maxiter_each=120):
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

def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = random_on_sphere(rng, 73, 13)
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (73, 13):
        Xb = random_on_sphere(rng, 73, 13)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = max_cos(Xb)
    X_best = Xb.copy()
    X_cur = Xb.copy()
    cur_val = best_val
    sigmas = [0.015, 0.03, 0.06, 0.11, 0.20, 0.32]
    it = 0
    # short exploration budget, leave time for LP polish
    EXPLORE_END = t0 + 135.0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 32.0:
        it += 1
        use_best_base = rng.random() < 0.72
        base = X_best if use_best_base else X_cur
        n, d = base.shape
        X_try = None
        mode = it % 10
        try:
            if mode in (0, 9):  # kill overloaded and reinsert into holes
                k = 2 if mode == 0 else 1
                if it % 20 == 0:
                    k = 3
                rm = rowwise_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                # sometimes random victim for diversity
                if rng.random() < 0.3:
                    victims[0] = int(rng.integers(0, n))
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                X_keep = base[keep_mask]
                X_try = base.copy()
                # sequential hole filling
                cur_keep = X_keep.copy()
                new_pts = []
                for _ in range(k):
                    h = find_hole(cur_keep, rng, n_cand=6000)
                    new_pts.append(h)
                    cur_keep = np.vstack([cur_keep, h[None, :]])
                for idx, pt in zip(victims, new_pts):
                    X_try[idx] = pt
                # small jitter on survivors
                X_try += 0.008 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            elif mode == 5:  # randomize one point to hole + jitter
                X_try = base.copy()
                vi = int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool); mask[vi] = False
                h = find_hole(base[mask], rng, n_cand=8000)
                X_try[vi] = h
                X_try += 0.01 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            else:
                sigma = sigmas[(it + int(rng.integers(0, len(sigmas)))) % len(sigmas)]
                # occasionally big shake of single rattler-like point
                X_try = base + sigma * rng.normal(size=base.shape)
                if rng.random() < 0.15:
                    vi = int(rng.integers(0, n))
                    X_try[vi] = base[vi] + 0.6 * rng.normal(size=(d,))
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        except Exception:
            continue
        # smooth optimization, cheaper for large sigma trials
        try:
            if best_val < 0.2074:
                alphas = (60.0, 150.0, 400.0)
                mi = 100
            else:
                alphas = (80.0, 250.0)
                mi = 110
            X_opt = smooth_opt(X_try, alphas=alphas, maxiter_each=mi)
        except Exception:
            continue
        try:
            v = max_cos(X_opt)
        except Exception:
            continue
        if v < best_val - 1e-12:
            X_best = X_opt.copy()
            best_val = float(v)
            X_cur = X_opt.copy()
            cur_val = float(v)
            # quick LP squeeze if promising and time allows
            if time.time() < EXPLORE_END - 8.0:
                try:
                    Xp = lp_polish(X_best, seconds=2.5, t_end=EXPLORE_END)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        X_best = Xp
                        best_val = float(vp)
                        X_cur = Xp.copy()
                        cur_val = float(vp)
                except Exception:
                    pass
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / 0.0025):
            X_cur = X_opt.copy()
            cur_val = float(v)
        # safety
        if best_val < 0.19:
            break
    # final polish of best
    remain = DEADLINE - time.time() - 2.0
    if remain > 4.0:
        try:
            X_best = lp_polish(X_best, seconds=min(remain, 30.0), t_end=DEADLINE - 1.0)
            best_val = max_cos(X_best)
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

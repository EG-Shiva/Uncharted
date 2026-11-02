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

def _random_on_sphere(rng, n, d):
    Y = rng.normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _smooth_opt(X0, alphas=(80.0, 250.0), maxiter_each=100):
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

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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

def _deep_hole(X_keep, rng, n_cand=8000, steps=120, alpha=150.0):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    order = np.argsort(m)
    best_val = float(m[order[0]])
    best_h = C[order[0]].copy()
    # refine top few candidates with Adam on sphere minimizing LSE
    for cand_idx in order[:4]:
        h = C[cand_idx].copy()
        mt = np.zeros(d)
        vt = np.zeros(d)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        lr0, lr1 = 0.06, 0.005
        cur_best_v = float(m[cand_idx])
        cur_best_h = h.copy()
        for k in range(steps):
            frac = k / max(steps - 1, 1)
            lr = np.exp(np.log(lr0) * (1 - frac) + np.log(lr1) * frac)
            s = h @ X_keep.T
            Mx = s.max()
            e = np.exp(alpha * (s - Mx))
            ss = e.sum()
            if not np.isfinite(ss) or ss == 0:
                break
            w = e / ss
            grad = w @ X_keep
            gt = grad - (grad @ h) * h
            mt = b1 * mt + (1 - b1) * gt
            vt = b2 * vt + (1 - b2) * (gt * gt)
            mhat = mt / (1 - b1p)
            vhat = vt / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            h = h - lr * step
            nh = np.linalg.norm(h)
            if nh == 0 or not np.isfinite(nh):
                break
            h /= nh
            vnow = float((h @ X_keep.T).max())
            if vnow < cur_best_v:
                cur_best_v = vnow
                cur_best_h = h.copy()
        if cur_best_v < best_val:
            best_val = cur_best_v
            best_h = cur_best_h.copy()
    return best_h

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = _random_on_sphere(rng, n, d)
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (n, d):
        Xb = _random_on_sphere(rng, n, d)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    # quick squeeze of starting point
    try:
        if DEADLINE - time.time() > 20:
            Xp = _lp_polish(X_best, seconds=2.0, t_end=DEADLINE - 5)
            vp = _max_cos(Xp)
            if vp < best_val:
                X_best = Xp
                best_val = float(vp)
    except Exception:
        pass
    X_cur = X_best.copy()
    cur_val = float(best_val)
    sigmas = [0.012, 0.025, 0.05, 0.09, 0.16, 0.28]
    EXPLORE_END = t0 + 138.0
    it = 0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 32.0:
        it += 1
        use_best = rng.random() < 0.70
        base = X_best if use_best else X_cur
        X_try = None
        try:
            # fresh random restart periodically for new basin
            if it % 24 == 7:
                X_try = _random_on_sphere(rng, n, d)
            elif it % 10 in (0, 9):
                k = 2 if (it % 10 == 0) else 1
                if it % 30 == 0:
                    k = 3
                rm = _rowwise_max(base)
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                thr = G.max() - 1e-3
                cnt = (G > thr).sum(axis=1)
                score = rm + 0.002 * (cnt / max(cnt.max(), 1))
                order = np.argsort(-score)
                victims = order[:k].copy()
                if rng.random() < 0.30:
                    victims[0] = int(rng.integers(0, n))
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                X_keep = base[keep_mask]
                X_try = base.copy()
                cur_keep = X_keep.copy()
                new_pts = []
                for _ in range(k):
                    h = _deep_hole(cur_keep, rng, n_cand=6000, steps=100, alpha=150.0)
                    new_pts.append(h)
                    cur_keep = np.vstack([cur_keep, h[None, :]])
                for idx, pt in zip(victims, new_pts):
                    X_try[idx] = pt
                X_try += 0.006 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            elif it % 10 == 5:
                # cluster shake: pick center, shake nearest neighbours
                X_try = base.copy()
                ci = int(rng.integers(0, n))
                dots_c = base @ base[ci]
                nn = np.argsort(-dots_c)[:8]
                X_try[nn] += 0.07 * rng.normal(size=(len(nn), d))
                X_try += 0.010 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            elif it % 10 == 3:
                # worst-pair repulsion
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                flat = np.argmax(G)
                wi, wj = flat // n, flat % n
                X_try = base.copy()
                vi = base[wi] - base[wj]
                vj = base[wj] - base[wi]
                X_try[wi] += 0.03 * vi + 0.015 * rng.normal(size=(d,))
                X_try[wj] += 0.03 * vj + 0.015 * rng.normal(size=(d,))
                X_try += 0.008 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            else:
                sigma = sigmas[(it + int(rng.integers(0, len(sigmas)))) % len(sigmas)]
                X_try = base + sigma * rng.normal(size=base.shape)
                if rng.random() < 0.15:
                    vi = int(rng.integers(0, n))
                    X_try[vi] = base[vi] + 0.6 * rng.normal(size=(d,))
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        except Exception:
            continue
        if X_try is None:
            continue
        try:
            # fast quench
            if it % 24 == 7:
                X_opt = _smooth_opt(X_try, alphas=(80.0, 250.0), maxiter_each=80)
            else:
                X_opt = _smooth_opt(X_try, alphas=(70.0, 220.0), maxiter_each=90)
        except Exception:
            continue
        try:
            v = _max_cos(X_opt)
        except Exception:
            continue
        # extra refinement if promising
        if v < best_val + 0.002 and v > -1:
            try:
                X_opt2 = _smooth_opt(X_opt, alphas=(600.0,), maxiter_each=40)
                v2 = _max_cos(X_opt2)
                if v2 < v:
                    X_opt = X_opt2
                    v = v2
            except Exception:
                pass
        if v < best_val - 1e-12:
            X_best = X_opt.copy()
            best_val = float(v)
            X_cur = X_opt.copy()
            cur_val = float(v)
            if time.time() < EXPLORE_END - 6.0:
                try:
                    Xp = _lp_polish(X_best, seconds=2.0, t_end=EXPLORE_END)
                    vp = _max_cos(Xp)
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
        if best_val < 0.19:
            break
    remain = DEADLINE - time.time() - 1.5
    if remain > 5.0:
        try:
            # high-alpha squeeze then LP
            X_best = _smooth_opt(X_best, alphas=(500.0,), maxiter_each=60)
        except Exception:
            pass
        try:
            X_best = _lp_polish(X_best, seconds=min(remain - 1.0, 30.0), t_end=DEADLINE - 1.0)
        except Exception:
            pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # guard NaN
    if not np.all(np.isfinite(X_best)):
        X_best = Xb.copy()
    return X_best

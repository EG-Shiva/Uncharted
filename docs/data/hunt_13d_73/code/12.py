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

def smooth_opt(X0, alphas=(80.0, 250.0), maxiter_each=100):
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

def refine_hole(h0, X_others, steps=260, a0=30.0, a1=700.0, lr0=0.25, lr1=0.002):
    h = h0.astype(float).copy()
    nh = np.linalg.norm(h)
    if nh == 0:
        h = X_others[0] * -1.0
        nh = np.linalg.norm(h)
    h /= nh
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    m = X_others.shape[0]
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        dots = X_others @ h
        cmax = dots.max()
        e = np.exp(alpha * (dots - cmax))
        s = e.sum()
        if s == 0 or not np.isfinite(s):
            w = np.zeros(m)
            w[np.argmax(dots)] = 1.0
        else:
            w = e / s
        grad = w @ X_others
        gt = grad - (float(grad @ h)) * h
        h = h - lr * gt
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            break
        h /= nh
    return h

def propose_holes(X_keep, k, rng, n_cand=12000):
    d = X_keep.shape[1]
    P = rng.normal(size=(n_cand, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    D = P @ X_keep.T
    holes = []
    for _ in range(k):
        md = D.max(axis=1)
        idx = int(np.argmin(md))
        h0 = P[idx].copy()
        if holes:
            Xr = np.vstack([X_keep] + [hh[None, :] for hh in holes])
        else:
            Xr = X_keep
        h = refine_hole(h0, Xr)
        holes.append(h)
        col = P @ h
        D = np.column_stack([D, col])
    return np.array(holes)

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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def random_on_sphere(rng, n, d):
    Y = rng.normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n, d = 73, 13
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = random_on_sphere(rng, n, d)
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (n, d):
        Xb = random_on_sphere(rng, n, d)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = max_cos(Xb)
    X_best = Xb.copy()
    X_cur = Xb.copy()
    cur_val = best_val
    sigmas = [0.008, 0.015, 0.03, 0.06, 0.12, 0.22, 0.35]
    it = 0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 32.0:
        it += 1
        if best_val < 0.19:
            break
        use_best = rng.random() < 0.75
        base = X_best if use_best else X_cur
        X_try = None
        r = rng.random()
        try:
            if r < 0.30:
                # kill k worst + refined holes
                if rng.random() < 0.5:
                    k = 2
                elif rng.random() < 0.7:
                    k = 3
                elif rng.random() < 0.5:
                    k = 4
                else:
                    k = int(rng.integers(1, 6))
                rm = rowwise_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                if rng.random() < 0.35:
                    victims[int(rng.integers(0, k))] = int(rng.integers(0, n))
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                X_keep = base[keep_mask]
                nc = 12000 if k > 1 else 8000
                holes = propose_holes(X_keep, k, rng, n_cand=nc)
                X_try = base.copy()
                for idx, pt in zip(victims, holes):
                    X_try[idx] = pt
                X_try += 0.006 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            elif r < 0.44:
                # single hole + jitter
                X_try = base.copy()
                vi = int(rng.integers(0, n))
                # bias to bad points
                if rng.random() < 0.7:
                    rm = rowwise_max(base)
                    vi = int(np.argmax(rm)) if rng.random() < 0.5 else int(rng.choice(np.argsort(-rm)[:8]))
                mask = np.ones(n, dtype=bool); mask[vi] = False
                holes = propose_holes(base[mask], 1, rng, n_cand=8000)
                X_try[vi] = holes[0]
                X_try += 0.008 * rng.normal(size=X_try.shape)
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            elif r < 0.58:
                # coherent cap shake
                X_try = base.copy()
                u = rng.normal(size=(d,)); u /= np.linalg.norm(u)
                dots_u = base @ u
                thr = float(rng.uniform(0.15, 0.55))
                cap = np.where(dots_u > thr)[0]
                if len(cap) < 2 or len(cap) > 25:
                    # fallback to gaussian
                    sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                    X_try = base + sigma * rng.normal(size=base.shape)
                    X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                else:
                    common = rng.normal(size=(d,)) * float(rng.uniform(0.08, 0.28))
                    for idx in cap:
                        X_try[idx] = base[idx] + common + 0.02 * rng.normal(size=(d,))
                    # jitter rest slightly
                    X_try += 0.004 * rng.normal(size=X_try.shape)
                    X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            else:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                # occasionally perturb sigma
                if rng.random() < 0.2:
                    sigma *= float(rng.uniform(0.7, 1.4))
                X_try = base + sigma * rng.normal(size=base.shape)
                if rng.random() < 0.12:
                    vi = int(rng.integers(0, n))
                    X_try[vi] = base[vi] + 0.7 * rng.normal(size=(d,))
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        except Exception:
            continue
        try:
            X_opt = smooth_opt(X_try, alphas=(80.0, 280.0), maxiter_each=95)
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
            if time.time() < EXPLORE_END - 6.0:
                try:
                    Xp = lp_polish(X_best, seconds=2.0, t_end=EXPLORE_END)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        X_best = Xp; best_val = float(vp)
                        X_cur = Xp.copy(); cur_val = float(vp)
                except Exception:
                    pass
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / 0.002):
            X_cur = X_opt.copy()
            cur_val = float(v)
    remain = DEADLINE - time.time() - 1.5
    if remain > 4.0:
        try:
            X_best = lp_polish(X_best, seconds=min(remain, 30.0), t_end=DEADLINE - 1.0)
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def contact_counts(x, mv, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > mv - tol, axis=1)

def lp_polish(x, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_lbfgs(x_init, betas, maxiters):
    from scipy.optimize import minimize
    n, d = x_init.shape
    Y = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    iu = np.triu_indices(n, 1)
    for beta, mi in zip(betas, maxiters):
        beta = float(beta)
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu]
            mv = float(np.max(c))
            E = np.exp(beta * (c - mv))
            S = float(np.sum(E)) + 1e-300
            f = mv + np.log(S) / beta
            W = np.zeros((n, n))
            W[iu] = E
            W = W + W.T
            GX = (W @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            return float(f), GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14,
                                    'gtol': 1e-6, 'maxcor': 12})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    X = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X

def refine_void(v0, Xrem, beta=130.0, steps=150, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = float(np.max(dots))
        W = np.exp(beta * (dots - mv))
        S = float(np.sum(W)) + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        nv = np.linalg.norm(v) + 1e-30
        v = v / nv
        if not np.all(np.isfinite(v)):
            break
    v = v / (np.linalg.norm(v) + 1e-30)
    return v

def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    rng = np.random.default_rng(1)
    np.random.seed(1)
    n, d = 138, 8

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)

    best = None
    best_val = np.inf
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    try:
        cc = contact_counts(best, best_val, tol=1e-7)
    except Exception:
        cc = np.ones(n)

    trial = 0
    # explore until ~125s, reserve rest for final polish
    while time.time() - t0 < 122.0:
        trial += 1
        if T_LIMIT - (time.time() - t0) < 14.0:
            break
        if trial % 6 == 1:
            try:
                cc = contact_counts(best, best_val, tol=1e-7)
            except Exception:
                pass
        x_try = best.copy()
        mode = trial % 3
        if mode == 0 or mode == 2:
            # worst-point void rebirth
            try:
                pm = per_max(best)
            except Exception:
                pm = cc.astype(float)
            order = np.argsort(-pm)
            pool = order[:12]
            victim = int(pool[rng.integers(len(pool))])
            Xrem = np.delete(best, victim, axis=0)
            Ncand = 15000
            cand = rng.standard_normal((Ncand, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            smax = (cand @ Xrem.T).max(axis=1)
            # pick among few best voids for diversity
            kbest = 4
            bidx = np.argpartition(smax, kbest)[:kbest]
            bi = int(bidx[rng.integers(kbest)])
            v0 = cand[bi]
            try:
                beta_v = 100.0 + rng.random() * 80.0
                v0 = refine_void(v0, Xrem, beta=beta_v, steps=150, lr=0.08)
            except Exception:
                pass
            x_try = best.copy()
            x_try[victim] = v0
            # small jitter to neighbours of victim to break symmetry
            if rng.random() < 0.7:
                Gv = best @ best[victim]
                Gv[victim] = -2.0
                nbr = np.argsort(-Gv)[:8]
                x_try[nbr] += 0.008 * rng.standard_normal((len(nbr), d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # subset kick biased to jammed points
            k = int(12 + rng.integers(0, 18))
            w = cc.astype(float) + 1.0
            w /= w.sum()
            try:
                idx = rng.choice(n, size=k, replace=False, p=w)
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
            sigma = float([0.25, 0.15, 0.08, 0.04][(trial // 3) % 4]) * (0.7 + 0.6 * rng.random())
            x_try = best + 0.012 * rng.standard_normal(best.shape)
            x_try[idx] += (sigma * 2.0) * rng.standard_normal((k, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # jam with smooth L-BFGS
        elapsed = time.time() - t0
        if elapsed > 100:
            betas = [60.0, 160.0]
            mits = [100, 100]
        else:
            b0 = 40.0 + rng.random() * 30.0
            b1 = 120.0 + rng.random() * 120.0
            betas = [b0, b1]
            mits = [130, 130]
        try:
            x_try = smooth_lbfgs(x_try, betas, mits)
        except Exception:
            continue
        try:
            v_try = max_cos(x_try)
        except Exception:
            continue
        if v_try < best_val + 0.012:
            ps = 1.4
            if v_try < best_val:
                ps = 2.2
            tl = T_LIMIT - (time.time() - t0)
            ps = min(ps, max(0.3, tl - 10.0))
            if ps < 0.25:
                # no time for LP, just greedy keep
                if v_try < best_val:
                    best, best_val = x_try, v_try
                    try:
                        cc = contact_counts(best, best_val)
                    except Exception:
                        pass
                continue
            try:
                x_pol = lp_polish(x_try, ps)
                v_pol = max_cos(x_pol)
            except Exception:
                continue
            if v_pol < best_val - 1e-12:
                best = x_pol
                best_val = v_pol
                try:
                    cc = contact_counts(best, best_val)
                except Exception:
                    pass

    tl = T_LIMIT - (time.time() - t0)
    if tl > 2.5:
        final_sec = max(1.0, min(30.0, tl - 1.5))
        try:
            best = lp_polish(best, final_sec)
            best_val = max_cos(best)
        except Exception:
            pass
    # micro-kicks if time remains
    while time.time() - t0 < T_LIMIT - 1.0:
        try:
            x_try = best + 0.004 * rng.standard_normal(best.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_pol = lp_polish(x_try, 0.7)
            v_pol = max_cos(x_pol)
            if v_pol < best_val:
                best, best_val = x_pol, v_pol
            else:
                break
        except Exception:
            break
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

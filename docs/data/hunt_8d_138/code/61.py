import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(X0, betas, iters_list, lrs, deadline):
    n, d = X0.shape
    Y = X0.copy()
    m1 = np.zeros_like(Y)
    v1 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for beta, iters, lr in zip(betas, iters_list, lrs):
        for t in range(iters):
            if (t % 50 == 0) and (time.time() > deadline):
                return _norm(Y)
            nn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-30
            X = Y / nn
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mv = float(G.max())
            E = np.exp(beta * (G - mv))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5) + 1e-300
            W = E / S
            GX = W @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nn
            step += 1
            m1 = b1 * m1 + (1 - b1) * GY
            v1 = b2 * v1 + (1 - b2) * GY * GY
            mh = m1 / (1 - b1 ** step)
            vh = v1 / (1 - b2 ** step)
            Y -= lr * mh / (np.sqrt(vh) + eps)
    return _norm(Y)

def _void_kick(X, n_kick, rng, gap=0.004, pool=20000):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    cnt = np.count_nonzero(G > (m - gap), axis=1)
    order = np.argsort(cnt)
    # half lowest (rattlers), half highest (hubs)
    nh = n_kick // 2
    nl = n_kick - nh
    victims = list(order[:nl]) + list(order[n - nh:])
    victims = sorted(victims)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    Xr = X[keep_mask]
    # candidate pool
    P = _norm(rng.standard_normal((pool, d)))
    # greedy farthest insertion
    cur = Xr.copy()
    new_pts = []
    # precompute in chunks to save memory
    for _ in range(n_kick):
        # max dot of each pool point to cur set
        best_i = -1
        best_v = 1e9
        # chunk pool
        ch = 5000
        maxs = np.empty(pool)
        for s in range(0, pool, ch):
            e = min(s + ch, pool)
            D = P[s:e] @ cur.T
            maxs[s:e] = D.max(axis=1)
        # pick best (minimal max dot); add tiny random tie break by taking argmin
        best_i = int(np.argmin(maxs))
        bp = P[best_i]
        new_pts.append(bp)
        cur = np.vstack([cur, bp.reshape(1, d)])
        # remove chosen from pool by setting max large so not reselected
        # replace it with new random point
        P[best_i] = _norm(rng.standard_normal((d,)))
    Xn = X.copy()
    for vi, bp in zip(victims, new_pts):
        # small tangent jitter
        jt = rng.standard_normal(d) * 0.01
        jt = jt - np.dot(jt, bp) * bp
        Xn[vi] = _norm((bp + jt).reshape(1, d))[0]
    return _norm(Xn)

def _tangent_perturb(X, sigma, rng):
    N = rng.standard_normal(X.shape)
    dot = np.sum(N * X, axis=1, keepdims=True)
    Nt = N - dot * X
    return _norm(X + sigma * Nt)

def _lp_polish(x, deadline, cap=2500):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 5e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 100:
        it += 1
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-4)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    deadline = t0 + 168.0
    n, d = 138, 8
    rng = np.random.default_rng(0)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = _norm(vv)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c
                    best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    base = best.copy()

    betas_full = [50.0, 150.0, 400.0, 1000.0]
    iters_full = [300, 300, 350, 350]
    lrs_full = [0.02, 0.015, 0.01, 0.005]
    betas_short = [150.0, 400.0, 1000.0]
    iters_short = [200, 250, 250]
    lrs_short = [0.015, 0.01, 0.005]

    opt_deadline = t0 + 125.0
    # diverse starts
    sigmas = [0.0, 0.06, 0.13, 0.22, 0.34, 0.5]
    for si, sig in enumerate(sigmas):
        if time.time() > opt_deadline:
            break
        r2 = np.random.default_rng(100 + si)
        if sig == 0.0:
            X0 = base.copy()
        elif si == len(sigmas) - 1:
            X0 = _norm(r2.standard_normal((n, d)))
        else:
            X0 = _tangent_perturb(base, sig, r2)
        try:
            Xopt = _adam_lse(X0, betas_full, iters_full, lrs_full, opt_deadline)
        except Exception:
            continue
        v = _max_cos(Xopt)
        if v < best_val:
            best_val = v
            best = Xopt.copy()
        # two kick cycles
        cur = Xopt.copy()
        for ki in range(2):
            if time.time() > opt_deadline:
                break
            try:
                Xk = _void_kick(cur, n_kick=6, rng=np.random.default_rng(1000 + si * 10 + ki))
                Xk2 = _adam_lse(Xk, betas_short, iters_short, lrs_short, opt_deadline)
            except Exception:
                break
            v2 = _max_cos(Xk2)
            if v2 < v:
                v = v2
                cur = Xk2.copy()
                if v2 < best_val:
                    best_val = v2
                    best = Xk2.copy()
            else:
                # keep kicked diversity once even if not better, try second kick from worse basin
                if ki == 0 and (v2 < best_val + 0.004):
                    cur = Xk2.copy()
                else:
                    break
    # final polish of best with remaining time
    try:
        best = _lp_polish(best, deadline)
        best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm(best)
    # tiny shrink guard against NaN/zero
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best

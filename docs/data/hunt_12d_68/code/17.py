import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def smooth_descent(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.25:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def worst_index(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return int(np.argmax(G.max(axis=1)))

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def best_hole(x_rest, rng, ncand=8000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, betas, maxiter_each=180):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    x = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).copy()
    for b in betas:
        fg = make_fg(n, d, IU0, float(b))
        y0 = x.ravel().copy()
        try:
            r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter_each*2})
            Y = r.x.reshape(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            x = Y
        except Exception:
            break
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x_init = rng0.standard_normal((68, 12))
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    else:
        bv = 1e9
        x_init = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bv:
                bv = v
                x_init = c.copy()
    x_init = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    best_val = max_cos(x_init)
    x_best = x_init.copy()
    n, d = x_best.shape
    rng = np.random.default_rng(777001)
    # elite pool for diversity
    pool = [(best_val, x_best.copy())]
    POOL_SIZE = 5
    def pool_add(v, x):
        for _, px in pool:
            if np.linalg.norm(x - px) < 1e-3 or np.linalg.norm(x + px) < 1e-3:
                return
        pool.append((v, x.copy()))
        pool.sort(key=lambda t: t[0])
        while len(pool) > POOL_SIZE:
            pool.pop()
    def pool_pick():
        if len(pool) == 1 or rng.random() < 0.7:
            return x_best.copy()
        else:
            k = int(rng.integers(0, len(pool)))
            return pool[k][1].copy()

    explore_end = t0 + TLIM - 30.0
    trial = 0
    no_improve = 0
    sigmas = [0.006, 0.015, 0.035, 0.08, 0.18, 0.35]
    while time.time() < explore_end and trial < 600:
        trial += 1
        x = pool_pick()
        u = rng.random()
        # adapt sigma upward if stuck
        boost = 1.0
        if no_improve > 25:
            boost = 1.8
        if u < 0.24:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))] * boost
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=650, beta0=60.0, beta1=380.0, lr0=0.03, lr1=0.003)
        elif u < 0.44:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(rng.integers(2, 8))
            # mix worst and random
            idx = set()
            nw = min(k, 3)
            for j in range(nw):
                idx.add(int(order[j]))
            while len(idx) < k:
                idx.add(int(rng.integers(0, n)))
            idx = np.array(list(idx), dtype=int)
            if rng.random() < 0.6:
                for q in idx:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    x[q] = v
            else:
                x[idx] += 0.6 * boost * rng.standard_normal((len(idx), d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=750, beta0=80.0, beta1=450.0, lr0=0.03, lr1=0.002)
        elif u < 0.62:
            # block re-init around random center
            c0 = int(rng.integers(0, n))
            dots = x @ x[c0]
            order = np.argsort(-dots)
            k = int(rng.integers(3, 9))
            blk = order[:k]
            for q in blk:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                x[q] = v
            # slight shake of shell
            shell = order[k:k+4]
            x[shell] += 0.03 * rng.standard_normal((len(shell), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=750, beta0=70.0, beta1=420.0, lr0=0.03, lr1=0.0025)
        elif u < 0.78:
            # cap jitter: random direction, move cap coherently + noise
            ud = rng.standard_normal(d)
            ud /= np.linalg.norm(ud)
            proj = x @ ud
            k = int(rng.integers(3, 10))
            order = np.argsort(-proj)
            cap = order[:k]
            # random tangent shift
            shift = rng.standard_normal(d)
            shift = shift - (shift @ ud) * ud
            shift /= (np.linalg.norm(shift) + 1e-12)
            amp = float(rng.uniform(0.1, 0.5)) * boost
            for q in cap:
                x[q] = x[q] + amp * shift + 0.08 * rng.standard_normal(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=700, beta0=70.0, beta1=430.0, lr0=0.028, lr1=0.002)
        else:
            wi = worst_index(x)
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            hole = best_hole(x[mask], rng, ncand=9000)
            x[wi] = hole
            G = x @ x.T
            nbrs = np.argsort(-G[wi])[:6]
            x[nbrs] += 0.025 * boost * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, steps=750, beta0=80.0, beta1=450.0, lr0=0.025, lr1=0.002)
        # high-beta fine tune
        x = smooth_descent(x, steps=220, beta0=450.0, beta1=700.0, lr0=0.008, lr1=0.001)
        v = max_cos(x)
        if v < best_val - 1e-12:
            # squeeze with L-BFGS then maybe LP
            x2 = run_lbfgs(x, betas=[150.0, 400.0, 900.0], maxiter_each=150)
            v2 = max_cos(x2)
            if v2 < v:
                x, v = x2, v2
            remain = explore_end - time.time()
            if remain > 4.0 and v < best_val - 2e-7:
                x = lp_polish(x, min(4.0, remain - 0.5))
                v = max_cos(x)
            if v < best_val:
                x_best = x.copy()
                best_val = v
                pool_add(v, x)
                no_improve = 0
                continue
        # near-miss handling
        if v < best_val + 0.003:
            pool_add(v, x)
        if v < best_val + 0.0015 and (trial % 5 == 0) and (time.time() < explore_end - 4.0):
            y = lp_polish(x, 1.4)
            v2 = max_cos(y)
            if v2 < best_val:
                x_best = y.copy()
                best_val = v2
                pool_add(v2, y)
                no_improve = 0
                continue
            elif v2 < best_val + 0.003:
                pool_add(v2, y)
        elif v < best_val + 0.0008 and (trial % 7 == 0):
            # occasional L-BFGS on promising basin
            x2 = run_lbfgs(x, betas=[300.0, 800.0], maxiter_each=120)
            v2 = max_cos(x2)
            if v2 < best_val:
                x_best = x2.copy()
                best_val = v2
                pool_add(v2, x2)
                no_improve = 0
                continue
            if v2 < best_val + 0.003:
                pool_add(v2, x2)
        no_improve += 1
        if time.time() > explore_end:
            break
    # final polish: polish best and top elites, keep best
    remain = TLIM - (time.time() - t0)
    if remain > 4.0:
        # short L-BFGS on best before LP
        try:
            x_best = run_lbfgs(x_best, betas=[200.0, 600.0, 1200.0], maxiter_each=200)
            best_val = max_cos(x_best)
        except Exception:
            pass
        remain = TLIM - (time.time() - t0)
        if remain > 3.0:
            x_best = lp_polish(x_best, min(remain - 1.0, 24.0))
            best_val = max_cos(x_best)
        # try polishing next elites briefly if time left
        remain = TLIM - (time.time() - t0)
        if remain > 5.0 and len(pool) > 1:
            for _, px in sorted(pool, key=lambda t: t[0])[:3]:
                if time.time() - t0 > TLIM - 2.0:
                    break
                try:
                    y = lp_polish(px, min(2.0, TLIM - (time.time() - t0) - 1.0))
                    vy = max_cos(y)
                    if vy < best_val:
                        x_best = y.copy()
                        best_val = vy
                except Exception:
                    continue
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

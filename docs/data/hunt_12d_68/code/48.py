import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def quench_full(x, steps, beta0, beta1, lr0, lr1, mu=0.85):
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        Tg = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        Tn = Tg / rms
        vel = mu * vel + (1.0 - mu) * Tn
        vc = vel / (1.0 - mu ** (s + 1))
        x = x - lr * vc
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def make_full_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_full_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = make_full_fg(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def make_subset_fg(C, K, d, beta):
    m = C.shape[0]
    def fg(yf):
        Y = yf.reshape(K, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        Gac = X @ C.T  # K x m
        Gbb = X @ X.T
        iu = np.triu_indices(K, 1)
        gbb = Gbb[iu]
        gac = Gac.ravel()
        # max over both
        m0 = float(max(float(gac.max()), float(gbb.max()) if gbb.size else -2.0))
        eac = np.exp(beta * (gac - m0))
        ebb = np.exp(beta * (gbb - m0)) if gbb.size else np.zeros(0)
        S = float(eac.sum() + ebb.sum()) + 1e-300
        F = m0 + np.log(S) / beta
        wac = (eac / S).reshape(K, m)
        GX = wac @ C
        if gbb.size:
            wbb = np.zeros((K, K))
            wbb[iu] = ebb / S
            wbb = wbb + wbb.T
            GX = GX + wbb @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_subset_lbfgs(Y0, C, beta, maxiter):
    K, d = Y0.shape
    fg = make_subset_fg(C, K, d, beta)
    y0 = (Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(K, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return Y0

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.2:
            break
        if deadline is not None and time.time() > deadline - 0.5:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        if int(sel.sum()) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                yy = x + res.x[:-1].reshape(n, d) * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def greedy_refill(pool, core, K):
    M = pool.shape[0]
    # core scores: max dot to core, chunked
    chunk = 20000
    scores = np.empty(M, dtype=float)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        scores[s:e] = (pool[s:e] @ core.T).max(axis=1)
    sel = []
    used = np.zeros(M, dtype=bool)
    for _ in range(K):
        # argmin among unused
        # mask used with large value
        idx = int(np.argmin(np.where(used, 2.0, scores)))
        sel.append(idx)
        used[idx] = True
        if len(sel) >= K:
            break
        d = pool @ pool[idx]
        # update scores = max(scores, d) for unused
        np.maximum(scores, d, out=scores)
        scores[idx] = 2.0
    return pool[np.array(sel)].copy()

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline = t0 + TLIM
    n, d = 68, 12
    rng = np.random.default_rng(7)
    # load incumbent
    try:
        xb0 = np.load("best.npy").astype(float)
        xb0 = xb0 / np.linalg.norm(xb0, axis=1, keepdims=True)
        if xb0.shape != (68, 12):
            xb0 = rng.standard_normal((n, d))
            xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
    except Exception:
        xb0 = rng.standard_normal((n, d))
        xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
    x_best = xb0.copy()
    best_val = max_cos(x_best)
    # candidate pool 200k Gaussian normalized
    MPOOL = 200000
    pool = rng.standard_normal((MPOOL, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    # optional: mix in some Sobol-like uniform? keep Gaussian

    # farthest-point seed order covering distinct geography
    # start from point 0, iteratively pick farthest (min dot) from set
    G0 = x_best @ x_best.T
    seed_order = [0]
    best_min = G0[0].copy()
    for _ in range(n - 1):
        # pick index not in order maximizing distance = minimizing max dot to chosen
        cand = [(i, float(best_min[i])) for i in range(n) if i not in seed_order]
        # smallest max-dot = farthest
        nxt = min(cand, key=lambda t: t[1])[0]
        seed_order.append(nxt)
        best_min = np.minimum(best_min, G0[nxt])
    # keep first ~12 distinct seeds spread out
    seeds_cycle = seed_order[:12]

    Ks = [14, 16, 18, 20, 22, 15, 17, 19, 21, 14, 18, 22]
    trial = 0
    # main surgery loop
    while time.time() < deadline - 12.0:
        for si, seed in enumerate(seeds_cycle):
            if time.time() > deadline - 12.0:
                break
            K = int(Ks[(trial) % len(Ks)])
            # find K nearest neighbours in cosine to seed (including seed)
            dots = x_best @ x_best[seed]
            Didx = np.argsort(-dots)[:K]
            mask = np.ones(n, dtype=bool)
            mask[Didx] = False
            C = x_best[mask]  # (n-K, d)
            # refill hole greedily
            try:
                Y0 = greedy_refill(pool, C, K)
            except Exception:
                continue
            # staged subset L-BFGS beta 100,300,900
            try:
                Y = run_subset_lbfgs(Y0, C, 100.0, 60)
                Y = run_subset_lbfgs(Y, C, 300.0, 60)
                Y = run_subset_lbfgs(Y, C, 900.0, 50)
            except Exception:
                Y = Y0
            Xt = np.empty_like(x_best)
            Xt[mask] = C
            Xt[Didx] = Y
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # short full anneal (unfreeze)
            try:
                Xt = quench_full(Xt, 160, 60.0, 350.0, 0.04, 0.004, mu=0.85)
            except Exception:
                pass
            # brief full L-BFGS
            if time.time() < deadline - 10.0:
                try:
                    Xt = run_full_lbfgs(Xt, 300.0, 40)
                except Exception:
                    pass
            # LP polish short
            rem = deadline - time.time()
            if rem > 8.0:
                try:
                    Xt = lp_polish(Xt, min(1.6, rem - 7.0), deadline=deadline - 6.0)
                except Exception:
                    pass
            v = max_cos(Xt)
            if v < best_val - 1e-12:
                x_best = Xt.copy()
                best_val = v
                # recompute seed order around new best for next cycle
                try:
                    Gn = x_best @ x_best.T
                    seed_order2 = [int(seed)]
                    bm = Gn[seed].copy()
                    for _ in range(n - 1):
                        cand2 = [(i, float(bm[i])) for i in range(n) if i not in seed_order2]
                        nxt2 = min(cand2, key=lambda t: t[1])[0]
                        seed_order2.append(nxt2)
                        bm = np.minimum(bm, Gn[nxt2])
                    seeds_cycle = seed_order2[:12]
                except Exception:
                    pass
            trial += 1
            if trial > 60:
                break
        if trial > 60:
            break
        # if no progress for a full cycle, perturb pool slightly to diversify
        if time.time() < deadline - 12.0:
            # refresh a slice of pool to keep greedy diverse
            try:
                ns = 20000
                ii = rng.choice(MPOOL, size=ns, replace=False)
                nw = rng.standard_normal((ns, d))
                nw /= np.linalg.norm(nw, axis=1, keepdims=True)
                pool[ii] = nw
            except Exception:
                pass

    # final intensive polish on winner
    remain = deadline - time.time()
    if remain > 5.0:
        try:
            x_best = run_full_lbfgs(x_best, 700.0, 60)
            best_val = max_cos(x_best)
        except Exception:
            pass
    remain = deadline - time.time()
    if remain > 4.0:
        try:
            x_best = lp_polish(x_best, min(remain - 0.8, 10.0), deadline=deadline - 0.5)
            best_val = max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # compare to stored best.npy (in case our search degraded)
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if xb.shape == (68, 12):
            vb = max_cos(xb)
            if vb < best_val:
                return xb
    except Exception:
        pass
    return x_best

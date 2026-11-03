import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def greedy_reinsert(X, idx, rng, n_cand=4000):
    n, d = X.shape
    idx = list(idx)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx] = False
    base = X[keep_mask]
    new_pts = []
    for _ in range(len(idx)):
        if len(new_pts) == 0:
            others = base
        else:
            others = np.vstack([base] + new_pts)
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ others.T
        md = dots.max(axis=1)
        b = int(np.argmin(md))
        new_pts.append(C[b:b + 1])
    Xn = X.copy()
    for ii, pt in zip(idx, new_pts):
        Xn[ii] = pt[0]
    return Xn


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        ns = int(sel.sum())
        if ns == 0:
            break
        if ns > 1300:
            idx = np.argpartition(c, -1300)[-1300:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
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


def solve():
    t0 = time.time()
    BUDGET = 174.0
    deadline = t0 + BUDGET
    EXPL_END = t0 + 136.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if len(cands) == 0:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    # short initial squeeze to get true local value if loaded point not squeezed
    # (cheap, 2 sec) - helps baseline
    if time.time() < EXPL_END:
        x_best = lp_polish(x_best, min(3.0, max(0.5, EXPL_END - time.time() - 100.0)))
        best_val = max_cos(x_best)

    cur = x_best.copy()
    cur_val = best_val
    n, d = 95, 13
    sigmas = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
    fails = 0
    trial = 0

    while time.time() < EXPL_END:
        trial += 1
        remain = EXPL_END - time.time()
        if remain < 0.8:
            break
        # acceptance temperature anneal
        prog = (time.time() - t0) / max(1e-9, (EXPL_END - t0))
        Tacc = 0.003 * (1.0 - prog) + 0.0002

        u = rng.random()
        Xt = cur.copy()
        if u < 0.42:
            sigma = float(sigmas[int(rng.integers(0, len(sigmas)))])
            if rng.random() < 0.25:
                ksub = int(rng.integers(4, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        elif u < 0.62:
            # subset stronger shake around worst points
            ppm = per_point_max(cur)
            order = np.argsort(-ppm)
            ksub = int(rng.integers(3, 10))
            pool = order[:max(15, ksub + 8)]
            idx = rng.choice(pool, size=min(ksub, len(pool)), replace=False)
            sigma = float(sigmas[int(rng.integers(0, len(sigmas)))]) * 1.5 + 0.01
            Xt[idx] += sigma * rng.standard_normal((len(idx), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        else:
            # kill and reinsert worst points into holes
            r = rng.random()
            if r < 0.30:
                k = 1
            elif r < 0.60:
                k = 2
            elif r < 0.85:
                k = int(rng.integers(3, 5))
            else:
                k = int(rng.integers(5, 8))
            ppm = per_point_max(cur)
            order = np.argsort(-ppm)
            idx = []
            for _ in range(k):
                if rng.random() < 0.72:
                    # pick among top-20 worst not already picked
                    for o in order:
                        if int(o) not in idx and (len(idx) < 20 or rng.random() < 0.3):
                            # prefer top 20
                            if len(idx) < 20:
                                pass
                            idx.append(int(o))
                            break
                    if len(idx) == 0 or len(idx) > k:
                        pass
                else:
                    c = int(rng.integers(0, n))
                    if c not in idx:
                        idx.append(c)
                    else:
                        c2 = int(rng.integers(0, n))
                        if c2 not in idx:
                            idx.append(c2)
                if len(idx) >= k:
                    break
            # fill if short
            while len(idx) < k:
                c = int(rng.integers(0, n))
                if c not in idx:
                    idx.append(c)
            idx = idx[:k]
            try:
                Xt = greedy_reinsert(cur, idx, rng, n_cand=4000)
            except Exception:
                Xt = cur.copy()
                Xt[idx] = rng.standard_normal((len(idx), d)) / np.linalg.norm(rng.standard_normal((len(idx), d)), axis=1, keepdims=True)

        steps = int(rng.integers(500, 1150))
        T0 = float(10.0 ** rng.uniform(-2.2, -1.7))
        T1 = float(10.0 ** rng.uniform(-3.5, -2.65))
        lr0 = float(10.0 ** rng.uniform(-1.3, -0.8))
        lr1 = lr0 * float(10.0 ** rng.uniform(-1.2, -0.4))
        mu = float(rng.uniform(0.80, 0.93))
        try:
            Xt = gd_refine(Xt, steps, T0, T1, lr0, lr1, mu)
        except Exception:
            continue
        v = max_cos(Xt)

        improved_best = False
        if v < best_val:
            # confirm with short LP squeeze (1s) to get true basin value
            v2 = v
            X2 = Xt
            try:
                X2 = lp_polish(Xt, 1.0, deadline=EXPL_END)
                v2 = max_cos(X2)
            except Exception:
                pass
            if v2 < best_val:
                x_best = X2.copy()
                best_val = float(v2)
                cur = X2.copy()
                cur_val = float(v2)
                fails = 0
                improved_best = True
            else:
                # still take GD point as cur if better than cur
                if v < cur_val:
                    cur = Xt.copy()
                    cur_val = float(v)
                    fails = 0
                else:
                    fails += 1
        else:
            if v < cur_val or rng.random() < np.exp((cur_val - v) / max(1e-12, Tacc)):
                cur = Xt.copy()
                cur_val = float(v)
                fails = 0
            else:
                fails += 1
            # restart wandering from best if stuck long
            if fails > 35:
                cur = x_best.copy()
                # kick it
                sigma = 0.05 + 0.05 * rng.random()
                cur += sigma * rng.standard_normal((n, d))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = max_cos(cur)
                fails = 0
        # keep time guard
        if time.time() > EXPL_END - 0.2:
            break

    # final long polish of best (and cur if competitive)
    frem = deadline - time.time() - 1.0
    if frem > 2.0:
        # polish best mainly
        half = frem * 0.75
        try:
            x_best = lp_polish(x_best, half, deadline=deadline - 1.0)
            best_val = max_cos(x_best)
        except Exception:
            pass
        # if cur is close, polish it and take better
        try:
            if cur_val < best_val + 0.002 and frem * 0.25 > 2.0:
                xc = lp_polish(cur, frem * 0.25, deadline=deadline - 0.5)
                vc = max_cos(xc)
                if vc < best_val:
                    x_best = xc
                    best_val = vc
        except Exception:
            pass
        # final tiny shake-polish if time left
        left = deadline - time.time() - 0.5
        if left > 3.0:
            try:
                x_best = lp_polish(x_best, left, deadline=deadline - 0.2)
            except Exception:
                pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict shrink guard (no-op for cosine but ensures finite)
    x_best = np.where(np.isfinite(x_best), x_best, rng.standard_normal((95, 13)))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

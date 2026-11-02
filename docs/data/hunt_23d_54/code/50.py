import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, t0=None, deadline=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t0 is not None and (time.time() - t0) > deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _riemann_adam_refine(V_init, F, steps=500, K0=80.0, K1=2500.0, lr0=0.03, lr1=0.0015):
    V = V_init.copy()
    k = V.shape[0]
    m1 = np.zeros_like(V)
    m2 = np.zeros_like(V)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lk0, lk1 = np.log(K0), np.log(K1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        K = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        D1 = V @ F.T
        D2 = V @ V.T
        np.fill_diagonal(D2, -2.0)
        mall = D1.max()
        m2v = D2.max()
        if m2v > mall:
            mall = m2v
        E1 = np.exp(K * (D1 - mall))
        E2 = np.exp(K * (D2 - mall))
        np.fill_diagonal(E2, 0.0)
        Grad = E1 @ F + E2 @ V
        dot = np.sum(Grad * V, axis=1, keepdims=True)
        Gt = Grad - dot * V
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = m2 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        V = V - lr * step
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        V /= nr
    return V

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(7)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            vr = _max_cos(xr)
            if vr < best:
                best, x_best = vr, xr.copy()
    except Exception:
        pass

    archive = set()
    it = 0
    P = 200000
    CHUNK = 50000
    # main surgery cycles: no full LP inside except conditional
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        remaining = BUDGET - 12.0 - (time.time() - t0)
        if remaining < 3.0:
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        permax = G.max(axis=1)
        cnt = (G > best - 0.015).sum(axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-permax)
        tot = float(cnt.sum()) + 1e-12
        prob = cnt.astype(float) / tot
        prob = 0.2 / n + 0.8 * prob
        prob /= prob.sum()
        mode = it % 3
        if mode == 0:
            # dense contact clique around high-contact seed
            k = int(rng.randint(5, 9))
            seed_pool = order_worst[:15]
            seed = int(seed_pool[rng.randint(0, len(seed_pool))])
            nbrs = np.argsort(-G[seed])
            nbrs = nbrs[nbrs != seed][:k - 1]
            S = [seed] + [int(v) for v in nbrs]
        elif mode == 1:
            # lowest-contact valley including 17,23,26,27
            k = int(rng.randint(5, 9))
            lowpool = order_lo[:12].tolist()
            S = [int(v) for v in rng.choice(lowpool, size=k, replace=False)]
            if rng.rand() < 0.6:
                forced = int(rng.choice([17, 23, 26, 27]))
                if forced not in S:
                    S[rng.randint(0, k)] = forced
        else:
            k = int(rng.randint(5, 9))
            S = [int(v) for v in rng.choice(n, size=k, replace=False, p=prob)]
        key = tuple(sorted(S))
        if key in archive:
            continue
        archive.add(key)
        Sarr = np.array(S, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[Sarr] = False
        F = x_best[mask]
        # --- massive random pool (>=200k) ---
        try:
            pool = rng.randn(P, d)
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        except Exception:
            continue
        # frozen scores for pool
        try:
            fmax = np.empty(P)
            for a in range(0, P, CHUNK):
                b = min(a + CHUNK, P)
                dots = pool[a:b] @ F.T
                fmax[a:b] = dots.max(axis=1)
        except Exception:
            continue
        cur = fmax.copy()
        picked = []
        # greedy joint construction minimizing max(frozen, mutual)
        for _ in range(k):
            idx = int(np.argmin(cur))
            picked.append(idx)
            try:
                dvec = pool @ pool[idx]
            except Exception:
                break
            cur = np.maximum(cur, dvec)
            cur[idx] = np.inf
            for q in picked:
                cur[q] = np.inf
        if len(picked) != k:
            continue
        V0 = pool[np.array(picked)].copy()
        # --- Riemannian Adam candidates against frozen + each other ---
        try:
            # vary hyperparams slightly per mode for diversity
            if mode == 0:
                Vopt = _riemann_adam_refine(V0, F, steps=450, K0=60.0, K1=2200.0, lr0=0.03, lr1=0.0012)
            elif mode == 1:
                Vopt = _riemann_adam_refine(V0, F, steps=500, K0=90.0, K1=3000.0, lr0=0.025, lr1=0.001)
            else:
                Vopt = _riemann_adam_refine(V0, F, steps=400, K0=70.0, K1=2000.0, lr0=0.035, lr1=0.0015)
        except Exception:
            continue
        x_cand = x_best.copy()
        x_cand[Sarr] = Vopt / np.linalg.norm(Vopt, axis=1, keepdims=True)
        vs = _max_cos(x_cand)
        # abandon if not competitive (cheap smooth score); allow subset only seconds
        if vs > best + 0.002 and vs >= best:
            continue
        # LP-polish only competitive candidates
        lp_sec = 1.2
        if vs < best:
            lp_sec = 2.0
        if remaining < 15:
            lp_sec = min(lp_sec, 0.8)
        try:
            xp = _lp_polish(x_cand, lp_sec, t0, BUDGET - 10.0)
        except Exception:
            continue
        vp = _max_cos(xp)
        if vp < best - 1e-12:
            best = vp
            x_best = xp.copy()
    try:
        left = BUDGET - (time.time() - t0)
        if left > 2.5:
            x_best = _lp_polish(x_best, min(left - 0.6, 9.0), t0, BUDGET - 0.4)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

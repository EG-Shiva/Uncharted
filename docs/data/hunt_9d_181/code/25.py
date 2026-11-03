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
    return float(np.max(G))

def _anneal(X, steps, T0=0.015, T1=0.002, lr0=0.08, lr1=0.02, mu=0.6, t_end=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    for s in range(steps):
        if t_end is not None and s % 20 == 0 and time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ X
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        Dir = WY * inv[:, None] - (r * inv)[:, None] * X
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        # normalize direction scale
        sc = np.mean(np.linalg.norm(V, axis=1)) + 1e-12
        X = X - (lr / max(1.0, sc * 5.0)) * V
        X = _norm(X)
    return X

def _best_hole(Xk, rng, n_cand=20000):
    d = Xk.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Xk.T
    vals = np.max(dots, axis=1)
    bi = int(np.argmin(vals))
    v = cand[bi].copy()
    # local refinement: push away from neighbours
    for _ in range(40):
        dv = Xk @ v
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xk
        g = g - np.dot(g, v) * v
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.02 * g
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    return v

def _grow(X0, rng, target=181, relax_steps=180, t_end=None, snap_at=165):
    X = X0.copy()
    snap = None
    while X.shape[0] < target:
        if t_end is not None and time.time() > t_end:
            break
        v = _best_hole(X, rng, n_cand=20000)
        X = np.vstack([X, v[None, :]])
        m = X.shape[0]
        if m == snap_at:
            snap = X.copy()
        if m % 5 == 0 or m == target:
            X = _anneal(X, steps=relax_steps, t_end=t_end)
            X = _norm(X)
    if snap is None and X.shape[0] >= snap_at:
        snap = X[:snap_at].copy()
    return X, snap

def _sorted_gram_vec(X):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    v = G[iu]
    v.sort()
    return v

def _lp_polish(X, budget):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.3 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        if int(np.sum(sel)) == 0:
            radius *= 0.5
            continue
        if int(np.sum(sel)) > 3000:
            kth = np.partition(c, -3000)[-3000]
            sel = c >= kth
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _greedy_select(pool, k, rng):
    P = pool.shape[0]
    start = int(rng.integers(0, P))
    sel = [start]
    maxd = pool @ pool[start]
    maxd[start] = np.inf
    in_sel = np.zeros(P, dtype=bool); in_sel[start] = True
    for _ in range(1, k):
        # pick min of maxd among not selected
        tmp = maxd.copy(); tmp[in_sel] = np.inf
        bi = int(np.argmin(tmp))
        sel.append(bi); in_sel[bi] = True
        d = pool @ pool[bi]
        maxd = np.maximum(maxd, d)
        maxd[bi] = np.inf
    return pool[np.array(sel)]

def solve():
    t0 = time.time()
    T_LIM = 155.0
    def remain(): return T_LIM - (time.time() - t0)
    n, d, target = 181, 9, 181
    rng = np.random.default_rng(12345)
    # fallback best
    try:
        fb = np.load("best.npy")
        fb = np.asarray(fb, dtype=float)
        if fb.shape == (181, 9):
            fb = _norm(fb)
        else:
            fb = None
    except Exception:
        fb = None
    fb_val = _max_cos(fb) if fb is not None else np.inf

    # ---- build 150-point seeds ----
    seeds = []
    # A: random
    a = _norm(rng.standard_normal((150, d)))
    a = _anneal(a, steps=250, t_end=t0 + T_LIM - 10)
    seeds.append(a)
    # B: D9 shells + random fill
    D = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(d); v[i] = s1; v[j] = s2
                    D.append(v)
    D = _norm(np.array(D))  # 144
    extra = _norm(rng.standard_normal((6, d)))
    b = np.vstack([D, extra])
    b = b + rng.standard_normal(b.shape) * 0.05
    b = _norm(b)
    b = _anneal(b, steps=250, t_end=t0 + T_LIM - 10)
    seeds.append(b)
    # C: E8 x R subset
    try:
        e8 = []
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in (1.0, -1.0):
                    for s2 in (1.0, -1.0):
                        v = np.zeros(8); v[i] = s1; v[j] = s2
                        e8.append(v)
        for mask in range(256):
            v = np.ones(8) * 0.5
            neg = 0
            for k in range(8):
                if (mask >> k) & 1:
                    v[k] = -0.5; neg += 1
            if neg % 2 == 0:
                e8.append(v.copy())
        e8 = np.array(e8)  # 240x8
        e8 = e8 / np.linalg.norm(e8[:1])  # sqrt2
        E = np.zeros((e8.shape[0], 9)); E[:, :8] = e8
        E[:, 8] = rng.standard_normal(E.shape[0]) * 0.35
        E = _norm(E)
        # greedy low-correlation subset of 150
        c = _greedy_select(E, 150, rng)
        c = _anneal(c, steps=200, t_end=t0 + T_LIM - 10)
        seeds.append(c)
    except Exception:
        pass
    # D: greedy from large random pool
    if remain() > 30:
        pool = _norm(rng.standard_normal((3000, d)))
        e = _greedy_select(pool, 150, rng)
        e = _anneal(e, steps=200, t_end=t0 + T_LIM - 10)
        seeds.append(e)

    grown = []
    snaps = []
    for si, s0 in enumerate(seeds):
        if remain() < 25:
            break
        # use different rng stream per seed
        r2 = np.random.default_rng(1000 + si)
        Xf, sn = _grow(s0, r2, target=target, relax_steps=170, t_end=t0 + T_LIM - 30)
        if Xf.shape[0] != target:
            # relax what we have and pad if needed
            Xf = _norm(Xf)
            while Xf.shape[0] < target:
                v = _best_hole(Xf, r2, n_cand=8000)
                Xf = np.vstack([Xf, v[None, :]])
            Xf = _anneal(Xf, steps=120, t_end=t0 + T_LIM - 25)
        grown.append(Xf)
        if sn is not None:
            snaps.append(sn)

    # ---- delete-and-regrow variants from best 165 snapshot ----
    if grown and snaps and remain() > 30:
        vals = [_max_cos(g) for g in grown]
        bi = int(np.argmin(vals))
        base_snap = snaps[min(bi, len(snaps) - 1)]
        for vi in range(2):
            if remain() < 22:
                break
            r3 = np.random.default_rng(777 + vi)
            # delete 20 points: variant0 random, variant1 high-contact biased
            m0 = base_snap.shape[0]
            if vi == 0:
                idx = r3.choice(m0, size=m0 - 20, replace=False)
                idx.sort()
                Xd = base_snap[idx]
            else:
                G = base_snap @ base_snap.T
                np.fill_diagonal(G, -2)
                mc = float(np.max(G))
                cnt = np.sum(G > mc - 1e-3, axis=1)
                order = np.argsort(cnt)  # keep low-contact, drop high
                # drop 15 highest + 5 random among rest
                drop = set(order[-15:].tolist())
                while len(drop) < 20:
                    drop.add(int(r3.integers(0, m0)))
                keep = [i for i in range(m0) if i not in drop]
                Xd = base_snap[np.array(keep)]
            Xf2, _ = _grow(Xd, r3, target=target, relax_steps=150, t_end=t0 + T_LIM - 22)
            if Xf2.shape[0] == target:
                grown.append(Xf2)

    if not grown:
        return fb if fb is not None else _norm(rng.standard_normal((181, 9)))

    # ---- diversity filter + pick two most promising distinct ----
    vals = [_max_cos(g) for g in grown]
    order = np.argsort(vals)
    vecs = [_sorted_gram_vec(g) for g in grown]
    picked = []
    for oi in order:
        ok = True
        for pj in picked:
            # rotation+permutation invariant distance via sorted Gram
            dd = float(np.mean((vecs[oi] - vecs[pj]) ** 2))
            if dd < 1e-8:  # near-duplicate
                ok = False
                break
        if ok:
            picked.append(oi)
        if len(picked) >= 2:
            break
    if not picked:
        picked = [int(order[0])]
    # also discard near-duplicates of record: if grown is essentially same sorted spectrum as fb, skip LP? keep but prefer other
    # LP polish only two
    bestX, bestV = None, np.inf
    # include fallback in comparison but don't waste LP on it
    for k, pi in enumerate(picked[:2]):
        if remain() < 6:
            break
        Xp = _lp_polish(grown[pi], budget=min(9.0, remain() - 2.0))
        vp = _max_cos(Xp)
        if vp < bestV:
            bestV, bestX = vp, Xp.copy()
    # also consider unpolished grown (in case LP hurt due to time)
    for g, v in zip(grown, vals):
        if v < bestV:
            bestV, bestX = v, g.copy()
    if fb_val < bestV:
        return fb.copy()
    # tiny shrink to enforce strict feasibility / avoid NaN
    bestX = _norm(bestX)
    return bestX

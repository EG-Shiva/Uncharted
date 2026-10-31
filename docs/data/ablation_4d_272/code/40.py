import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, float(rm.max())

def _repulse(x, steps, a0, a1, s0, s1, mu=0.85, b0=None, b1=None, lam=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0.0))
        np.fill_diagonal(W, 0.0)
        if lam != 0.0 and b0 is not None:
            beta = b0 * (b1 / b0) ** frac if b0 != b1 else b0
            W2 = np.exp(np.clip(beta * (G - m), -50, 0.0))
            np.fill_diagonal(W2, 0.0)
            # balance: row-normalize second term to share load
            rs = W2.sum(axis=1, keepdims=True) + 1e-12
            W2 = W2 / rs
            W = W / (float(W.sum() * 0.5) + 1e-12) + lam * W2
            grad = W @ x
        else:
            s = float(W.sum() * 0.5)
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            grad = (W @ x) / s
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm(x)
    return x

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5500:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5500 - 1)[:5500]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
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
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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

def _largest_hole(Y, rng, ns=10000):
    d = Y.shape[1]
    C = rng.standard_normal((ns, d))
    C = _norm(C)
    best_s = 2.0
    best_c = C[0].copy()
    bs = 2500
    for a in range(0, ns, bs):
        Cb = C[a:a+bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = Cb[k].copy()
    # refine single point
    c = best_c / np.linalg.norm(best_c)
    vel = np.zeros(4)
    s = 0.03
    for _ in range(120):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel):
            g = Y[sel].sum(axis=0)
        else:
            g = Y[int(np.argmax(dots))]
        gt = g - np.dot(g, c) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt /= nn
        vel = 0.5 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn > 1e-14:
            c = c - (s / vn) * vel
            c = c / np.linalg.norm(c)
            s *= 0.995
    return c / np.linalg.norm(c)

def _hopf_grid(rng, n, jitter=0.0):
    # S3 via (cos p e^{ia}, sin p e^{ib})
    X = np.empty((n, 4))
    for i in range(n):
        # stratified phi + random angles with golden offset for diversity
        phi = np.arccos(1 - 2 * ((i + 0.5) / n + rng.random() * 0.0))
        # use random torus angles
        a = rng.random() * 2 * np.pi
        b = rng.random() * 2 * np.pi
        cp = np.cos(phi / 2); sp = np.sin(phi / 2)
        X[i, 0] = cp * np.cos(a); X[i, 1] = cp * np.sin(a)
        X[i, 2] = sp * np.cos(b); X[i, 3] = sp * np.sin(b)
    if jitter > 0:
        X = X + rng.standard_normal(X.shape) * jitter
    return _norm(X)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    rng = np.random.default_rng(20261031)
    n, d = 272, 4
    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [(_max_cos(c), c) for c in cands]
    vals.sort(key=lambda t: t[0])
    best_v, best_x = vals[0][0], vals[0][1].copy()
    pool = [(best_v, best_x.copy())]
    # add hopf + random diverse starts quickly annealed
    iu_all = np.triu_indices(n, 1)
    # Phase 1: diverse seeds (fast)
    seeds = []
    seeds.append(_norm(rng.standard_normal((n, d))))
    seeds.append(_hopf_grid(rng, n))
    seeds.append(_norm(best_x + rng.standard_normal(best_x.shape) * 0.25))
    seeds.append(_norm(rng.standard_normal((n, d))))
    for si, s0 in enumerate(seeds):
        if time.time() > T_END - 140.0:
            break
        x = s0.copy()
        # two-stage anneal with dual term to balance load
        x = _repulse(x, 500, 12.0, 90.0, 0.03, 0.006, mu=0.82, b0=12.0, b1=12.0, lam=0.15)
        if time.time() > T_END - 138.0:
            # quick eval
            v = _max_cos(x)
            pool.append((v, x.copy()))
            break
        x = _repulse(x, 500, 90.0, 240.0, 0.008, 0.0012, mu=0.87)
        v = _max_cos(x)
        pool.append((v, x.copy()))
        if v < best_v:
            best_v, best_x = v, x.copy()
    pool.sort(key=lambda t: t[0])
    pool = pool[:6]

    LOOP_END = T_END - 20.0
    it = 0
    while time.time() < LOOP_END:
        it += 1
        remain = LOOP_END - time.time()
        if remain < 2.0:
            break
        # pick base: mostly best, sometimes diverse
        r = rng.random()
        if len(pool) == 1 or r < 0.55:
            base = pool[0][1]
        else:
            base = pool[rng.integers(len(pool))][1]
        u = rng.random()
        xp = None
        try:
            if u < 0.38:
                # small noise + anneal (basin exploitation, balanced)
                sigma = float(10.0 ** rng.uniform(-2.6, -1.2))
                xp = base + rng.standard_normal(base.shape) * sigma
                # targeted push on worst pair
                if rng.random() < 0.5:
                    G = base @ base.T
                    np.fill_diagonal(G, -2.0)
                    cg = G[iu_all]
                    k = 24
                    top = np.argpartition(-cg, k - 1)[:k]
                    ch = top[rng.integers(k)]
                    i = int(iu_all[0][ch]); j = int(iu_all[1][ch])
                    push = 0.5 * sigma + 0.004
                    xp[i] += push * (base[i] - base[j])
                    xp[j] += push * (base[j] - base[i])
                xp = _norm(xp)
                steps = 650 + int(rng.integers(0, 400))
                if remain < 30:
                    steps = min(steps, 450)
                a0 = 18.0 + 14.0 * rng.random()
                a1 = 130.0 + 120.0 * rng.random()
                lam = 0.1 + 0.2 * rng.random()
                xp = _repulse(xp, steps, a0, a1, 0.007 + 0.006 * rng.random(), 0.0006 + 0.0008 * rng.random(), mu=0.80 + 0.12 * rng.random(), b0=10.0, b1=14.0, lam=lam)
            elif u < 0.58:
                # medium/large shake to change basin
                sigma = float(10.0 ** rng.uniform(-1.0, -0.3))  # 0.1 .. 0.5
                xp = base + rng.standard_normal(base.shape) * sigma
                xp = _norm(xp)
                steps = 700 + int(rng.integers(0, 500))
                if remain < 30:
                    steps = min(steps, 500)
                xp = _repulse(xp, steps, 10.0, 160.0, 0.015, 0.001, mu=0.82, b0=10.0, b1=10.0, lam=0.2)
            elif u < 0.76:
                # delete-and-regrow 1-3 victims into largest holes
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm)
                nv = 1 + int(rng.integers(0, 3))  # 1-3
                # mostly worst, occasionally random high-degree
                victims = []
                for t in range(nv):
                    if rng.random() < 0.75 and t < len(order):
                        victims.append(int(order[(t + int(rng.integers(0, 4))) % n]))
                    else:
                        victims.append(int(rng.integers(n)))
                victims = list(dict.fromkeys(victims))
                xp = base.copy()
                keep = [i for i in range(n) if i not in set(victims)]
                Y = xp[keep]
                for v in victims:
                    c = _largest_hole(Y, rng, ns=8000)
                    xp[v] = c
                    Y = np.vstack([Y, c[None, :]])
                xp = _norm(xp)
                # local regrow polish on victims + neighbors then global
                xp = _repulse(xp, 600, 20.0, 170.0, 0.008, 0.0008, mu=0.84, b0=12.0, b1=12.0, lam=0.15)
            else:
                # collective cluster move: move k nearest together
                ci = int(rng.integers(n))
                dots = base @ base[ci]
                k = int(rng.integers(10, 30))
                cl = np.argpartition(-dots, k - 1)[:k]
                # random common tangent direction at center
                vdir = rng.standard_normal(4)
                vdir = vdir - np.dot(vdir, base[ci]) * base[ci]
                vn = float(np.linalg.norm(vdir))
                if vn < 1e-12:
                    vdir = np.array([1.0, 0, 0, 0])
                else:
                    vdir /= vn
                step = 0.05 + 0.20 * rng.random()
                xp = base.copy()
                for idx in cl:
                    # project direction to tangent at idx
                    tt = vdir - np.dot(vdir, xp[idx]) * xp[idx]
                    xp[idx] = xp[idx] + step * tt + rng.standard_normal(4) * 0.01
                # also jitter a few random outsiders lightly
                for _ in range(3):
                    q = int(rng.integers(n))
                    xp[q] += rng.standard_normal(4) * 0.02
                xp = _norm(xp)
                xp = _repulse(xp, 650, 15.0, 180.0, 0.009, 0.0009, mu=0.84, b0=10.0, b1=12.0, lam=0.18)
        except Exception:
            continue
        try:
            v = _max_cos(xp)
        except Exception:
            continue
        # pool management: keep best + diverse near-best
        if v < pool[0][0] - 1e-12:
            pool.insert(0, (v, xp.copy()))
            pool = sorted(pool, key=lambda t: t[0])[:6]
            if v < best_v:
                best_v, best_x = v, xp.copy()
        elif v < best_v + 0.004 and len(pool) < 6:
            # check diversity
            div = True
            for _, q in pool:
                if float(np.abs(xp * q).sum(axis=1).max()) > 0.9999 and _max_cos(0.5 * (xp + q)) < min(v, pool[0][0]) + 1e-6:
                    div = False
                    break
            if div:
                pool.append((v, xp.copy()))
                pool = sorted(pool, key=lambda t: t[0])[:6]
        elif rng.random() < 0.02 and v < best_v + 0.008:
            # occasional random injection to preserve exploration
            if len(pool) >= 6:
                # replace worst if diverse
                pool[-1] = (v, xp.copy())
                pool = sorted(pool, key=lambda t: t[0])[:6]
            else:
                pool.append((v, xp.copy()))
        # early exit if clearly new record by margin? continue to polish further
    # final LP polish on top candidates
    pool = sorted(pool, key=lambda t: t[0])[:4]
    for pi, (pv, px) in enumerate(list(pool)):
        if time.time() > T_END - 2.0:
            break
        try:
            dl = min(T_END - 1.0, time.time() + max(2.0, (T_END - 1.0 - time.time()) / (len(pool) - pi)))
            q = _lp_polish(px, dl)
            vq = _max_cos(q)
            if vq < best_v:
                best_v, best_x = vq, q.copy()
        except Exception:
            pass
    # ultra-final tiny shrink not needed; ensure unit norms
    best_x = _norm(best_x)
    return best_x

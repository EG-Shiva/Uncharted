import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _quench(y, steps, K0, K1, lr0, rng=None):
    # fast projected soft-max repulsion with momentum, annealed K
    n, d = y.shape
    vel = np.zeros_like(y)
    mu = 0.85
    for s in range(steps):
        frac = s / max(1, steps - 1)
        K = K0 * (K1 / K0) ** frac if K0 > 0 else K1
        lr = lr0 * (1.0 - 0.6 * frac)
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        E = np.exp(np.clip(K * (G - gm), -20, 0))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        grad = E @ y
        # tangent projection
        grad -= (np.einsum('ij,ij->i', grad, y)[:, None]) * y
        grad /= rs
        vel = mu * vel + (1.0 - mu) * grad * (K / 100.0 + 1.0)
        # adapt scale: clip
        y = y - lr * vel
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return y

def _void_candidates(fixed, k, ntry, rng):
    # fixed: (n-k,d) remaining points; return k new points far from fixed and each other
    d = fixed.shape[1]
    C = rng.randn(ntry, d)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    # score = max dot with fixed
    D = C @ fixed.T
    s = D.max(axis=1)
    order = np.argsort(s)
    # greedy pick among best to also spread among themselves
    pool = C[order[:max(200, 20*k)]]
    picked = []
    # first = best
    picked.append(pool[0])
    for _ in range(1, k):
        rem = np.stack(picked)  # (p,d)
        dd = pool @ rem.T
        ddmax = dd.max(axis=1)
        # combine with fixed score: rank by max(fixed_score, inter_score)
        # fixed scores for pool:
        fs = (pool @ fixed.T).max(axis=1)
        tot = np.maximum(fs, ddmax)
        # mask already picked (approx by distance)
        # pick min tot not already picked
        for idx in np.argsort(tot):
            v = pool[idx]
            ok = True
            for u in picked:
                if abs(float(v @ u)) > 0.9999:
                    ok = False
                    break
            if ok:
                picked.append(v)
                break
    return np.stack(picked)

def _lp_polish_fast(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < 30:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            idx = np.argpartition(-c[sel], 1500)[:1500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 1500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 168.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (54, 21):
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = _proj(xb)
    best = xb.copy()
    bv = _max_cos(best)
    n, d = 54, 21

    def tl():
        return LIMIT - (time.time() - T0)

    # params for quenches
    Kpairs = [(60, 500), (80, 800), (50, 1200), (100, 600)]
    lrs = [0.02, 0.03, 0.05, 0.08]

    it = 0
    # budget: reserve 14s for final polish
    while tl() > 16 and it < 60:
        it += 1
        mode = it % 6
        Y = best.copy()
        if mode in (0, 3):
            # multi-point void reinsertion: kill 2-3 worst points
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            cnt = (G > m0 - 0.002).sum(axis=1)
            rowmax = G.max(axis=1)
            score = rowmax + 0.002 * cnt
            order = np.argsort(-score)
            k = 3 if (it % 2 == 0) else 2
            kill = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            fixed = best[keep_mask]
            try:
                newpts = _void_candidates(fixed, k, 8000, rng)
            except Exception:
                newpts = rng.randn(k, d)
                newpts /= (np.linalg.norm(newpts, axis=1, keepdims=True) + 1e-12)
            Y[kill] = newpts
            # small jitter on neighbours of killed points
            Y += 0.005 * rng.randn(n, d)
            Y = _proj(Y)
            Kp = Kpairs[it % len(Kpairs)]
            lr = lrs[it % len(lrs)]
            Y2 = _quench(Y, steps=900, K0=Kp[0], K1=Kp[1], lr0=lr)
        elif mode in (1, 4):
            # collective subspace shake: rotate random subset in random 2D plane
            sigma = 0.10 if mode == 1 else 0.22
            Y = Y + sigma * rng.randn(n, d)
            # extra collective rotation: pick 8-12 points, rotate them jointly
            m = 10
            idx = rng.choice(n, size=m, replace=False)
            a = rng.randn(d); a /= (np.linalg.norm(a) + 1e-12)
            b = rng.randn(d); b -= (b @ a) * a; b /= (np.linalg.norm(b) + 1e-12)
            ang = rng.randn() * 0.25
            ca, sa = np.cos(ang), np.sin(ang)
            for ii in idx:
                v = Y[ii]
                pa, pb = v @ a, v @ b
                v = v - pa * a - pb * b + (ca * pa - sa * pb) * a + (sa * pa + ca * pb) * b
                Y[ii] = v
            Y = _proj(Y)
            Kp = Kpairs[(it + 1) % len(Kpairs)]
            lr = lrs[(it + 1) % len(lrs)]
            Y2 = _quench(Y, steps=1000, K0=Kp[0], K1=Kp[1], lr0=lr)
        else:
            # broad Gaussian kick to escape
            sigma = 0.30 if mode == 2 else 0.06
            Y = Y + sigma * rng.randn(n, d)
            Y = _proj(Y)
            Kp = Kpairs[(it + 2) % len(Kpairs)]
            lr = lrs[(it + 2) % len(lrs)]
            Y2 = _quench(Y, steps=800, K0=Kp[0], K1=Kp[1], lr0=lr)
        v = _max_cos(Y2)
        if v < bv:
            best, bv = Y2, v
        # opportunistic small polish if big win and time allows (rare, cheap)
        if bv < 0.0660 and tl() > 30:
            break

    # final squeeze: short LP polish only
    if tl() > 6:
        try:
            best2 = _lp_polish_fast(best, min(12.0, tl() - 2.0))
            v2 = _max_cos(best2)
            if v2 < bv:
                best = best2
        except Exception:
            pass
    best = _proj(best)
    return best

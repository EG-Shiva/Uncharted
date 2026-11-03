import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    n, d = 121, 8
    rng = np.random.default_rng(12345)

    cands0 = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands0.append(a)
        except Exception:
            pass
    if not cands0:
        a = rng.standard_normal((n, d))
        cands0.append(_norm(a))
    v0 = [_max_cos(a) for a in cands0]
    Xbest = cands0[int(np.argmin(v0))].copy()
    Vbest = float(min(v0))

    def anneal(X, steps, k0, k1, lr0):
        X = _norm(X.copy())
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        lk0, lk1 = np.log(k0), np.log(k1)
        for t in range(1, steps + 1):
            if time.time() > DEADLINE - 6.0:
                break
            frac = t / float(steps)
            k = float(np.exp(lk0 + (lk1 - lk0) * frac))
            lr = lr0 * (1.0 - 0.5 * frac)
            G = X @ X.T
            np.fill_diagonal(G, -np.inf)
            gmax = float(G.max())
            # weights
            W = np.exp(k * (G - gmax))
            np.fill_diagonal(W, 0.0)
            s = float(W.sum())
            if not np.isfinite(s) or s <= 0:
                continue
            Grad = (W @ X) / s
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            b1t *= b1; b2t *= b2
            m = b1 * m + (1.0 - b1) * Gt
            vv = b2 * v + (1.0 - b2) * (Gt * Gt)
            v = vv
            mh = m / (1.0 - b1t); vh = v / (1.0 - b2t)
            step = lr * mh / (np.sqrt(vh) + eps)
            step = np.clip(step, -0.03, 0.03)
            X = X - step
            X = _norm(X)
        return _norm(X)

    def hole_grow(base, n_target):
        cur = base.copy()
        m_cur = cur.shape[0]
        for _ in range(n_target - m_cur):
            if time.time() > DEADLINE - 30.0:
                break
            C = rng.standard_normal((20000, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ cur.T
            cm = S.max(axis=1)
            k = int(np.argmin(cm))
            bc = C[k].copy(); curmin = float(cm[k])
            for st in range(60):
                sig = 0.03 * (1.0 - st / 60.0) + 0.002
                prop = bc + sig * rng.standard_normal(d)
                pn = np.linalg.norm(prop)
                if pn == 0:
                    continue
                prop /= pn
                md = float((prop @ cur.T).max())
                if md < curmin:
                    bc = prop; curmin = md
            cur = np.vstack([cur, bc[None, :]])
        # shuffle order
        perm = rng.permutation(cur.shape[0])
        return _norm(cur[perm])

    def greedy_squeeze(X, iters=400, lr=0.002):
        X = _norm(X.copy())
        best = _max_cos(X)
        Xb = X.copy()
        for it in range(iters):
            if time.time() > DEADLINE - 5.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # find worst pair
            flat = int(np.argmax(G))
            ii = flat // n; jj = flat % n
            g = float(G[ii, jj])
            # push apart symmetrically along tangent
            # xi <- xi - lr*(xj - g*xi), same for xj
            xi = X[ii].copy(); xj = X[jj].copy()
            gi = xj - g * xi
            gj = xi - g * xj
            # remove already tangent (gi,g j are tangent)
            X[ii] = xi - lr * gi
            X[jj] = xj - lr * gj
            X[ii] /= max(1e-12, np.linalg.norm(X[ii]))
            X[jj] /= max(1e-12, np.linalg.norm(X[jj]))
            if (it % 25) == 24:
                vv = _max_cos(X)
                if vv < best:
                    best = vv; Xb = X.copy()
                else:
                    # revert partially if worsened a lot
                    pass
        vv = _max_cos(X)
        if vv < best:
            return X, vv
        return Xb, best

    # build starts
    starts = []
    starts.append(Xbest.copy())
    for sig, sd in [(0.02, 11), (0.08, 12), (0.20, 13)]:
        r = np.random.default_rng(sd)
        Xp = Xbest + sig * r.standard_normal((n, d))
        starts.append(_norm(Xp))
    for sd in [21, 22]:
        r = np.random.default_rng(sd)
        starts.append(_norm(r.standard_normal((n, d))))
    # hole-grown: drop worst 6 points then regrow
    try:
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        rm = G0.max(axis=1)
        order = np.argsort(-rm)
        drop = order[:6]
        keep = np.ones(n, dtype=bool); keep[drop] = False
        base = Xbest[keep].copy()
        rj = np.random.default_rng(31)
        base = _norm(base + 0.01 * rj.standard_normal(base.shape))
        hg = hole_grow(base, n)
        starts.append(hg)
    except Exception:
        pass

    pool = []
    # stage 1: short anneal each start
    for si, s0 in enumerate(starts):
        if time.time() > DEADLINE - 35.0:
            break
        steps = 700 if si < 2 else 500
        Xa = anneal(s0, steps=steps, k0=30.0, k1=800.0, lr0=0.010)
        va = _max_cos(Xa)
        pool.append((va, Xa))
        if va < Vbest - 1e-12:
            Vbest = va; Xbest = Xa.copy()

    pool.sort(key=lambda z: z[0])
    # stage 2: promote top 3 to sharp anneal + squeeze
    for pi in range(min(3, len(pool))):
        if time.time() > DEADLINE - 12.0:
            break
        _, Xp = pool[pi]
        Xa = anneal(Xp, steps=1200, k0=100.0, k1=5000.0, lr0=0.006)
        va = _max_cos(Xa)
        Xs, vs = greedy_squeeze(Xa, iters=400, lr=0.002)
        # extra very sharp short anneal
        if time.time() < DEADLINE - 10.0:
            Xs2 = anneal(Xs, steps=400, k0=800.0, k1=8000.0, lr0=0.003)
            v2 = _max_cos(Xs2)
            if v2 < vs:
                Xs, vs = Xs2, v2
        pool.append((vs, Xs))
        if vs < Vbest - 1e-12:
            Vbest = vs; Xbest = Xs.copy()

    # final safety: global best over pool
    for vv, XX in pool:
        c = _max_cos(XX)
        if c < Vbest - 1e-12:
            Vbest = c; Xbest = XX.copy()

    Xbest = _norm(Xbest)
    # tiny shrink guard against boundary rounding (no-op, keeps unit rows)
    return Xbest

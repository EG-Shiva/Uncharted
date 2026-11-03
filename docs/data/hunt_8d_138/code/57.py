import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    X[~np.isfinite(X)] = 0.0
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _build_e8():
    d = 8
    pts = []
    # D8: (+-1,+-1,0...)
    for i in range(d):
        for j in range(i+1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    # A: (+-0.5 ... ) even parity
    for mask in range(256):
        bits = [(mask >> k) & 1 for k in range(8)]
        parity = sum(bits) % 2
        if parity == 0:
            v = np.ones(d) * 0.5
            for k in range(8):
                if bits[k]:
                    v[k] = -0.5
            pts.append(v)
    E = np.array(pts, dtype=float)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E  # 240x8

def _anneal_adam(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    t = 0
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        E = np.exp(beta * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = E.sum() + 1e-300
        Gr = (E @ X) / S
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Tan = Gr - dot * X
        t += 1
        m = 0.9 * m + 0.1 * Tan
        vv = 0.999 * v + 0.001 * Tan * Tan
        v = vv
        mhat = m / (1.0 - 0.9 ** t)
        vhat = v / (1.0 - 0.999 ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + 1e-8)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return _norm(X)

def _lp_polish(X, deadline, radius0=8e-4, floor_slack=3e-4, cap=3000):
    X = _norm(X)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, floor_slack)
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
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _degrees(X, tol):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    deg = (G > (m - tol)).sum(axis=1)
    return deg, m, G

def _find_voids(X_fixed, k_needed, rng, n_samples=15000):
    d = X_fixed.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    cur = X_fixed.copy()
    outs = []
    for _ in range(k_needed):
        smax = (cand @ cur.T).max(axis=1)
        bi = int(np.argmin(smax))
        vv = cand[bi].copy()
        # refine: move away from weighted neighbours
        for tt in range(60):
            dots = cur @ vv
            mv = dots.max()
            W = np.exp(80.0 * (dots - mv))
            W = W / (W.sum() + 1e-300)
            g = W @ cur
            gt = g - np.dot(g, vv) * vv
            vv = vv - 0.06 * gt
            vv = vv / (np.linalg.norm(vv) + 1e-30)
        outs.append(vv)
        cur = np.vstack([cur, vv[None, :]])
    return np.array(outs)

def solve():
    rng = np.random.default_rng(1)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
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
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    try:
        E8 = _build_e8()
    except Exception:
        E8 = None
    # quick baseline polish
    try:
        if time.time() < deadline - 20:
            best = _lp_polish(best, min(deadline - 15, time.time() + 3.0),
                              radius0=1e-3, floor_slack=1e-9, cap=3000)
            best_val = _max_cos(best)
    except Exception:
        pass
    pool = [(best_val, best.copy())]
    it = 0
    while time.time() < deadline - 9.0:
        it += 1
        remain = deadline - 9.0 - time.time()
        if remain < 2.0:
            break
        # choose base cycling through pool
        try:
            base_val, base = pool[(it - 1) % len(pool)]
            base = base.copy()
        except Exception:
            base = best.copy()
        op = it % 5
        X0 = None
        try:
            if op == 4 and E8 is not None:
                # E8 subset restart
                idx = rng.choice(E8.shape[0], size=n, replace=False)
                X0 = E8[idx].copy()
                nz = rng.standard_normal(size=X0.shape)
                proj = (nz * X0).sum(axis=1, keepdims=True)
                nz = nz - proj * X0
                X0 = _norm(X0 + 0.02 * nz)
            elif op == 0:
                deg, m, G = _degrees(base, 0.004)
                order = np.argsort(deg)
                k_low = 6
                low_idx = order[:k_low]
                mask = np.ones(n, dtype=bool)
                mask[low_idx] = False
                X_rem = base[mask]
                voids = _find_voids(X_rem, k_low, rng, n_samples=12000)
                X0 = np.vstack([X_rem, voids])
                X0 = _norm(X0)
            elif op == 1:
                deg, m, G = _degrees(base, 0.003)
                order = np.argsort(-deg)
                hubs = order[:4]
                X0 = base.copy()
                for h in hubs:
                    nz = rng.standard_normal(size=(d,))
                    nz = nz - np.dot(nz, X0[h]) * X0[h]
                    nz = nz / (np.linalg.norm(nz) + 1e-30)
                    sig = 0.30 if (it % 2 == 0) else 0.20
                    X0[h] = _norm((X0[h] + sig * nz)[None, :])[0]
                # also shake neighbours of top hub
                G2 = X0 @ X0.T
                np.fill_diagonal(G2, -2.0)
                nb = np.argsort(-G2[hubs[0]])[:4]
                for h in nb[1:]:
                    nz = rng.standard_normal(size=(d,))
                    nz = nz - np.dot(nz, X0[h]) * X0[h]
                    nz = nz / (np.linalg.norm(nz) + 1e-30)
                    X0[h] = _norm((X0[h] + 0.15 * nz)[None, :])[0]
            elif op == 2:
                X0 = base.copy()
                seed = int(rng.integers(0, n))
                G2 = X0 @ X0.T
                np.fill_diagonal(G2, -2.0)
                neigh = np.argsort(-G2[seed])[:7]
                grp = [seed] + neigh.tolist()
                for h in grp:
                    nz = rng.standard_normal(size=(d,))
                    nz = nz - np.dot(nz, X0[h]) * X0[h]
                    nz = nz / (np.linalg.norm(nz) + 1e-30)
                    X0[h] = _norm((X0[h] + 0.22 * nz)[None, :])[0]
            else:
                X0 = base.copy()
                nz = rng.standard_normal(size=X0.shape)
                proj = (nz * X0).sum(axis=1, keepdims=True)
                nz = nz - proj * X0
                nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-30
                nz = nz / nn
                sig = 0.04 if (it % 2 == 1) else 0.08
                X0 = _norm(X0 + sig * nz)
        except Exception:
            X0 = base.copy()
        try:
            # anneal: steps adapt to remaining time
            if remain > 30:
                steps1, steps2 = 550, 250
            elif remain > 12:
                steps1, steps2 = 400, 150
            else:
                steps1, steps2 = 300, 100
            Xcur = _anneal_adam(X0, steps1, 30.0, 320.0, 0.03, 0.003)
            Xcur = _anneal_adam(Xcur, steps2, 320.0, 500.0, 0.008, 0.002)
            # short LP
            lp_dead = min(time.time() + 1.2, deadline - 8.0)
            if time.time() < lp_dead - 0.2:
                Xcur = _lp_polish(Xcur, lp_dead, radius0=6e-4, floor_slack=3e-4, cap=2500)
            vv = _max_cos(Xcur)
        except Exception:
            continue
        # update pool / best
        try:
            if vv < best_val - 1e-12:
                best_val = vv
                best = Xcur.copy()
                pool.append((vv, Xcur.copy()))
                pool = sorted(pool, key=lambda z: z[0])[:3]
            else:
                # keep diverse near-best for exploration (within 0.002)
                if vv < best_val + 0.002:
                    # avoid duplicates: check min distance to pool
                    pool.append((vv, Xcur.copy()))
                    pool = sorted(pool, key=lambda z: z[0])[:4]
                    # keep best always, drop worst beyond 3 if too many
                    if len(pool) > 3:
                        # keep best + 2 most recent diverse? just keep 3 best
                        pool = sorted(pool, key=lambda z: z[0])[:3]
        except Exception:
            pass
    try:
        if time.time() < deadline - 0.4:
            best = _lp_polish(best, deadline - 0.3, radius0=1e-3, floor_slack=1e-9, cap=3000)
    except Exception:
        pass
    best = _norm(best)
    return best

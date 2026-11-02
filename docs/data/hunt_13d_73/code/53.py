import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    iu0, iu1 = np.triu_indices(n, 1)

    # load incumbent
    Xbest = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best:
                        best = c
                        Xbest = v.copy()
        except Exception:
            pass
    if Xbest is None:
        Xbest = rng.standard_normal(size=(n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        best = _max_cos(Xbest)

    def adam_lse(X0, s0, s1, steps, lr0, lr1, noise=0.0):
        X = X0.copy()
        m = np.zeros((n, d))
        v = np.zeros((n, d))
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        WX = np.empty((n, d))
        for k in range(steps):
            if (k & 127) == 0 and time.time() > SEARCH_END:
                break
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = s0 + (s1 - s0) * frac
            lr = np.exp(l0 + (l1 - l0) * frac)
            G = X @ X.T
            c = G[iu0, iu1]
            mx = float(c.max())
            e = np.exp(s * (c - mx))
            se = float(e.sum())
            if not np.isfinite(se) or se <= 0:
                # fallback to max pair
                w = np.zeros_like(c)
                w[int(np.argmax(c))] = 1.0
            else:
                w = e / se
            WX.fill(0.0)
            np.add.at(WX, iu0, w[:, None] * X[iu1])
            np.add.at(WX, iu1, w[:, None] * X[iu0])
            sp = np.sum(WX * X, axis=1)
            Gt = WX - sp[:, None] * X
            m *= b1
            m += (1 - b1) * Gt
            v *= b2
            v += (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1p)
            vhat = v / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X = X - lr * step
            if noise > 0:
                X += rng.standard_normal(size=(n, d)) * (noise * lr)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def largest_hole(X, n_cand=3000):
        # return point on S^12 minimizing max dot to X
        C = rng.standard_normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ X.T  # (n_cand, n)
        m = S.max(axis=1)
        return C[int(np.argmin(m))].copy()

    # Stage A: long annealed polish of incumbent
    try:
        Xa = adam_lse(Xbest, 80.0, 600.0, 1800, 0.02, 0.002)
        va = _max_cos(Xa)
        if va < best:
            best, Xbest = va, Xa
        # second anneal to higher sharpness
        if time.time() < SEARCH_END - 10:
            Xb = adam_lse(Xbest, 300.0, 1200.0, 1200, 0.008, 0.0012)
            vb = _max_cos(Xb)
            if vb < best:
                best, Xbest = vb, Xb
    except Exception:
        pass

    # Stage B: one fresh random basin (different topology) cheap
    if time.time() < SEARCH_END - 60:
        try:
            Xr = rng.standard_normal(size=(n, d))
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            Xr = adam_lse(Xr, 30.0, 500.0, 2000, 0.03, 0.002)
            vr = _max_cos(Xr)
            if vr < best:
                best, Xbest = vr, Xr
        except Exception:
            pass

    # Stage C: shake + repolish hops
    sigmas = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.015]
    it = 0
    while time.time() < SEARCH_END - 25:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        # occasionally log-uniform
        if it % 5 == 0:
            sigma = float(10 ** rng.uniform(-3.3, -1.9))
        X0 = Xbest + sigma * rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        s0 = float(rng.choice([60.0, 100.0, 150.0]))
        s1 = float(rng.choice([600.0, 900.0, 1300.0]))
        steps = int(rng.choice([700, 900, 1200]))
        lr0 = float(10 ** rng.uniform(-1.9, -1.4))
        lr1 = float(10 ** rng.uniform(-3.0, -2.4))
        try:
            Y = adam_lse(X0, s0, s1, steps, lr0, lr1)
            vy = _max_cos(Y)
            if vy < best:
                best, Xbest = vy, Y.copy()
        except Exception:
            pass
        # periodic hole-reinsertion attempt on worst point
        if it % 3 == 0 and time.time() < SEARCH_END - 30:
            try:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                # worst point = max row max
                rm = G.max(axis=1)
                wi = int(np.argmax(rm))
                hole = largest_hole(Xbest, n_cand=2500)
                Xt = Xbest.copy()
                Xt[wi] = hole
                # quick polish
                Y2 = adam_lse(Xt, 200.0, 900.0, 600, 0.006, 0.001)
                v2 = _max_cos(Y2)
                if v2 < best:
                    best, Xbest = v2, Y2.copy()
                else:
                    # try moving one endpoint of global max pair instead
                    flat = int(np.argmax(G[iu0, iu1]))
                    a, b = int(iu0[flat]), int(iu1[flat])
                    for cand_idx in (a, b):
                        Xt2 = Xbest.copy()
                        Xt2[cand_idx] = largest_hole(Xbest, n_cand=1500)
                        Y3 = adam_lse(Xt2, 200.0, 800.0, 400, 0.006, 0.001)
                        v3 = _max_cos(Y3)
                        if v3 < best:
                            best, Xbest = v3, Y3.copy()
                            break
                        if time.time() > SEARCH_END - 25:
                            break
            except Exception:
                pass
        if it > 40:
            break

    # Stage D: final high-sharpness squeeze
    if time.time() < HARD_END - 20:
        try:
            Y = adam_lse(Xbest, 800.0, 2000.0, 900, 0.004, 0.0006)
            vy = _max_cos(Y)
            if vy < best:
                best, Xbest = vy, Y
        except Exception:
            pass

    # Stage E: tiny trust-region LP squeeze (time-guarded, capped pairs)
    if time.time() < HARD_END - 12:
        try:
            X = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
            cur = _max_cos(X)
            radius = 4e-4
            lp_start = time.time()
            while time.time() - lp_start < 10.0 and time.time() < HARD_END - 3 and radius > 1e-14:
                Cfull = X @ X.T
                c = Cfull[iu0, iu1]
                m = float(c.max())
                sel = c > m - max(20.0 * radius, 1.5e-7)
                if np.count_nonzero(sel) > 1800:
                    idx = np.where(sel)[0]
                    order = np.argsort(-c[idx])[:1800]
                    mask = np.zeros_like(sel)
                    mask[idx[order]] = True
                    sel = mask
                I, J = iu0[sel], iu1[sel]
                p = len(I)
                if p == 0:
                    radius *= 0.5
                    continue
                csel = c[sel]
                nv = n * d + 1
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cI = I[:, None] * d + np.arange(d)
                cJ = J[:, None] * d + np.arange(d)
                ct = np.full((p, 1), n * d)
                cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
                if res.status != 0:
                    radius *= 0.5
                    continue
                y = X + res.x[:-1].reshape(n, d)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                vv = _max_cos(y)
                if vv < cur:
                    X, cur = y, vv
                    if cur < best:
                        best, Xbest = cur, X.copy()
                    radius = min(radius * 1.5, 0.01)
                else:
                    radius *= 0.5
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    Xbest = np.where(np.isfinite(Xbest), Xbest, 1.0 / np.sqrt(13))
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest
